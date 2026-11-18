"""Make monthly OOI profiler and ROMS Doppio plots on OOI JupyterHub."""

from pathlib import Path
import re
from time import perf_counter

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


KDATA_ROOT = Path("/home/jovyan/ooi/kdata")
OUTPUT_ROOT = Path("roms_comparison_output")
CACHE_ROOT = Path("roms_cache")
DOPPIO_URL = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best"

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2026-12-31")
MINIMUM_COVERAGE = 0.50
TIME_BIN = "3h"
DEPTH_BIN_METERS = 1.0
REGION_PADDING = 1

PLOT_VARIABLES = ("temperature", "salinity")
VARIABLES = {
	"temperature": {
		"ooi": "sea_water_temperature",
		"roms": "temp",
		"label": "Temperature (°C)",
		"cmap": "plasma",
	},
	"salinity": {
		"ooi": "sea_water_practical_salinity",
		"roms": "salt",
		"label": "Practical salinity",
		"cmap": "viridis",
	},
}

# Coordinates are fixed profiler locations; eta/xi are the selected ROMS points.
# Adjust eta/xi here if a nearby model point better matches the profiler depth.
PROFILERS = [
	{
		"id": "CP12CNSW-WFP01-01-CTDPFP000",
		"lat": 35.94550,
		"lon": -75.12493,
		"eta": 45,
		"xi": 38,
	},
	{
		"id": "CP12WESW-WFP01-01-CTDPFP000",
		"lat": 35.94552,
		"lon": -75.33332,
		"eta": 46,
		"xi": 36,
	},
	{
		"id": "CP13EAPM-WFP01-03-CTDPFK000",
		"lat": 35.94533,
		"lon": -74.84538,
		"eta": 43, # closest point is 42
		"xi": 41,
	},
	{
		"id": "CP13NOPM-WFP01-03-CTDPFK000",
		"lat": 36.17033,
		"lon": -74.82665,
		"eta": 45,
		"xi": 43,
	},
	{
		"id": "CP13SOPM-WFP01-03-CTDPFK000",
		"lat": 35.72697,
		"lon": -74.85640,
		"eta": 41, # closest point is 40
		"xi": 39,
	},
	{
		"id": "CP14NEPM-WFP01-03-CTDPFK000",
		"lat": 36.04887,
		"lon": -74.77860,
		"eta": 43,
		"xi": 43,
	},
	{
		"id": "CP14SEPM-WFP01-03-CTDPFK000",
		"lat": 35.84725,
		"lon": -74.84667,
		"eta": 41,
		"xi": 40,
	},
]


def open_profiler(instrument_id: str) -> xr.Dataset:
	"""Open recovered files when available, otherwise use telemetered files."""
	request_end = END_DATE + pd.Timedelta(days=1)
	selected_files = []
	selected_source = None
	for source in ("recovered", "telemetered"):
		discovery_start = perf_counter()
		all_files = sorted(KDATA_ROOT.glob(f"{instrument_id}-{source}*/*.nc"))
		discovery_seconds = perf_counter() - discovery_start
		if not all_files:
			continue

		filter_start = perf_counter()
		files = []
		for path in all_files:
			match = re.search(
				r"_(\d{8}T\d{6}(?:\.\d+)?)-(\d{8}T\d{6}(?:\.\d+)?)\.nc$",
				path.name,
			)
			if match is None:
				files.append(path)
				continue

			file_start, file_end = [
				pd.to_datetime(
					value,
					format="%Y%m%dT%H%M%S.%f" if "." in value else "%Y%m%dT%H%M%S",
				)
				for value in match.groups()
			]
			if file_end >= START_DATE and file_start < request_end:
				files.append(path)
		filter_seconds = perf_counter() - filter_start
		print(
			f"{instrument_id}: {source} {len(files)}/{len(all_files)} files match dates "
			f"(list {discovery_seconds:.2f}s, filter {filter_seconds:.2f}s)"
		)
		if files:
			selected_files = files
			selected_source = source
			break

	if not selected_files:
		raise FileNotFoundError(
			f"No recovered or telemetered files overlap the configured dates for {instrument_id}"
		)

	open_start = perf_counter()
	dataset = xr.open_mfdataset(
		selected_files,
		combine="nested",
		concat_dim="obs",
	)
	dataset = dataset.swap_dims({"obs": "time"})
	print(
		f"  {selected_source} xarray open/index: {perf_counter() - open_start:.2f}s; "
		f"{dataset.sizes['time']:,} observations"
	)
	return dataset


def get_month_window(month: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp, int]:
	"""Return this month's requested date window and its number of days."""
	month_start = month.start_time
	month_end = (month + 1).start_time - pd.Timedelta(nanoseconds=1)
	start = max(START_DATE, month_start)
	end = min(END_DATE + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1), month_end)
	days = (end.normalize() - start.normalize()).days + 1
	return start, end, days


def count_valid_days(values: xr.DataArray) -> int:
	"""Count days with at least one non-missing sample."""
	valid = values.notnull()
	for dimension in values.dims:
		if dimension != "time":
			valid = valid.any(dim=dimension)
	valid_times = pd.DatetimeIndex(values.time.where(valid, drop=True).values)
	return int(valid_times.normalize().nunique())


def make_time_grid(dataset: xr.Dataset, variable: str) -> xr.DataArray:
	"""Average observations into configured time and one-meter depth bins."""
	data = dataset[[variable, "depth"]]
	frame = data.to_dataframe().reset_index()
	frame = frame.dropna(subset=["time", "depth"])
	frame["time_bin"] = frame["time"].dt.floor(TIME_BIN)
	frame["depth_bin"] = (
		np.floor(frame["depth"] / DEPTH_BIN_METERS) * DEPTH_BIN_METERS
		+ DEPTH_BIN_METERS / 2
	)

	binned = frame.groupby(["time_bin", "depth_bin"])[variable].mean().unstack("depth_bin")
	first_bin = frame["time_bin"].min()
	last_bin = frame["time_bin"].max()
	time_bins = pd.date_range(first_bin, last_bin, freq=TIME_BIN)
	binned = binned.reindex(time_bins)
	binned.index.name = "time"
	binned.columns.name = "depth_bins"
	return xr.DataArray(
		binned.to_numpy(),
		dims=("time", "depth_bins"),
		coords={"time": time_bins, "depth_bins": binned.columns.to_numpy()},
		name=variable,
	)


def calculate_roms_depth(model: xr.Dataset, bathymetry: xr.DataArray) -> xr.DataArray:
	"""Convert ROMS s-coordinates to time-varying, positive-down depth."""
	zeta = model["zeta"]
	hc = model["hc"]
	s_rho = model["s_rho"]
	cs_r = model["Cs_r"]
	vtransform = int(model["Vtransform"])

	if vtransform == 1:
		z0 = hc * (s_rho - cs_r) + bathymetry * cs_r
		z = z0 + zeta * (1 + z0 / bathymetry)
	elif vtransform == 2:
		z0 = (hc * s_rho + bathymetry * cs_r) / (hc + bathymetry)
		z = zeta + (zeta + bathymetry) * z0
	else:
		raise ValueError(f"Unsupported ROMS Vtransform: {vtransform}")

	return (-z).transpose("time", "s_rho")


def get_color_limits(observed: xr.DataArray, modeled: xr.DataArray) -> tuple[float, float]:
	"""Use common finite color limits for each observed/model variable pair."""
	values = np.concatenate((observed.values.ravel(), modeled.values.ravel()))
	values = values[np.isfinite(values)]
	if not values.size:
		return 0.0, 1.0
	minimum, maximum = float(values.min()), float(values.max())
	if minimum == maximum:
		return minimum - 0.01, maximum + 0.01
	return minimum, maximum


def load_monthly_roms(
	dataset: xr.Dataset,
	month: pd.Period,
	eta_bounds: tuple[int, int],
	xi_bounds: tuple[int, int],
) -> xr.Dataset:
	"""Load one monthly regional ROMS subset, reusing its local cache if present."""
	CACHE_ROOT.mkdir(parents=True, exist_ok=True)
	cache_name = f"{month}_e{eta_bounds[0]}-{eta_bounds[1]}_x{xi_bounds[0]}-{xi_bounds[1]}.nc"
	cache_path = CACHE_ROOT / cache_name
	if cache_path.exists():
		cache_start = perf_counter()
		monthly = xr.open_dataset(cache_path).load()
		print(f"  ROMS cache read: {monthly.nbytes / 1e6:.1f} MB in {perf_counter() - cache_start:.2f}s")
		return monthly

	roms_variables = [VARIABLES[name]["roms"] for name in PLOT_VARIABLES]
	roms_variables.extend(["zeta", "hc", "s_rho", "Cs_r", "Vtransform"])
	start, end, _ = get_month_window(month)
	print(
		f"  Requesting ROMS region for {month} "
		f"({eta_bounds[1] - eta_bounds[0]}x{xi_bounds[1] - xi_bounds[0]} cells)"
	)
	request_start = perf_counter()
	monthly = dataset[roms_variables].sel(time=slice(start, end)).isel(
		eta_rho=slice(*eta_bounds),
		xi_rho=slice(*xi_bounds),
	).load()
	print(f"  ROMS THREDDS load: {monthly.nbytes / 1e6:.1f} MB in {perf_counter() - request_start:.2f}s")
	cache_start = perf_counter()
	monthly.encoding.pop("unlimited_dims", None)
	monthly.to_netcdf(cache_path)
	print(f"  ROMS cache write: {perf_counter() - cache_start:.2f}s ({cache_path.name})")
	return monthly


def plot_month(
	month: pd.Period,
	profiler: dict,
	ooi_grids: dict[str, xr.DataArray],
	model: xr.Dataset,
	bathymetry: xr.DataArray,
) -> None:
	"""Plot OOI and ROMS profiles for each variable selected at the top."""
	model_depth = calculate_roms_depth(model, bathymetry)
	model_depth = model_depth.resample(time=TIME_BIN).mean()
	figure, axes = plt.subplots(
		len(PLOT_VARIABLES),
		2,
		figsize=(15, 4 * len(PLOT_VARIABLES)),
		sharex=True,
		sharey=True,
		squeeze=False,
		constrained_layout=True,
	)

	for row, variable_name in enumerate(PLOT_VARIABLES):
		settings = VARIABLES[variable_name]
		observed = ooi_grids[variable_name]
		modeled = model[settings["roms"]].resample(time=TIME_BIN).mean()
		modeled = modeled.assign_coords(depth=model_depth)
		minimum, maximum = get_color_limits(observed, modeled)
		plot_options = {
			"cmap": settings["cmap"],
			"vmin": minimum,
			"vmax": maximum,
			"add_colorbar": False,
		}
		mesh = observed.plot.pcolormesh(
			x="time",
			y="depth_bins",
			ax=axes[row, 0],
			**plot_options,
		)
		modeled.plot.pcolormesh(
			x="time",
			y="depth",
			ax=axes[row, 1],
			**plot_options,
		)
		axes[row, 0].set_title(f"OOI {variable_name}")
		axes[row, 1].set_title(f"ROMS {variable_name}")
		axes[row, 0].set_ylabel("Depth (m)")
		axes[row, 1].set_ylabel("Depth (m)")
		figure.colorbar(mesh, ax=axes[row, :], label=settings["label"], shrink=0.95)

	depth_bottom = float(model_depth.max(skipna=True))
	axes[0, 0].set_ylim(depth_bottom, 0)

	for axis in axes.flat:
		axis.set_xlabel("")
	for axis in axes[-1, :]:
		axis.xaxis.set_major_locator(mdates.DayLocator(interval=4))
		axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

	figure.suptitle(f"OOI vs ROMS Doppio: {profiler['id']} ({month})")
	output_dir = OUTPUT_ROOT / profiler["id"]
	output_dir.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_dir / f"{month}.png", dpi=150)
	plt.close(figure)
	print(f"  Saved plot: {output_dir / f'{month}.png'}")


def main() -> None:
	run_start = perf_counter()
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	months = pd.period_range(START_DATE.to_period("M"), END_DATE.to_period("M"), freq="M")
	print(f"Comparisons: {START_DATE:%Y-%m-%d} through {END_DATE:%Y-%m-%d}")
	print(f"Variables: {', '.join(PLOT_VARIABLES)}; coverage threshold: {MINIMUM_COVERAGE:.0%}")
	print(f"Reading OOI data from {KDATA_ROOT}")
	roms = xr.open_dataset(DOPPIO_URL)
	profiler_data = {}
	try:
		eta_values = [profiler["eta"] for profiler in PROFILERS]
		xi_values = [profiler["xi"] for profiler in PROFILERS]
		eta_bounds = (
			max(0, min(eta_values) - REGION_PADDING),
			max(eta_values) + REGION_PADDING + 1,
		)
		xi_bounds = (
			max(0, min(xi_values) - REGION_PADDING),
			max(xi_values) + REGION_PADDING + 1,
		)
		print(f"Fixed ROMS region: {eta_bounds[1] - eta_bounds[0]}x{xi_bounds[1] - xi_bounds[0]} grid cells")

		profiler_open_start = perf_counter()
		for profiler in PROFILERS:
			try:
				print(f"Opening OOI files for {profiler['id']}...")
				profiler_data[profiler["id"]] = open_profiler(profiler["id"])
				print(f"  Loaded {profiler_data[profiler['id']].sizes['time']:,} observations")
			except (OSError, FileNotFoundError) as error:
				print(f"{profiler['id']}: {error}")
		print(f"Profiler open/index total: {perf_counter() - profiler_open_start:.2f}s")

		plot_count = 0
		for month in months:
			print(f"\n{month}: checking OOI coverage...")
			coverage_start = perf_counter()
			start, end, days_in_window = get_month_window(month)
			eligible = {}
			for profiler in PROFILERS:
				instrument_id = profiler["id"]
				if instrument_id not in profiler_data:
					continue
				profiler_dataset = profiler_data[instrument_id]
				in_month = (profiler_dataset.time >= start) & (profiler_dataset.time <= end)
				dataset = profiler_dataset[
					[VARIABLES[name]["ooi"] for name in PLOT_VARIABLES] + ["depth"]
				].where(in_month, drop=True).load()
				if not dataset.sizes.get("time", 0):
					continue

				valid_days = [
					count_valid_days(dataset[VARIABLES[name]["ooi"]])
					for name in PLOT_VARIABLES
				]
				threshold = int(np.ceil(days_in_window * MINIMUM_COVERAGE))
				if max(valid_days, default=0) < threshold:
					continue
				if not np.isfinite(float(dataset.depth.max(skipna=True))):
					continue
				eligible[instrument_id] = (profiler, dataset)

			if not eligible:
				print(
					f"  No profiler meets coverage; skipping ROMS "
					f"(OOI selection/check {perf_counter() - coverage_start:.2f}s)"
				)
				continue

			print(
				f"  {len(eligible)} profiler(s) meet coverage "
				f"(OOI selection/check {perf_counter() - coverage_start:.2f}s)"
			)
			model_month = load_monthly_roms(roms, month, eta_bounds, xi_bounds)
			for profiler, dataset in eligible.values():
				plot_start = perf_counter()
				eta_index, xi_index = profiler["eta"], profiler["xi"]
				model = model_month.isel(
					eta_rho=eta_index - eta_bounds[0],
					xi_rho=xi_index - xi_bounds[0],
				)
				ooi_grids = {
					name: make_time_grid(dataset, VARIABLES[name]["ooi"])
					for name in PLOT_VARIABLES
				}
				bathymetry = roms.h.isel(eta_rho=eta_index, xi_rho=xi_index)
				plot_month(month, profiler, ooi_grids, model, bathymetry)
				print(f"  Plot processing: {perf_counter() - plot_start:.2f}s")
				plot_count += 1
				dataset.close()
			model_month.close()
		print(f"Finished: wrote {plot_count} comparison plot(s) under {OUTPUT_ROOT}")
	finally:
		for dataset in profiler_data.values():
			dataset.close()
		roms.close()
		print(f"Total run time: {perf_counter() - run_start:.2f}s")


if __name__ == "__main__":
	main()