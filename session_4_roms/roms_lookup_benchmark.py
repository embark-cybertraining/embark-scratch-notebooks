"""Benchmark OOI profiler lookup and ROMS point-versus-region requests."""

from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from erddapy import ERDDAP


ERDDAP_SERVER = "https://erddap.dataexplorer.oceanobservatories.org/erddap"
DOPPIO_URL = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best"

PROFILERS = (
	"CP12CNSW-WFP01-01-CTDPFP000",
	"CP12WESW-WFP01-01-CTDPFP000",
	"CP13EAPM-WFP01-03-CTDPFK000",
	"CP13NOPM-WFP01-03-CTDPFK000",
	"CP13SOPM-WFP01-03-CTDPFK000",
	"CP14NEPM-WFP01-03-CTDPFK000",
	"CP14SEPM-WFP01-03-CTDPFK000",
)

TEMPERATURE = "sea_water_temperature_profiler_depth_enabled"
SALINITY = "sea_water_practical_salinity_profiler_depth_enabled"
ROMS_VARIABLES = ("temp", "salt")

START = pd.Timestamp("2024-06-20")
END = pd.Timestamp("2024-06-23")
REGION_PADDING = 1
MAX_REGION_GRID_CELLS = 2500


def load_ooi_sample(instrument_id: str) -> pd.DataFrame:
	"""Fetch a short ERDDAP sample to estimate site location and data coverage."""
	erddap = ERDDAP(server=ERDDAP_SERVER, protocol="tabledap")
	erddap.dataset_id = f"ooi-{instrument_id.lower()}"
	erddap.variables = ["time", "latitude", "longitude", "z", TEMPERATURE, SALINITY]
	erddap.constraints = {
		"time>=": START.strftime("%Y-%m-%dT%H:%M:%SZ"),
		"time<=": (END - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
	}
	frame = erddap.to_pandas(
		response="csv",
		parse_dates=True,
		skiprows=[1],
		index_col="time",
	).dropna(how="all")
	frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert(None)
	return frame


def find_nearest_wet_point(
	lon_grid: xr.DataArray,
	lat_grid: xr.DataArray,
	 wet_mask: xr.DataArray,
	target_lon: float,
	target_lat: float,
) -> tuple[int, int]:
	"""Find the nearest wet ROMS rho point using local lon/lat distance."""
	distance_squared = (lon_grid - target_lon) ** 2 + (lat_grid - target_lat) ** 2
	distance_squared = distance_squared.where(wet_mask == 1)
	eta_index, xi_index = np.unravel_index(
		np.nanargmin(distance_squared.values), distance_squared.shape
	)
	return int(eta_index), int(xi_index)


def timed_point_loads(
	ds_roms: xr.Dataset,
	points: dict[str, tuple[int, int]],
	variables: tuple[str, ...],
	start: str,
	end: str,
) -> float:
	"""Load the requested variables separately at each profiler's grid point."""
	started = perf_counter()
	for eta_index, xi_index in points.values():
		for variable in variables:
			ds_roms[variable].sel(
				time=slice(start, end), eta_rho=eta_index, xi_rho=xi_index
			).load()
	return perf_counter() - started


def timed_combined_point_loads(
	ds_roms: xr.Dataset,
	points: dict[str, tuple[int, int]],
	start: str,
	end: str,
) -> float:
	"""Load temperature and salinity together for each profiler point."""
	started = perf_counter()
	for eta_index, xi_index in points.values():
		ds_roms[list(ROMS_VARIABLES)].sel(
			time=slice(start, end), eta_rho=eta_index, xi_rho=xi_index
		).load()
	return perf_counter() - started


def timed_region_load(
	ds_roms: xr.Dataset,
	points: dict[str, tuple[int, int]],
	start: str,
	end: str,
) -> tuple[float, int, tuple[int, int]]:
	"""Load both variables once for the padded rectangle enclosing all points."""
	eta_values = [point[0] for point in points.values()]
	xi_values = [point[1] for point in points.values()]
	eta_start = max(0, min(eta_values) - REGION_PADDING)
	eta_stop = min(ds_roms.sizes["eta_rho"], max(eta_values) + REGION_PADDING + 1)
	xi_start = max(0, min(xi_values) - REGION_PADDING)
	xi_stop = min(ds_roms.sizes["xi_rho"], max(xi_values) + REGION_PADDING + 1)
	region_shape = (eta_stop - eta_start, xi_stop - xi_start)
	region_cells = region_shape[0] * region_shape[1]

	if region_cells > MAX_REGION_GRID_CELLS:
		raise ValueError(
			f"Bounding region is {region_shape[0]}x{region_shape[1]} "
			f"({region_cells} grid cells), above the {MAX_REGION_GRID_CELLS}-cell limit."
		)

	started = perf_counter()
	ds_roms[list(ROMS_VARIABLES)].sel(time=slice(start, end)).isel(
		eta_rho=slice(eta_start, eta_stop), xi_rho=slice(xi_start, xi_stop)
	).load()
	return perf_counter() - started, region_cells, region_shape


def main() -> None:
	start = START.strftime("%Y-%m-%d")
	end = (END - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
	locations: dict[str, tuple[float, float]] = {}

	print(f"Fetching OOI samples from {START:%Y-%m-%d} through {END:%Y-%m-%d} (end exclusive)")
	for instrument_id in PROFILERS:
		try:
			frame = load_ooi_sample(instrument_id)
		except Exception as error:
			print(f"{instrument_id}: ERDDAP lookup failed: {error}")
			continue

		usable = frame[[TEMPERATURE, SALINITY]].dropna(how="all")
		if usable.empty or frame[["latitude", "longitude"]].dropna().empty:
			print(f"{instrument_id}: no sample data in this window; skipping")
			continue

		location = frame[["latitude", "longitude"]].dropna().mean()
		observed_depth = -frame["z"].dropna()
		print(
			f"{instrument_id}: lat={location['latitude']:.5f}, "
			f"lon={location['longitude']:.5f}, "
			f"observed depth={observed_depth.min():.1f}-{observed_depth.max():.1f} m, "
			f"usable sample days={usable.index.normalize().nunique()}"
		)
		locations[instrument_id] = (float(location["latitude"]), float(location["longitude"]))

	if not locations:
		print("No profiler locations were found; nothing to benchmark.")
		return

	print("Opening ROMS metadata and locating wet grid points...")
	ds_roms = xr.open_dataset(DOPPIO_URL)
	lon_grid = ds_roms["lon_rho"]
	lat_grid = ds_roms["lat_rho"]
	wet_mask = ds_roms["mask_rho"]
	grid_points: dict[str, tuple[int, int]] = {}
	for instrument_id, (latitude, longitude) in locations.items():
		grid_point = find_nearest_wet_point(
			lon_grid, lat_grid, wet_mask, longitude, latitude
		)
		grid_points[instrument_id] = grid_point
		eta_index, xi_index = grid_point
		depth_m = float(ds_roms["h"].isel(eta_rho=eta_index, xi_rho=xi_index).values)
		print(f"{instrument_id}: eta={eta_index}, xi={xi_index}, bathymetry={depth_m:.1f} m")

	print(f"\nBenchmarking ROMS data from {start} through {end}")
	print("Timing separate temperature/salinity point loads...")
	separate_seconds = timed_point_loads(
		ds_roms, grid_points, ROMS_VARIABLES, start, end
	)
	print("Timing combined temperature/salinity point loads...")
	combined_seconds = timed_combined_point_loads(ds_roms, grid_points, start, end)
	print(f"Separate temp/salt point loads: {separate_seconds:.2f} s")
	print(f"Combined temp/salt per-point loads: {combined_seconds:.2f} s")

	try:
		print("Timing combined regional load...")
		region_seconds, region_cells, region_shape = timed_region_load(
			ds_roms, grid_points, start, end
		)
	except ValueError as error:
		print(f"Combined regional load skipped: {error}")
	else:
		print(
			f"Combined regional load ({region_shape[0]}x{region_shape[1]} "
			f"= {region_cells} grid cells): {region_seconds:.2f} s"
		)

	ds_roms.close()


if __name__ == "__main__":
	main()