"""
NetCDF/xarray data loading functionality.
"""
import os
import xarray as xr

__all__ = ["load_weather_dataset"]

def load_weather_dataset(config) -> xr.Dataset:
    """
    Opens NetCDF files for all variables listed in config.data.variables using xarray.
    Handles multiple files via xr.open_mfdataset with concat_dim='time'.
    Returns a single Dataset with dims (time, lat, lon).
    """
    raw_path = config["data"]["raw_path"]
    file_pattern = os.path.join(raw_path, "*.nc")
    
    ds = xr.open_mfdataset(
        file_pattern,
        engine='netcdf4',
        concat_dim='time',
        combine='nested'
    )
    
    variables = config["data"]["variables"]
    present_vars = [v for v in variables if v in ds.data_vars]
    ds = ds[present_vars]
    
    return ds
