"""
Data transformations: unit homogenisation and normalisation.
"""
import xarray as xr
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["homogenise_units", "compute_global_stats", "normalise"]

def homogenise_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Handles unit conversions across meteorological variables.
    """
    ds_out = ds.copy()
    
    if 'sp' in ds_out.data_vars:
        sp_max = float(ds_out['sp'].max())
        if sp_max > 10000:
            ds_out['sp'] = ds_out['sp'] / 100.0
            logger.info("Converted surface pressure 'sp' from Pa to hPa (divided by 100)")
            
    if 'tp' in ds_out.data_vars:
        ds_out['tp'] = np.log1p(ds_out['tp'])
        logger.info("Applied log1p transform to precipitation 'tp'")
        
    return ds_out

def compute_global_stats(ds: xr.Dataset, variables: list) -> dict[str, tuple[float, float]]:
    """
    Returns {var_name: (mean, std)} computed over all space and time.
    """
    stats = {}
    for var in variables:
        if var in ds.data_vars:
            mean_val = float(ds[var].mean(skipna=True))
            std_val = float(ds[var].std(skipna=True))
            stats[var] = (mean_val, std_val)
    return stats

def normalise(ds: xr.Dataset, stats: dict) -> xr.Dataset:
    """
    Applies standardization or min-max normalization to raw features.
    """
    for var, (mean_val, std_val) in stats.items():
        if var in ds.data_vars:
            if std_val > 0.0:
                ds[var] = (ds[var] - mean_val) / std_val
            else:
                ds[var] = ds[var] - mean_val
            
            # Outlier guard
            ds[var] = ds[var].clip(-10.0, 10.0)
    return ds
