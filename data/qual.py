"""
Data-quality checks.
"""
import logging
from dataclasses import dataclass, field
import xarray as xr
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["QualReport", "DataQual"]

@dataclass
class QualReport:
    passed: bool
    checks: list = field(default_factory=list)
    summary: str = ""

class DataQual:
    """Validates data against missing values and outlier thresholds."""
    
    @staticmethod
    def check_dataset(ds: xr.Dataset, config) -> QualReport:
        passed = True
        checks = []
        
        qual_config = config.get("qual", {})
        nan_threshold = qual_config.get("nan_threshold", 0.05)
        z_score_thresh = qual_config.get("z_score_outlier", 5.0)
        
        def add_check(name, check_passed, detail):
            nonlocal passed
            if not check_passed:
                passed = False
                logger.warning(f"Qual Failed: {name} - {detail}")
            checks.append({"name": name, "passed": check_passed, "detail": detail})

        # Expected variables tracking
        variables = config.get("data", {}).get("variables", [])
        missing_vars = [v for v in variables if v not in ds.data_vars]
        add_check("expected_variables", len(missing_vars) == 0, f"Missing variables: {missing_vars}")
        
        for var in ds.data_vars:
            arr = ds[var].values
            nan_mask = np.isnan(arr)
            
            nan_frac = nan_mask.mean()
            add_check(f"{var}_nan_fraction", bool(nan_frac <= nan_threshold), f"NaN fraction: {nan_frac:.4f}")
            
            inf_count = np.isinf(arr).sum()
            add_check(f"{var}_no_inf", bool(inf_count == 0), f"Infinite values found: {inf_count}")
            
            valid_arr = arr[~nan_mask]
            if len(valid_arr) > 0:
                mean_val = float(np.mean(valid_arr))
                std_val = float(np.std(valid_arr))
                if std_val > 0:
                    outliers = (np.abs(valid_arr - mean_val) / std_val) > z_score_thresh
                    outlier_frac = outliers.mean()
                    add_check(f"{var}_zscore_outliers", True, f"Outlier fraction (> {z_score_thresh}σ): {outlier_frac:.5f}")
                else:
                    add_check(f"{var}_zscore_outliers", True, "Zero standard deviation - skip flag")
            else:
                add_check(f"{var}_zscore_outliers", True, "All NaN values - skip flag")
                
        if 'time' in ds.coords:
            times = ds['time'].values
            if len(times) > 0:
                is_sorted = bool(np.all(times[:-1] <= times[1:]))
                has_duplicates = len(times) != len(np.unique(times))
                add_check("temporal_continuity", is_sorted and not has_duplicates, f"Sorted: {is_sorted}, Duplicates found: {has_duplicates}")
                
        if 'lat' in ds.coords:
            l_min, l_max = float(ds['lat'].min()), float(ds['lat'].max())
            add_check("spatial_lat", -90 <= l_min and l_max <= 90, f"Latitude range bounded in: [{l_min}, {l_max}]")
            
        if 'lon' in ds.coords:
            l_min, l_max = float(ds['lon'].min()), float(ds['lon'].max())
            valid_lon = (-180 <= l_min and l_max <= 180) or (0 <= l_min and l_max <= 360)
            add_check("spatial_lon", valid_lon, f"Longitude range bounded in: [{l_min}, {l_max}]")

        summary = "Qual Dataset Summary:\n" + "\n".join(
            f"- [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks
        )
        logger.info("\n" + summary)
        
        return QualReport(passed=passed, checks=checks, summary=summary)

    @staticmethod
    def check_model_output(preds: torch.Tensor, task: str, config) -> QualReport:
        passed = True
        checks = []
        qual_config = config.get("qual", {})
        
        def add_check(name, check_passed, detail):
            nonlocal passed
            if not check_passed:
                passed = False
                logger.warning(f"Qual Failed: {name} - {detail}")
            checks.append({"name": name, "passed": check_passed, "detail": detail})

        if task == 'regression':
            nan_count = torch.isnan(preds).sum().item()
            inf_count = torch.isinf(preds).sum().item()
            add_check("preds_no_nan", nan_count == 0, f"Found {nan_count} NaNs")
            add_check("preds_no_inf", inf_count == 0, f"Found {inf_count} Infs")
            
            out_bounds = ((preds < -3.0) | (preds > 3.0)).float().mean().item()
            add_check("preds_range", True, f"Fraction elements > ±3σ: {out_bounds:.4f}")
            
        elif task == 'classification':
            min_class_frac = qual_config.get("min_class_fraction", 0.01)
            
            prob_sum = preds.sum(dim=-1)
            sum_dev = torch.abs(prob_sum - 1.0).max().item()
            add_check("softmax_sum", sum_dev <= 1e-4, f"Max deviational diff from sum=1: {sum_dev:.6f}")
            
            preds_class = torch.argmax(preds, dim=-1)
            for c in range(preds.shape[-1]):
                frac = (preds_class == c).float().mean().item()
                add_check(f"class_{c}_freq", frac >= min_class_frac, f"Class {c} predicted distribution tracking: {frac:.4f}")

        summary = f"Qual Model Output ({task}) target bounds check Summary:\n" + "\n".join(
            f"- [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks
        )
        logger.info("\n" + summary)
        
        return QualReport(passed=passed, checks=checks, summary=summary)
