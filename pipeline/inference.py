"""
Chained Stage 1 -> Stage 2 forward pass inference.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import xarray as xr
import numpy as np

from models.weather_model import WeatherForecastModel
from models.drought_model import DroughtClassifier
from data.qual import DataQual

__all__ = ["DroughtForecastPipeline"]

class DroughtForecastPipeline:
    def __init__(self, weather_model, drought_model, edge_index, config):
        self.weather_model = weather_model
        self.drought_model = drought_model
        self.edge_index = edge_index
        self.config = config
        
        self.device = next(self.weather_model.parameters()).device
        self.weather_model.eval()
        self.drought_model.eval()
        
    def forecast(self, x_history: torch.Tensor) -> dict:
        """
        Input:  x_history shape (N, T, V)  — recent past time steps window
        """
        # Range validations limits mappings
        DataQual.check_model_output(x_history, 'regression', self.config)
        
        x_history = x_history.to(self.device)
        edge_idx = self.edge_index.to(self.device)
        
        with torch.no_grad():
            weather_pred = self.weather_model(x_history, edge_idx)
            
            drought_logits = self.drought_model(weather_pred, edge_idx)
            drought_proba = torch.softmax(drought_logits, dim=-1)
            drought_class = torch.argmax(drought_logits, dim=-1)
            
        DataQual.check_model_output(drought_proba, 'classification', self.config)
        
        return {
            'weather_pred': weather_pred.cpu(),
            'drought_logits': drought_logits.cpu(),
            'drought_class': drought_class.cpu(),
            'drought_proba': drought_proba.cpu()
        }
        
    @classmethod
    def from_checkpoints(cls, config) -> "DroughtForecastPipeline":
        processed_path = config["data"].get("processed_path", "data/processed")
        chkpt_dir = os.path.join(processed_path, "checkpoints")
        
        weather_path = os.path.join(chkpt_dir, "weather_model_best.pt")
        drought_path = os.path.join(chkpt_dir, "drought_model_best.pt")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        w_model = WeatherForecastModel(config)
        d_model = DroughtClassifier(config)
        
        w_model.load_state_dict(torch.load(weather_path, map_location=device))
        d_model.load_state_dict(torch.load(drought_path, map_location=device))
        
        w_model.to(device)
        d_model.to(device)
        
        from data.graph_builder import build_grid_graph
        graph_cfg = config.get("graph", {})
        
        # Calculate size assuming strictly uniform steps natively
        lat_sz = int((config["data"]["lat_range"][1] - config["data"]["lat_range"][0]) / config["data"]["grid_step"])
        lon_sz = int((config["data"]["lon_range"][1] - config["data"]["lon_range"][0]) / config["data"]["grid_step"])
        
        edge_index = build_grid_graph(
            lat_size=lat_sz, 
            lon_size=lon_sz, 
            connectivity=graph_cfg.get("connectivity", "4-neighbor"),
            periodic_lon=graph_cfg.get("periodic_longitude", True)
        )
        
        return cls(w_model, d_model, edge_index, config)
        
    def to_xarray(self, result: dict, lat: np.ndarray, lon: np.ndarray, time=None) -> xr.Dataset:
        lat_sz = len(lat)
        lon_sz = len(lon)
        
        ds_vars = {}
        V_names = self.config["data"]["variables"]
        
        w_p = result['weather_pred'].numpy().reshape(lat_sz, lon_sz, -1)
        for i, v_name in enumerate(V_names):
            ds_vars[f"pred_{v_name}"] = (("lat", "lon"), w_p[:, :, i])
            
        d_c = result['drought_class'].numpy().reshape(lat_sz, lon_sz)
        ds_vars["drought_class"] = (("lat", "lon"), d_c)
        
        C = result['drought_proba'].shape[-1]
        d_p = result['drought_proba'].numpy().reshape(lat_sz, lon_sz, C)
        for i in range(C):
            ds_vars[f"drought_proba_class_{i}"] = (("lat", "lon"), d_p[:, :, i])
            
        coords = {"lat": lat, "lon": lon}
        
        if time is not None:
            for k in ds_vars:
                ds_vars[k] = (["time", "lat", "lon"], np.expand_dims(ds_vars[k][1], axis=0))
            coords["time"] = [time] if not isinstance(time, (list, np.ndarray)) else time
            
        ds = xr.Dataset(data_vars=ds_vars, coords=coords)
        return ds
