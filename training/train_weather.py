"""
Training code for the weather model.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import numpy as np
import xarray as xr
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from models.weather_model import WeatherForecastModel

__all__ = ["train_weather"]

def prepare_data_object(ds: xr.Dataset, config: dict, edge_index: torch.Tensor) -> Data:
    vars_list = config["data"]["variables"]
    V = len(vars_list)
    T = config["data"]["T"]
    
    lat_sz = ds.sizes['lat']
    lon_sz = ds.sizes['lon']
    N = lat_sz * lon_sz
    
    arr_list = []
    for var in vars_list:
        arr = ds[var].values 
        arr = arr.reshape(arr.shape[0], N)
        arr_list.append(arr)
    arr_all = np.stack(arr_list, axis=-1) 
    
    if arr_all.shape[0] <= T:
        raise ValueError("Dataset time dimension must be > T")
        
    x_val = arr_all[:T, :, :] 
    x_val = np.transpose(x_val, (1, 0, 2)) 
    x_val = x_val.reshape(N, T * V) 
    
    y_val = arr_all[T, :, :] 
    
    x_val = np.nan_to_num(x_val)
    y_val = np.nan_to_num(y_val)
    
    x_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    return Data(x=x_tensor, y=y_tensor, edge_index=edge_index)

def train_weather(config: dict, ds_train: xr.Dataset, ds_val: xr.Dataset, edge_index: torch.Tensor):
    train_data = prepare_data_object(ds_train, config, edge_index)
    val_data = prepare_data_object(ds_val, config, edge_index)
    
    batch_size = config["training"].get("batch_nodes", 4096)
    
    train_loader = NeighborLoader(
        train_data,
        num_neighbors=[10, 10],
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        val_data,
        num_neighbors=[10, 10],
        batch_size=batch_size,
        shuffle=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherForecastModel(config).to(device)
    
    lr = float(config["training"].get("lr", 1e-3))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    epochs = config["training"].get("epochs_weather", 50)
    patience_limit = config["training"].get("patience", 7)
    clip_norm = float(config["model"].get("weight_norm_clip", 1.0))
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    
    best_val_mse = float('inf')
    patience_counter = 0
    
    T = config["data"]["T"]
    V = len(config["data"]["variables"])
    
    processed_path = config["data"].get("processed_path", "data/processed")
    chkpt_dir = os.path.join(processed_path, "checkpoints")
    os.makedirs(chkpt_dir, exist_ok=True)
    
    best_path = os.path.join(chkpt_dir, "weather_model_best.pt")
    last_path = os.path.join(chkpt_dir, "weather_model_last.pt")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            x_seq = batch.x.view(-1, T, V)
            out = model(x_seq, batch.edge_index)
            
            out_target = out[:batch.batch_size]
            y_target = batch.y[:batch.batch_size]
            
            loss = criterion(out_target, y_target)
            loss.backward()
            optimizer.step()
            
            if hasattr(model, 'sage') and hasattr(model.sage, 'clip_weights'):
                model.sage.clip_weights(clip_norm)
                
            total_train_loss += loss.item() * batch.batch_size
            
        avg_train_loss = total_train_loss / train_data.num_nodes
        scheduler.step()
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_seq = batch.x.view(-1, T, V)
                out = model(x_seq, batch.edge_index)
                
                out_target = out[:batch.batch_size]
                y_target = batch.y[:batch.batch_size]
                
                loss = criterion(out_target, y_target)
                total_val_loss += loss.item() * batch.batch_size
                
                val_preds.append(out_target.cpu().numpy())
                val_targets.append(y_target.cpu().numpy())
                
        avg_val_loss = total_val_loss / val_data.num_nodes
        all_val_preds = np.concatenate(val_preds, axis=0)
        all_val_targets = np.concatenate(val_targets, axis=0)
        
        val_rmse = np.sqrt(mean_squared_error(all_val_targets, all_val_preds))
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss (MSE): {avg_val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        
        torch.save(model.state_dict(), last_path)
        
        if avg_val_loss < best_val_mse:
            best_val_mse = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best model (MSE: {best_val_mse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model

if __name__ == '__main__':
    # Default runnable wrapper setup
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Train Weather script runnable - pass datasets manually through an external executor mapping.")
