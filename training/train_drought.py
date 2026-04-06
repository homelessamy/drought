"""
Training code for the drought classification model.
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
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from models.drought_model import DroughtClassifier
from training.losses import weighted_cross_entropy

__all__ = ["train_drought"]

def prepare_drought_data(ds: xr.Dataset, config: dict, edge_index: torch.Tensor, weather_model=None, device='cpu') -> Data:
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
    
    if 'drought_class' in ds.data_vars:
        y_val = ds['drought_class'].values[T, :, :].reshape(N)
    else:
        y_val = np.zeros((N,))
        
    y_tensor = torch.tensor(y_val, dtype=torch.long)
    
    if weather_model is not None:
        weather_model.eval()
        weather_model.to(device)
        
        x_val = arr_all[:T, :, :]
        x_val = np.transpose(x_val, (1, 0, 2)).reshape(N, T * V)
        x_val = np.nan_to_num(x_val)
        x_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        edges_dev = edge_index.to(device)
        
        with torch.no_grad():
            x_seq = x_tensor.view(-1, T, V)
            x_hat_t = weather_model(x_seq, edges_dev)
        
        x_final = x_hat_t.cpu()
    else:
        x_val = arr_all[T, :, :]
        x_val = np.nan_to_num(x_val)
        x_final = torch.tensor(x_val, dtype=torch.float32)
        
    return Data(x=x_final, y=y_tensor, edge_index=edge_index)

def train_drought(config: dict, ds_train: xr.Dataset, ds_val: xr.Dataset, edge_index: torch.Tensor, weather_model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = prepare_drought_data(ds_train, config, edge_index, weather_model, device)
    val_data = prepare_drought_data(ds_val, config, edge_index, weather_model, device)
    
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
    
    labels_np = train_data.y.numpy()
    classes = np.arange(config["data"].get("C", 5))
    c_weights = compute_class_weight('balanced', classes=classes, y=labels_np)
    c_weights_tensor = torch.tensor(c_weights, dtype=torch.float32).to(device)
    
    model = DroughtClassifier(config).to(device)
    
    lr = float(config["training"].get("lr", 1e-3))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    epochs = config["training"].get("epochs_drought", 30)
    patience_limit = config["training"].get("patience", 7)
    clip_norm = float(config["model"].get("weight_norm_clip", 1.0))
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_f1 = -float('inf')
    patience_counter = 0
    
    processed_path = config["data"].get("processed_path", "data/processed")
    chkpt_dir = os.path.join(processed_path, "checkpoints")
    os.makedirs(chkpt_dir, exist_ok=True)
    
    best_path = os.path.join(chkpt_dir, "drought_model_best.pt")
    last_path = os.path.join(chkpt_dir, "drought_model_last.pt")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch.x, batch.edge_index)
            
            out_target = logits[:batch.batch_size]
            y_target = batch.y[:batch.batch_size]
            
            loss = weighted_cross_entropy(out_target, y_target, c_weights_tensor)
            loss.backward()
            optimizer.step()
            
            if hasattr(model, 'sage') and hasattr(model.sage, 'clip_weights'):
                model.sage.clip_weights(clip_norm)
                
            total_train_loss += loss.item() * batch.batch_size
            
        avg_train_loss = total_train_loss / train_data.num_nodes
        scheduler.step()
        
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                
                out_target = logits[:batch.batch_size]
                y_target = batch.y[:batch.batch_size]
                
                preds = torch.argmax(out_target, dim=-1)
                
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y_target.cpu().numpy())
                
        all_val_preds = np.concatenate(val_preds, axis=0)
        all_val_targets = np.concatenate(val_targets, axis=0)
        
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        torch.save(model.state_dict(), last_path)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best model (Macro-F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Train Drought script runnable - pass datasets manually through an external executor mapping.")
