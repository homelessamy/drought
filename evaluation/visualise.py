"""
Visualisations: global heatmap and scatter plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import r2_score

__all__ = ["plot_drought_map", "plot_scatter"]

def plot_drought_map(pred_classes: np.ndarray, true_classes: np.ndarray, 
                     lat: np.ndarray, lon: np.ndarray,
                     title: str, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    cmap = ListedColormap(['#edf8b1', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8'])
    
    if lat.ndim == 1 and lon.ndim == 1:
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        pred_2d = pred_classes.reshape(len(lat), len(lon))
        true_2d = true_classes.reshape(len(lat), len(lon))
    else:
        lon_mesh, lat_mesh = lon, lat
        pred_2d = pred_classes
        true_2d = true_classes

    for ax, data_2d, sub_title in zip(axes, [true_2d, pred_2d], ["Ground Truth", "Predictions"]):
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        im = ax.pcolormesh(lon_mesh, lat_mesh, data_2d, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-0.5, vmax=4.5)
        ax.set_title(sub_title)
        
    fig.suptitle(title, fontsize=16)
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_scatter(preds: np.ndarray, targets: np.ndarray, var_names: list, save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    
    V = min(preds_flat.shape[1], 8)
    
    for v in range(V):
        ax = axes[v]
        p = preds_flat[:, v]
        t = targets_flat[:, v]
        
        if len(p) > 10000:
            idx = np.random.choice(len(p), 10000, replace=False)
            p_plot = p[idx]
            t_plot = t[idx]
        else:
            p_plot = p
            t_plot = t
            
        r2 = r2_score(t, p)
        
        ax.scatter(t_plot, p_plot, alpha=0.3, s=5, c='b')
        
        min_val = min(np.min(t_plot), np.min(p_plot))
        max_val = max(np.max(t_plot), np.max(p_plot))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f"{var_names[v]} (R²: {r2:.3f})")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
