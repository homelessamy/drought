"""
Graph edge_index construction for 1-degree lat-lon grid.
"""
import os
import torch

__all__ = ["build_grid_graph"]

def build_grid_graph(lat_size=180, lon_size=360, connectivity='4-neighbor', periodic_lon=True) -> torch.Tensor:
    """
    Constructs edge indices for 4-neighbor or 8-neighbor grids.
    Nodes indexed row-major: idx = lat_i * lon_size + lon_j
    Returns edge_index as undirected (both directions) LongTensor.
    """
    cache_path = f"grid_graph_{lat_size}x{lon_size}_{connectivity}_periodic{periodic_lon}.pt"
    if os.path.exists(cache_path):
        return torch.load(cache_path)
        
    edges = []
    
    for lat_i in range(lat_size):
        for lon_j in range(lon_size):
            node_idx = lat_i * lon_size + lon_j
            
            neighbors = []
            
            # 4-neighbor connectivity
            if lat_i > 0:
                neighbors.append((lat_i - 1, lon_j))
            if lat_i < lat_size - 1:
                neighbors.append((lat_i + 1, lon_j))
                
            if lon_j > 0:
                neighbors.append((lat_i, lon_j - 1))
            elif periodic_lon:
                neighbors.append((lat_i, lon_size - 1))
                
            if lon_j < lon_size - 1:
                neighbors.append((lat_i, lon_j + 1))
            elif periodic_lon:
                neighbors.append((lat_i, 0))
                
            # Additional edges for 8-neighbor
            if connectivity == '8-neighbor':
                # Up-left / Up-right
                if lat_i > 0:
                    if lon_j > 0:
                        neighbors.append((lat_i - 1, lon_j - 1))
                    elif periodic_lon:
                        neighbors.append((lat_i - 1, lon_size - 1))
                        
                    if lon_j < lon_size - 1:
                        neighbors.append((lat_i - 1, lon_j + 1))
                    elif periodic_lon:
                        neighbors.append((lat_i - 1, 0))
                        
                # Down-left / Down-right
                if lat_i < lat_size - 1:
                    if lon_j > 0:
                        neighbors.append((lat_i + 1, lon_j - 1))
                    elif periodic_lon:
                        neighbors.append((lat_i + 1, lon_size - 1))
                        
                    if lon_j < lon_size - 1:
                        neighbors.append((lat_i + 1, lon_j + 1))
                    elif periodic_lon:
                        neighbors.append((lat_i + 1, 0))
                        
            # Populate directed edges mapping
            for (n_lat, n_lon) in neighbors:
                n_idx = n_lat * lon_size + n_lon
                edges.append([node_idx, n_idx])

    # Transform to 2xE shape
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Save cache
    torch.save(edge_index, cache_path)
    
    return edge_index
