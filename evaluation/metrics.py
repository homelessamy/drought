"""
Metrics including accuracy, F1 macro/weighted, precision, and recall.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

__all__ = ["evaluate_regression", "evaluate_classification", "print_classification_report"]

def evaluate_regression(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Returns: {MSE, RMSE, MAE, R2} — one value per variable + global mean.
    Requires preds/targets to be shape (..., V), where V is channels.
    """
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    V = preds_flat.shape[1]
    
    metrics = {
        'MSE': [], 'RMSE': [], 'MAE': [], 'R2': []
    }
    
    for v in range(V):
        p = preds_flat[:, v]
        t = targets_flat[:, v]
        
        mse = mean_squared_error(t, p)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(np.sqrt(mse))
        metrics['MAE'].append(mean_absolute_error(t, p))
        metrics['R2'].append(r2_score(t, p))
        
    metrics['global'] = {k: float(np.mean(v)) for k, v in metrics.items()}
    return metrics

def evaluate_classification(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Returns accuracy, f1_macro, f1_weighted, precision_per_class, recall_per_class, confusion_matrix.
    """
    preds_flat = preds.flatten()
    ts_flat = targets.flatten()
    
    acc = accuracy_score(ts_flat, preds_flat)
    f1_m = f1_score(ts_flat, preds_flat, average='macro', zero_division=0)
    f1_w = f1_score(ts_flat, preds_flat, average='weighted', zero_division=0)
    
    prec_per = precision_score(ts_flat, preds_flat, average=None, zero_division=0).tolist()
    rec_per = recall_score(ts_flat, preds_flat, average=None, zero_division=0).tolist()
    
    cm = confusion_matrix(ts_flat, preds_flat)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_m,
        'f1_weighted': f1_w,
        'precision_per_class': prec_per,
        'recall_per_class': rec_per,
        'confusion_matrix': cm
    }

def print_classification_report(metrics: dict) -> None:
    """
    Pretty-prints a classification metrics table to stdout. 
    """
    print(f"{'Class':<6} | {'Precision':<9} | {'Recall':<8} | {'F1':<8}")
    print("-------+-----------+----------+---------")
    
    prec_per = metrics['precision_per_class']
    rec_per = metrics['recall_per_class']
    C = len(prec_per)
    
    for c in range(C):
        p = prec_per[c]
        r = rec_per[c]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        print(f"{c:<6} | {p:<9.4f} | {r:<8.4f} | {f1:<8.4f}")
        
    print("-------+-----------+----------+---------")
    print(f"{'Macro':<6} | {'':<9} | {'':<8} | {metrics['f1_macro']:<8.4f}")
    print(f"{'Acc':<6} | {'':<9} | {'':<8} | {metrics['accuracy']:<8.4f}")
