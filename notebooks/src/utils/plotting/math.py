import numpy as np
import torch

from IPython.display import display, Latex



def draw_matrix(X, decimals=None):
    np.set_printoptions(precision=2)
    if isinstance(X, np.ndarray):
        pass
    elif isinstance(X, torch.Tensor):
        pass
    else:
        pass

    # Convert 1d array to (n,1) column vector
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Iterate through each entry to process/prepare value for matrix representation
    rows = []
    for row in X:
        values = []
        for val in row:
            # If input is a PyTorch tensor, get value via .item()
            if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
                val = val.item()
            # Returns True if the variable is either an int or a float
            if isinstance(val, float):
                val = float(val)
            elif isinstance(val, int):
                val = int(val)
            else:
                print("Unsupported type")
            # Check if the value should be rounded
            if decimals is not None and isinstance(decimals, int) and decimals >= 0:
                #values.append(format(val, f".{decimals}f"))
                values.append(f"{val:.{decimals}f}")
            else:
                values.append(repr(val))
        row_str = " & ".join(values)
        rows.append(row_str)
    
    body = r" \\ ".join(rows)
    display(Latex(rf"$\begin{{bmatrix}} {body} \end{{bmatrix}}$"))