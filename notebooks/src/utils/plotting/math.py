import numpy as np

from IPython.display import display, Latex



def draw_matrix(arr, decimals=None):
    if decimals is not None:
        try:
            format_func = np.vectorize(lambda x: f"{x:.{decimals}f}")
            arr = format_func(arr)            
        except:
            pass

    
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    body = r" \\ ".join([" & ".join(map(str, row)) for row in arr])
    display(Latex(rf"$\begin{{bmatrix}} {body} \end{{bmatrix}}$"))