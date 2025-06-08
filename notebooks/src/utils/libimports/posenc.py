import numpy as np
import torch
import torch.nn as nn

def int_to_binary_array(number, width):
    binary_string = np.binary_repr(number, width=width)
    binary_array = np.fromiter(binary_string, dtype='int8')
    return binary_array
