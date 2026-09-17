import numpy as np

import copy
import torch
import torch.nn as nn
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig, 
    quantize_
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
