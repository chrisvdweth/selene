import torch
import random


def select_device(force_cpu=False, required_memory_gb=None):

    # Check if any available GPU should be ignored and the CPU should be used
    if force_cpu == True:
        return torch.device("cpu")
    
    candidate_gpus = []
    # Loop over all available devices
    for i in range(torch.cuda.device_count()):
        # Get free and total amount of memory for current device
        free, total = torch.cuda.mem_get_info(device=i)
        if (required_memory_gb is None) or ((free/(1024*1024*1024)) > required_memory_gb):
            candidate_gpus.append((i, free))

    # If no candidates found return CPU, otherwise pick randomly
    if len(candidate_gpus) == 0:
        return torch.device("cpu")
    else:
        device = torch.cuda.device(random.choice(candidate_gpus)[0])
        return torch.device(f"cuda:{device.idx}")