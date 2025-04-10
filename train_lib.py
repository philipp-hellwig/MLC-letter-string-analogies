import random
import numpy as np
import torch
import time
import math
from copy import deepcopy, copy

def seed_all(seed=0):
    print("* Setting all random seeds to ",seed,'*')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract(include,arr):
    # Create a new list only using the included (boolean) elements of arr 
    #
    # Input
    #  include : [n len] boolean array (numpy or python)
    #  arr : [ n length array]
    assert len(include)==len(arr)
    return [a for idx,a in enumerate(arr) if include[idx]]

# Robertson's asMinutes and timeSince helper functions to print time elapsed and estimated time
# remaining given the current time and progress

def asMinutes(s): 
    # convert seconds to minutes
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    # prints time elapsed and estimated time remaining
    #
    # Input 
    #  since : previous time
    #  percent : amount of training complete
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))  
