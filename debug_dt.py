import sys
sys.path.insert(0, '.')
from pure_model import build_pure_model

try:
    @build_pure_model
    def dt(age, income, score):
        if score > 700:
            if income > 50000:
                return 95
            else:
                return 70
        else:
            return 10
    print(dt._forward_code)
except Exception as e:
    import traceback
    traceback.print_exc()

import torch
dt = dt.to('cuda')
a = torch.tensor([30.0], device='cuda')
i = torch.tensor([60000.0], device='cuda')
s = torch.tensor([750.0], device='cuda')
print(dt(a, i, s))
