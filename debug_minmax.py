import sys
sys.path.insert(0, '.')
from py2tensor import tensorize

@tensorize
def clamp_func(x):
    return max(min(x, 10), -10)

print(clamp_func._tensor_source)
