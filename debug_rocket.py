import sys, ast, inspect, textwrap
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize

@tensorize
def rocket(thrust, angle, burn):
    g = 9.81
    mass = 1000
    vy = thrust * 0.5 / mass * burn - g * burn
    if vy > 0:
        t_peak = vy / g
        h = 0.5 * vy * t_peak
        return h
    else:
        return 0

print("Generated:")
print(rocket._tensor_source)
