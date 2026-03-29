"""
ULTIMATE DEMO: @tensorize_model as layers inside a real neural network.
Custom activation + custom gate from plain Python, trained with backprop.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from model_backend import tensorize_model

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("Python functions as neural network layers")
print("=" * 60)

@tensorize_model
def custom_activation(x):
    sp = math.log(1.0 + math.exp(x))
    return x * math.tanh(sp)

@tensorize_model
def custom_gate(x):
    if x > 3:
        return 1.0
    else:
        if x < -3:
            return 0.0
        else:
            return 0.5 + x / 6.0

custom_activation = custom_activation.to(device)
custom_gate = custom_gate.to(device)

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.act1 = custom_activation
        self.fc2 = nn.Linear(64, 64)
        self.gate = custom_gate
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        g = self.gate(self.fc2(x))
        return self.fc3(x * g)

model = CustomNet().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters())}")

N = 50000
X = torch.randn(N, 10, device=device)
y = (X[:,0]**2 + X[:,1]*X[:,2] + torch.sin(X[:,3]) + torch.randn(N, device=device)*0.1).unsqueeze(1)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("\nTraining...")
t0 = time.time()
for epoch in range(50):
    perm = torch.randperm(N, device=device)
    for start in range(0, N, 1024):
        idx = perm[start:start+1024]
        loss = loss_fn(model(X[idx]), y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            test_loss = loss_fn(model(X[:5000]), y[:5000]).item()
        print(f"  Epoch {epoch+1}: loss={test_loss:.4f}")

print(f"  Done in {time.time()-t0:.1f}s")

# Compare with ReLU
class ReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self, x): return self.net(x)

relu_model = ReLUNet().to(device)
relu_opt = optim.Adam(relu_model.parameters(), lr=1e-3)
t0 = time.time()
for epoch in range(50):
    perm = torch.randperm(N, device=device)
    for start in range(0, N, 1024):
        idx = perm[start:start+1024]
        loss = loss_fn(relu_model(X[idx]), y[idx])
        relu_opt.zero_grad()
        loss.backward()
        relu_opt.step()
relu_time = time.time()-t0

with torch.no_grad():
    custom_loss = loss_fn(model(X[:5000]), y[:5000]).item()
    relu_loss = loss_fn(relu_model(X[:5000]), y[:5000]).item()

print(f"\nCustom activation loss: {custom_loss:.4f}")
print(f"ReLU loss:              {relu_loss:.4f}")
print(f"Winner: {'CUSTOM' if custom_loss < relu_loss else 'ReLU'}")

torch.save(model.state_dict(), r"C:\Users\salih\Desktop\py2tensor\custom_net.pt")
print(f"\nSaved model. Plain Python -> nn.Module -> trained -> saved. Done.")
