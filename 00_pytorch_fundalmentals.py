# %%
import torch

# %%
#Exercise 2
X = torch.rand(7,7)
X

# %%
#Exercise 3
Y = torch.rand(1,7).T
MULT = torch.matmul(X, Y)
MULT

# %%
#Exercise 4
torch.manual_seed(0)
X = torch.rand(7,7)
Y = torch.rand(1,7).T

mult = torch.matmul(X,Y)
mult

# %%
# Exercise 6
torch.manual_seed(1234)
X = torch.rand(2,3)
Y = torch.rand(2,3).T

Y
# %%
# Exercise 7
M = torch.matmul(X,Y)
M

# %%
# Exercise 8
max = torch.max(M)
min = torch.min(M)
print(f"Max: {max}")
print(f"Min: {min}")

# %%
#Exercise 9
print(f"Arg max: {torch.argmax(M)}")
print(f"Arg min: {torch.argmin(M)}")

# %%
#Exercise 10
torch.manual_seed(7)

first_tensor = torch.rand(1,1,1,10)
squeezed_tensor = T.squeeze()

print(f"Tensor: {first_tensor}")
print(f"Tensor enxuto: {squeezed_tensor}")
print(f"Shape: {squeezed_tensor.shape}")