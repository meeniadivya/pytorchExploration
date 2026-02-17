import torch

# PyTorch Tensors Exploration

# Creating tensors
print("Creating Tensors:")
print("=" * 50)

# From lists
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"Tensor from list: {tensor_from_list}")

# From numpy array
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from numpy: {tensor_from_numpy}")

# Zeros and ones
zeros_tensor = torch.zeros(3, 4)
ones_tensor = torch.ones(2, 3)
print(f"Zeros tensor shape: {zeros_tensor.shape}")
print(f"Ones tensor shape: {ones_tensor.shape}")

# Random tensors
random_tensor = torch.randn(2, 3)
print(f"Random tensor:\n{random_tensor}")

print("\n" + "=" * 50)
print("Tensor Operations:")
print("=" * 50)

# Basic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"Addition: {a + b}")
print(f"Element-wise multiplication: {a * b}")
print(f"Matrix multiplication: {torch.dot(a, b)}")

print("\nDone!")
