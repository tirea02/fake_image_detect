import torch
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create two large random matrices
size = 10000  # Large matrix size
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Measure time taken on GPU
torch.cuda.synchronize()  # Ensure previous operations are done
start_time = time.time()
result = torch.mm(a, b)  # Matrix multiplication
torch.cuda.synchronize()  # Wait for the computation to finish
end_time = time.time()

print(f"Matrix Multiplication (10000x10000) on {device} took {end_time - start_time:.4f} seconds.")