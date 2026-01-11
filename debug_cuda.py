
import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("❌ No CUDA device detected. You are likely running on CPU.")
    print("If you have an NVIDIA GPU, you need to install PyTorch with CUDA support.")
    print("Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
