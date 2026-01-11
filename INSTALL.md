# Installation Guide

## 1. Prerequisites
*   **Python**: Version 3.10 or 3.11 recommended.
*   **OS**: Windows, Linux, or macOS.

## 2. Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### ⚡ GPU Support (Highly Recommended)
By default, `pip install torch` installs the CPU-only version. To enable GPU training (CUDA):

1.  **Uninstall current torch**:
    ```bash
    pip uninstall torch torchvision torchaudio -y
    ```
2.  **Install CUDA version** (Check [pytorch.org](https://pytorch.org/) for your specific version):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

## 3. Windows Configuration (Important)
On Windows, you must set the encoding environment variable to avoid errors with progress bars and emojis:

```powershell
$env:PYTHONIOENCODING="utf-8"
```
*(Add this to your PowerShell profile to make it permanent)*

## 4. Verify Setup
Run a quick check to ensure the environment and GPU are ready:

```bash
python -c "import gymnasium; import stable_baselines3; import torch; print(f'Ready! Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```
