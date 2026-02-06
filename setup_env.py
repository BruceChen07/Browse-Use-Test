import sys
import subprocess
import platform
import os
import shutil

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available via nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def install_packages(packages, index_url=None, extra_args=None):
    """Install packages using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if isinstance(packages, str):
        packages = packages.split()
    cmd.extend(packages)
    
    if index_url:
        cmd.extend(["--index-url", index_url])
    
    if extra_args:
        cmd.extend(extra_args)
        
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def uninstall_packages(packages):
    """Uninstall packages."""
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y"]
    if isinstance(packages, str):
        packages = packages.split()
    cmd.extend(packages)
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.call(cmd)

def main():
    print("🔍 Checking system environment...")
    system = platform.system()
    has_gpu = check_nvidia_gpu()
    
    print(f"🖥️  System: {system}")
    print(f"🎮 NVIDIA GPU Detected: {'Yes' if has_gpu else 'No'}")
    
    # Core ML dependencies
    torch_pkgs = ["torch", "torchvision", "torchaudio"]
    
    # 1. Handle PyTorch Installation
    if system == "Windows" and has_gpu:
        print("\n🚀 Configuring for Windows with CUDA 12.1...")
        # Uninstall existing torch to prevent conflicts/version mix
        print("   Removing existing torch installations to ensure clean state...")
        uninstall_packages(torch_pkgs)
        
        # Install CUDA versions
        print("   Installing CUDA-enabled PyTorch...")
        install_packages(torch_pkgs, index_url="https://download.pytorch.org/whl/cu121")
    else:
        print("\n⚙️  Configuring for CPU (or standard PyPI)...")
        # Standard install (usually CPU on Windows, or default on Linux/Mac)
        install_packages(torch_pkgs)

    # 2. Install other project dependencies
    print("\n📦 Installing other required dependencies...")
    other_deps = [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "pillow",
        "browser-use",
        "python-dotenv",
        "qwen-vl-utils" # often needed for Qwen VL models, adding just in case or we stick to main.py imports
    ]
    
    # Check if requirements.txt exists and install from it too/instead?
    # For now, let's explicitly install the ML stack + browser-use to be sure.
    install_packages(other_deps)

    print("\n✅ Environment setup complete!")
    print("   Run 'python main.py' to start the service.")

if __name__ == "__main__":
    main()
