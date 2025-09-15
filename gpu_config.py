#!/usr/bin/env python3
"""
GPU Configuration for Multi-Agent RL Training
"""
import torch
import os

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        return "cpu"
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🚀 GPU detected: {gpu_name}")
    print(f"💾 GPU memory: {gpu_memory:.1f} GB")
    
    # Set CUDA device
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    # Optimize memory usage for RTX 4070 SUPER
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    # Clear cache
    torch.cuda.empty_cache()
    
    print("✅ GPU configured for training")
    return device

def get_device():
    """Get the appropriate device (cuda or cpu)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.1f}GB")
    else:
        print("CUDA not available")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cache cleared")

# Configuration for Stable Baselines3
def get_sb3_device():
    """Get device string for Stable Baselines3"""
    return "cuda" if torch.cuda.is_available() else "cpu"

# Configuration for Ray
def get_ray_config():
    """Get Ray configuration for GPU usage"""
    if torch.cuda.is_available():
        return {
            "num_gpus": 1,
            "num_cpus": os.cpu_count(),
        }
    else:
        return {
            "num_gpus": 0,
            "num_cpus": os.cpu_count(),
        }

if __name__ == "__main__":
    device = setup_gpu()
    print_memory_usage()
    
    print("\n🔧 Configuration:")
    print(f"Device for PyTorch: {device}")
    print(f"Device for SB3: {get_sb3_device()}")
    print(f"Ray config: {get_ray_config()}")
