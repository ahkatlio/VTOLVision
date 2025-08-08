"""
GPU Information Script
Check PyTorch CUDA setup and GPU memory details
"""

import torch

def check_gpu_info():
    print("=" * 50)
    print("PyTorch GPU Information")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  • Compute capability: {props.major}.{props.minor}")
            print(f"  • Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  • Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  • Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  • Memory free: {(props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3:.2f} GB")
            print(f"  • Multi-processor count: {props.multi_processor_count}")
            print()
        
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test GPU computation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("✅ GPU computation test: PASSED")
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU computation test: FAILED - {e}")
    else:
        print("❌ No CUDA GPU available - using CPU only")
        print("📝 To use GPU:")
        print("   1. Install CUDA-enabled PyTorch:")
        print("   2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu_info()
