"""
Test Environment Setup
Check if we're using the correct Python environment with CUDA support
"""

import sys
import torch
print("=" * 60)
print("🐍 PYTHON ENVIRONMENT TEST")
print("=" * 60)

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

print("🔥 PYTORCH & CUDA TEST")
print("-" * 30)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Test computation
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x)
        print("✅ GPU computation test: PASSED")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU computation test: FAILED - {e}")
else:
    print("❌ No CUDA support - using CPU only")

print()
print("📦 ULTRALYTICS TEST")
print("-" * 30)
try:
    from ultralytics import YOLO
    print("✅ ultralytics imported successfully")
    
    # Test if ultralytics can use GPU
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎯 YOLO will use device: {device}")
    
except ImportError as e:
    print(f"❌ ultralytics import failed: {e}")
except Exception as e:
    print(f"⚠️ ultralytics test failed: {e}")

print("=" * 60)
print("🎯 ENVIRONMENT STATUS")
print("=" * 60)

if torch.cuda.is_available():
    print("✅ Environment ready for GPU training!")
else:
    print("⚠️ CPU-only environment - install CUDA PyTorch")
