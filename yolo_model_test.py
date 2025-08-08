"""
YOLO Model Testing and Benchmarking Script

This script tests different YOLO model variants on the development machine
and benchmarks their performance for Raspberry Pi deployment planning.

Usage:
    python yolo_model_test.py
"""

import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class YOLOModelTester:
    def __init__(self):
        self.models_dir = Path("Models/YOLO")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {
            'YOLOv8n': self.models_dir / 'yolov8n.pt',
            'YOLOv8s': self.models_dir / 'yolov8s.pt',
            'YOLOv5n': self.models_dir / 'yolov5nu.pt',  
            'YOLOv5s': self.models_dir / 'yolov5su.pt'   
        }
        self.test_image_path = "Datasets/mixed_test/mixed_000_pentagon_LI_LR.png"
        self.results = {}
    
    def download_models(self):
        """Download YOLO models for testing"""
        console.print(Panel(f"📥 Downloading YOLO Models to {self.models_dir}", style="blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for model_name, model_path in self.models.items():
                task = progress.add_task(f"Downloading {model_name}...", total=None)
                
                try:
                    if model_path.exists():
                        progress.update(task, description=f"✅ {model_name} already exists")
                        console.print(f"✅ {model_name} already downloaded at {model_path}")
                        continue
                    
                    model = YOLO(model_path.name)  # Download with just the filename
                    
                    current_dir = Path(".")
                    pt_files = list(current_dir.glob("*.pt"))
                    
                    for pt_file in pt_files:
                        if pt_file.name in [p.name for p in self.models.values()]:
                            target_path = self.models_dir / pt_file.name
                            if not target_path.exists():
                                pt_file.rename(target_path)
                                console.print(f"📁 Moved {pt_file.name} to {target_path}")
                    
                    progress.update(task, description=f"✅ {model_name} ready")
                    console.print(f"✅ {model_name} organized to {model_path}")
                    
                except Exception as e:
                    console.print(f"❌ Failed to download {model_name}: {e}")
                    
                time.sleep(0.5)
        
        self.cleanup_model_files()
    
    def cleanup_model_files(self):
        """Move any remaining .pt files to the organized Models/YOLO folder"""
        current_dir = Path(".")
        pt_files = list(current_dir.glob("*.pt"))
        
        if pt_files:
            console.print(f"🧹 Cleaning up {len(pt_files)} remaining .pt files...")
            for pt_file in pt_files:
                target_path = self.models_dir / pt_file.name
                if not target_path.exists():
                    pt_file.rename(target_path)
                    console.print(f"📁 Moved {pt_file.name} to Models/YOLO/")
                else:
                    # File already exists, remove duplicate
                    pt_file.unlink()
                    console.print(f"🗑️ Removed duplicate {pt_file.name}")
    
    def get_model_info(self, model):
        """Get detailed information about the model"""
        try:
            total_params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, 'model') else 0
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad) if hasattr(model, 'model') else 0
            
            model_size_mb = 0
            if hasattr(model, 'ckpt_path') and model.ckpt_path and os.path.exists(model.ckpt_path):
                model_size_mb = os.path.getsize(model.ckpt_path) / (1024 * 1024)
            elif hasattr(model, 'model_path') and model.model_path and os.path.exists(model.model_path):
                model_size_mb = os.path.getsize(model.model_path) / (1024 * 1024)
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': model_size_mb
            }
        except Exception as e:
            console.print(f"⚠️ Could not get detailed model info: {e}")
            return {
                'total_params': 0,
                'trainable_params': 0,
                'model_size_mb': 0
            }
    
    def benchmark_model(self, model_name, model_path, num_runs=5):
        """Benchmark a single YOLO model"""
        console.print(f"🔍 Testing {model_name}...")
        
        try:
            model = YOLO(str(model_path))
            
            model_info = self.get_model_info(model)
            
            # Ensure we get the correct model size from file
            if model_path.exists():
                model_info['model_size_mb'] = model_path.stat().st_size / (1024 * 1024)
            elif model_info['model_size_mb'] == 0:
                # Try to find the model file size another way
                try:
                    import torch
                    if hasattr(model, 'model'):
                        # Calculate approximate size from parameters
                        total_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
                        model_info['model_size_mb'] = total_size / (1024 * 1024)
                except:
                    model_info['model_size_mb'] = 0
            
            if not os.path.exists(self.test_image_path):
                console.print(f"⚠️ Test image not found: {self.test_image_path}")
                console.print("Using a sample image instead...")
                test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                test_img = cv2.imread(self.test_image_path)
                test_img = cv2.resize(test_img, (640, 640))  
            
            _ = model(test_img, verbose=False)
            
            inference_times = []
            
            for i in range(num_runs):
                start_time = time.time()
                results = model(test_img, verbose=False)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
            
            avg_inference_time = np.mean(inference_times)
            fps = 1.0 / avg_inference_time
            
            rpi_scaling_factor = 0.3
            estimated_rpi_fps = fps * rpi_scaling_factor
            
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            benchmark_result = {
                'model_name': model_name,
                'avg_inference_time': avg_inference_time,
                'fps': fps,
                'estimated_rpi_fps': estimated_rpi_fps,
                'num_detections': num_detections,
                'total_params': model_info['total_params'],
                'model_size_mb': model_info['model_size_mb'],
                'memory_usage_mb': self.estimate_memory_usage(model_info['total_params'])
            }
            
            console.print(f"✅ {model_name}: {fps:.1f} FPS (Est. RPi: {estimated_rpi_fps:.1f} FPS)")
            return benchmark_result
            
        except Exception as e:
            console.print(f"❌ Error testing {model_name}: {e}")
            return None
    
    def estimate_memory_usage(self, total_params):
        """Estimate memory usage based on model parameters"""
        base_memory = (total_params * 4) / (1024 * 1024)  # MB
        overhead = 100  # MB for framework overhead
        return base_memory + overhead
    
    def run_benchmarks(self):
        """Run benchmarks on all models"""
        console.print(Panel("🚀 Starting YOLO Model Benchmarking", style="green"))
        
        self.download_models()
        
        console.print("\n🔬 Running Performance Benchmarks...")
        
        for model_name, model_path in self.models.items():
            result = self.benchmark_model(model_name, model_path)
            if result:
                self.results[model_name] = result
        
        self.display_results()
        self.save_results()
        self.provide_recommendations()
    
    def display_results(self):
        """Display benchmark results in a nice table"""
        if not self.results:
            console.print("❌ No benchmark results to display")
            return
        
        console.print(f"\n📊 [bold]YOLO Model Benchmark Results[/bold]")
        
        table = Table(title="🎯 Performance Comparison")
        table.add_column("Model", style="cyan", width=12)
        table.add_column("FPS (PC)", style="green", width=10)
        table.add_column("Est. RPi FPS", style="yellow", width=12)
        table.add_column("Inference (ms)", style="blue", width=14)
        table.add_column("Parameters", style="magenta", width=12)
        table.add_column("Size (MB)", style="red", width=10)
        table.add_column("Memory (MB)", style="white", width=12)
        
        sorted_results = sorted(self.results.values(), 
                               key=lambda x: x['estimated_rpi_fps'], 
                               reverse=True)
        
        for result in sorted_results:
            # Format parameters properly
            params_str = f"{result['total_params']/1e6:.1f}M" if result['total_params'] > 0 else "N/A"
            size_str = f"{result['model_size_mb']:.1f}" if result['model_size_mb'] > 0 else "N/A"
            
            table.add_row(
                result['model_name'],
                f"{result['fps']:.1f}",
                f"{result['estimated_rpi_fps']:.1f}",
                f"{result['avg_inference_time']*1000:.1f}",
                params_str,
                size_str,
                f"{result['memory_usage_mb']:.0f}"
            )
        
        console.print(table)
    
    def provide_recommendations(self):
        """Provide model selection recommendations"""
        if not self.results:
            return
        
        best_speed = max(self.results.values(), key=lambda x: x['estimated_rpi_fps'])
        best_balance = None
        
        # Lower threshold to 2.0 FPS as minimum viable for RPi
        balanced_candidates = [r for r in self.results.values() if r['estimated_rpi_fps'] >= 2.0]
        if balanced_candidates:
            best_balance = max(balanced_candidates, key=lambda x: x['fps'])
        
        balance_fps = f"{best_balance['estimated_rpi_fps']:.1f}" if best_balance else "N/A"
        balance_memory = f"{best_balance['memory_usage_mb']:.0f}" if best_balance else "N/A"
        balance_name = best_balance['model_name'] if best_balance else 'N/A'
        backup_name = best_balance['model_name'] if best_balance else best_speed['model_name']
        
        console.print(Panel(f"""
🏆 MODEL SELECTION RECOMMENDATIONS

🚀 Best for Speed: {best_speed['model_name']}
   • Estimated RPi FPS: {best_speed['estimated_rpi_fps']:.1f}
   • Memory Usage: {best_speed['memory_usage_mb']:.0f} MB
   • Best for: Real-time applications, battery life

⚖️ Best Balance: {balance_name}
   • Estimated RPi FPS: {balance_fps}
   • Memory Usage: {balance_memory} MB
   • Best for: Competition deployment

🎯 Recommendation for VTOL Competition:
   • Primary: {best_speed['model_name']} (fastest, most reliable)
   • Backup: {backup_name} (if accuracy is critical)
   
💡 Next Steps:
   1. Test selected model with your custom dataset
   2. Fine-tune for shape, letter, and number detection
   3. Validate on Raspberry Pi hardware
""", style="bright_green"))
    
    def save_results(self):
        """Save benchmark results to file"""
        import json
        
        results_dir = Path("Results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "yolo_benchmark_results.json"
        
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': 'Development PC',
            'test_conditions': {
                'input_size': '640x640',
                'num_runs': 5,
                'framework': 'Ultralytics YOLOv8'
            },
            'models_location': str(self.models_dir),
            'results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        console.print(f"💾 Results saved to: [cyan]{results_file}[/cyan]")

def main():
    console.print(Panel("""
🎯 [bold]YOLO Model Testing & Benchmarking[/bold]

This script will:
• Download YOLO model variants
• Test inference performance
• Estimate Raspberry Pi performance  
• Provide deployment recommendations

Ready to start testing?
""", style="blue"))
    
    tester = YOLOModelTester()
    tester.run_benchmarks()
    
    console.print("\n✅ [green]Model testing complete![/green]")
    console.print("📝 [yellow]Check the results and update your action plan![/yellow]")

if __name__ == "__main__":
    main()
