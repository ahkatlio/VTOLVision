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
        # Create Models/YOLO directory if it doesn't exist
        self.models_dir = Path("Models/YOLO")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {
            'YOLOv8n': self.models_dir / 'yolov8n.pt',
            'YOLOv8s': self.models_dir / 'yolov8s.pt',
            'YOLOv5n': self.models_dir / 'yolov5n.pt',
            'YOLOv5s': self.models_dir / 'yolov5s.pt'
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
                    # Check if model already exists
                    if model_path.exists():
                        progress.update(task, description=f"✅ {model_name} already exists")
                        console.print(f"✅ {model_name} already downloaded at {model_path}")
                        continue
                    
                    # Load model (this will download if not present)
                    # For organized storage, we need to download to the correct location
                    model = YOLO(model_path.name)  # Download with just the filename
                    
                    # Move the downloaded model to our organized folder if needed
                    current_location = Path(model_path.name)
                    if current_location.exists() and current_location != model_path:
                        current_location.rename(model_path)
                        console.print(f"📁 Moved {model_name} to {model_path}")
                    
                    progress.update(task, description=f"✅ {model_name} ready")
                    console.print(f"✅ {model_name} downloaded to {model_path}")
                    
                except Exception as e:
                    console.print(f"❌ Failed to download {model_name}: {e}")
                    
                time.sleep(0.5)  # Small delay for visual effect
    
    def get_model_info(self, model):
        """Get detailed information about the model"""
        try:
            # Get model size
            model_path = model.model_info()
            
            # Calculate parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            # Get model file size if available
            model_size_mb = 0
            if hasattr(model, 'ckpt_path') and model.ckpt_path:
                model_size_mb = os.path.getsize(model.ckpt_path) / (1024 * 1024)
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': model_size_mb
            }
        except Exception as e:
            console.print(f"⚠️ Could not get model info: {e}")
            return {
                'total_params': 0,
                'trainable_params': 0,
                'model_size_mb': 0
            }
    
    def benchmark_model(self, model_name, model_file, num_runs=5):
        """Benchmark a single YOLO model"""
        console.print(f"🔍 Testing {model_name}...")
        
        try:
            # Load model
            model = YOLO(model_file)
            
            # Get model information
            model_info = self.get_model_info(model)
            
            # Load test image
            if not os.path.exists(self.test_image_path):
                console.print(f"⚠️ Test image not found: {self.test_image_path}")
                console.print("Using a sample image instead...")
                # Create a dummy image for testing
                test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                test_img = cv2.imread(self.test_image_path)
                test_img = cv2.resize(test_img, (640, 640))  # Standard YOLO input size
            
            # Warm-up run
            _ = model(test_img, verbose=False)
            
            # Benchmark inference time
            inference_times = []
            
            for i in range(num_runs):
                start_time = time.time()
                results = model(test_img, verbose=False)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_inference_time = np.mean(inference_times)
            fps = 1.0 / avg_inference_time
            
            # Estimate Raspberry Pi performance (rough estimation based on CPU scaling)
            # Typical scaling factor for RPi4 vs modern PC (conservative estimate)
            rpi_scaling_factor = 0.3
            estimated_rpi_fps = fps * rpi_scaling_factor
            
            # Get detection results from last run
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
        # Rough estimation: 4 bytes per parameter + overhead
        base_memory = (total_params * 4) / (1024 * 1024)  # MB
        overhead = 100  # MB for framework overhead
        return base_memory + overhead
    
    def run_benchmarks(self):
        """Run benchmarks on all models"""
        console.print(Panel("🚀 Starting YOLO Model Benchmarking", style="green"))
        
        # Download models first
        self.download_models()
        
        console.print("\n🔬 Running Performance Benchmarks...")
        
        for model_name, model_file in self.models.items():
            result = self.benchmark_model(model_name, model_file)
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
        
        # Sort by estimated RPi FPS (descending)
        sorted_results = sorted(self.results.values(), 
                               key=lambda x: x['estimated_rpi_fps'], 
                               reverse=True)
        
        for result in sorted_results:
            table.add_row(
                result['model_name'],
                f"{result['fps']:.1f}",
                f"{result['estimated_rpi_fps']:.1f}",
                f"{result['avg_inference_time']*1000:.1f}",
                f"{result['total_params']/1e6:.1f}M",
                f"{result['model_size_mb']:.1f}" if result['model_size_mb'] > 0 else "N/A",
                f"{result['memory_usage_mb']:.0f}"
            )
        
        console.print(table)
    
    def provide_recommendations(self):
        """Provide model selection recommendations"""
        if not self.results:
            return
        
        # Find best models for different criteria
        best_speed = max(self.results.values(), key=lambda x: x['estimated_rpi_fps'])
        best_balance = None
        
        # Find best balance (FPS > 10 and good performance)
        balanced_candidates = [r for r in self.results.values() if r['estimated_rpi_fps'] >= 8]
        if balanced_candidates:
            best_balance = max(balanced_candidates, key=lambda x: x['fps'])
        
        console.print(Panel(f"""
🏆 [bold]MODEL SELECTION RECOMMENDATIONS[/bold]

🚀 [green]Best for Speed:[/green] {best_speed['model_name']}
   • Estimated RPi FPS: {best_speed['estimated_rpi_fps']:.1f}
   • Memory Usage: {best_speed['memory_usage_mb']:.0f} MB
   • Best for: Real-time applications, battery life

⚖️ [yellow]Best Balance:[/yellow] {best_balance['model_name'] if best_balance else 'N/A'}
   • Estimated RPi FPS: {best_balance['estimated_rpi_fps']:.1f if best_balance else 'N/A'}
   • Memory Usage: {best_balance['memory_usage_mb']:.0f if best_balance else 'N/A'} MB
   • Best for: Competition deployment

🎯 [cyan]Recommendation for VTOL Competition:[/cyan]
   • Primary: {best_speed['model_name']} (fastest, most reliable)
   • Backup: {best_balance['model_name'] if best_balance else best_speed['model_name']} (if accuracy is critical)
   
💡 [blue]Next Steps:[/blue]
   1. Test selected model with your custom dataset
   2. Fine-tune for shape, letter, and number detection
   3. Validate on Raspberry Pi hardware
""", style="bright_green"))
    
    def save_results(self):
        """Save benchmark results to file"""
        import json
        
        results_file = "yolo_benchmark_results.json"
        
        # Add metadata
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': 'Development PC',
            'test_conditions': {
                'input_size': '640x640',
                'num_runs': 5,
                'framework': 'Ultralytics YOLOv8'
            },
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
