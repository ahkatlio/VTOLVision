"""
VTOL Vision - YOLO Model Training
=================================
Phase 3: Model Training with enhanced dataset

This script trains a YOLOv8 model on the prepared dataset with proper monitoring and validation.

Author: VTOL Vision Team (Ahkatlio)
Date: 2025-08-09
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import time

console = Console()

class VTOLModelTrainer:
    def __init__(self):
        self.dataset_path = Path("YOLO_Dataset")
        self.models_dir = Path("Models/YOLO")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration - Optimized for MX450 (2GB VRAM) with stable training
        self.config = {
            'model_size': str(self.models_dir / 'yolov8n.pt'),  # Use local model file
            'epochs': 100,
            'batch_size': 8,     # Reduced for 2GB VRAM
            'img_size': 640,
            'patience': 20,      # Early stopping
            'lr0': 0.001,        # MUCH lower initial learning rate to prevent NaN
            'lrf': 0.0001,       # Lower final learning rate
            'momentum': 0.9,     # Slightly lower momentum
            'weight_decay': 0.0005,
            'warmup_epochs': 5,  # More warmup epochs for stability
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.01,  # Lower warmup bias LR
            'box': 7.5,          # Box loss gain
            'cls': 0.5,          # Class loss gain
            'dfl': 1.5,          # DFL loss gain
            'device': 0          # Use GPU 0
        }
    
    def validate_dataset(self):
        """Validate dataset structure and configuration"""
        console.print("🔍 [bold blue]Validating Dataset Structure[/bold blue]")
        
        dataset_yaml = self.dataset_path / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
        
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'test', 'nc', 'names']
        for key in required_keys:
            if key not in dataset_config:
                raise ValueError(f"Missing required key in dataset.yaml: {key}")
        
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_path / "images" / split
            label_dir = self.dataset_path / "labels" / split
            
            if not img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            if not label_dir.exists():
                raise FileNotFoundError(f"Label directory not found: {label_dir}")
            
            img_count = len(list(img_dir.glob("*.jpg")))
            label_count = len(list(label_dir.glob("*.txt")))
            
            console.print(f"  ✅ {split.upper()}: {img_count} images, {label_count} labels")
        
        console.print(f"✅ Dataset validation complete! Classes: {dataset_config['nc']}")
        return dataset_config
    
    def setup_training_environment(self):
        """Set up training environment and display configuration"""
        console.print(Panel("""
🚀 [bold]VTOL Vision - YOLO Model Training[/bold]

Phase 3: Training enhanced dataset with style-matched data
• Training data matches mixed test dataset visual style
• 48 classes: shapes, numbers, letters
• Enhanced augmentation with noise and realistic backgrounds
        """, style="green"))
        
        config_table = Table(title="🔧 Training Configuration", border_style="blue")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")
        
        for key, value in self.config.items():
            config_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(config_table)
        
        # Create results directory
        self.results_dir = self.models_dir / f"training_run_{int(time.time())}"
        self.results_dir.mkdir(exist_ok=True)
        
        console.print(f"📁 Results will be saved to: {self.results_dir}")
        return True
    
    def train_model(self, dataset_config):
        """Train the YOLO model"""
        console.print("\n🎯 [bold green]Starting Model Training[/bold green]")
        
        # Debug: Check GPU status before training
        import torch
        console.print(f"🔍 PyTorch version: {torch.__version__}")
        console.print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"🔍 GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"🔍 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        try:
            model_path = Path(self.config['model_size'])
            if not model_path.exists():
                raise FileNotFoundError(f"Local model file not found: {model_path}")
            
            model = YOLO(str(model_path))
            console.print(f"✅ Loaded local model: {model_path}")
            console.print(f"📁 Using existing model (no download required)")
            
            # Display training parameters that will prevent NaN
            console.print(f"🔧 Learning rate: {self.config['lr0']} (reduced to prevent NaN)")
            console.print(f"🔧 Optimizer: AdamW (more stable)")
            console.print(f"🔧 Mixed precision: Disabled (prevents NaN)")
            
            # Start training
            with console.status("[bold green]Training in progress..."):
                results = model.train(
                    data=str(self.dataset_path / "dataset.yaml"),
                    epochs=self.config['epochs'],
                    batch=self.config['batch_size'],
                    imgsz=self.config['img_size'],
                    patience=self.config['patience'],
                    lr0=self.config['lr0'],
                    lrf=self.config['lrf'],
                    momentum=self.config['momentum'],
                    weight_decay=self.config['weight_decay'],
                    warmup_epochs=self.config['warmup_epochs'],
                    warmup_momentum=self.config['warmup_momentum'],
                    warmup_bias_lr=self.config['warmup_bias_lr'],
                    box=self.config['box'],
                    cls=self.config['cls'],
                    dfl=self.config['dfl'],
                    project=str(self.models_dir),
                    name=f"vtol_training_{int(time.time())}",
                    exist_ok=True,
                    pretrained=True,
                    optimizer='AdamW',  # AdamW is more stable than SGD for some datasets
                    verbose=True,
                    seed=42,
                    deterministic=True,
                    single_cls=False,
                    rect=False,
                    cos_lr=False,
                    close_mosaic=10,
                    resume=False,
                    amp=False,  # Disable Mixed Precision to prevent NaN issues
                    fraction=1.0,
                    profile=False,
                    overlap_mask=True,
                    mask_ratio=4,
                    dropout=0.0,
                    val=True,
                    split='val',
                    save_json=True,
                    save_hybrid=False,
                    conf=None,
                    iou=0.7,
                    max_det=300,
                    half=False,
                    device=self.config['device'],  # Use GPU 0
                    dnn=False,
                    plots=True,
                    source=None,
                    vid_stride=1,
                    stream_buffer=False,
                    visualize=False,
                    augment=False,
                    agnostic_nms=False,
                    retina_masks=False,
                    embed=None,
                    show=False,
                    save_frames=False,
                    save_txt=False,
                    save_conf=False,
                    save_crop=False,
                    show_labels=True,
                    show_conf=True,
                    show_boxes=True,
                    line_width=None
                )
            
            console.print("✅ [bold green]Training completed successfully![/bold green]")
            return results, model
            
        except Exception as e:
            console.print(f"❌ [bold red]Training failed: {e}[/bold red]")
            raise
    
    def validate_model(self, model, dataset_config):
        """Validate trained model on test dataset"""
        console.print("\n📊 [bold yellow]Validating Model Performance[/bold yellow]")
        
        try:
            # Validate on test set
            test_results = model.val(
                data=str(self.dataset_path / "dataset.yaml"),
                split='test',
                imgsz=self.config['img_size'],
                batch=self.config['batch_size'],
                conf=0.25,
                iou=0.7,
                max_det=300,
                half=False,
                device=self.config['device'],
                dnn=False,
                plots=True,
                save_json=True,
                save_hybrid=False,
                augment=False,
                verbose=True,
                project=str(self.models_dir),
                name=f"vtol_validation_{int(time.time())}",
                exist_ok=True
            )
            
            # Display results
            metrics_table = Table(title="📈 Model Performance Metrics", border_style="yellow")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")
            
            if hasattr(test_results, 'box'):
                box_metrics = test_results.box
                metrics_table.add_row("mAP50", f"{box_metrics.map50:.4f}")
                metrics_table.add_row("mAP50-95", f"{box_metrics.map:.4f}")
                metrics_table.add_row("Precision", f"{box_metrics.mp:.4f}")
                metrics_table.add_row("Recall", f"{box_metrics.mr:.4f}")
            
            console.print(metrics_table)
            
            # Check if success metrics are met
            if hasattr(test_results, 'box') and test_results.box.map50 >= 0.85:
                console.print("🎉 [bold green]SUCCESS! Model exceeds 85% mAP50 target![/bold green]")
            else:
                console.print("⚠️ [yellow]Model performance below 85% target - consider fine-tuning[/yellow]")
            
            return test_results
            
        except Exception as e:
            console.print(f"❌ [bold red]Validation failed: {e}[/bold red]")
            raise
    
    def save_model_for_deployment(self, model):
        """Save model in formats suitable for deployment"""
        console.print("\n💾 [bold cyan]Saving Model for Deployment[/bold cyan]")
        
        deployment_dir = self.models_dir / "deployment"
        deployment_dir.mkdir(exist_ok=True)
        
        try:
            model.save(deployment_dir / "vtol_vision_model.pt")
            console.print("✅ Saved PyTorch model (.pt)")
            
            model.export(format='onnx', 
                        imgsz=self.config['img_size'],
                        half=False,
                        dynamic=False,
                        simplify=True,
                        opset=None)
            console.print("✅ Exported ONNX model (.onnx)")
            
            deployment_info = {
                'model_format': 'PyTorch (.pt) and ONNX (.onnx)',
                'input_size': self.config['img_size'],
                'classes': 48,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'framework': 'YOLOv8n',
                'deployment_notes': 'Model trained on style-matched dataset. Use ONNX for RPi4 deployment.'
            }
            
            with open(deployment_dir / "deployment_info.yaml", 'w') as f:
                yaml.dump(deployment_info, f, default_flow_style=False)
            
            console.print(f"📁 Deployment files saved to: {deployment_dir}")
            
        except Exception as e:
            console.print(f"⚠️ [yellow]Error during export: {e}[/yellow]")
            console.print("💡 PyTorch model still available for deployment")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            dataset_config = self.validate_dataset()
            
            self.setup_training_environment()
            
            results, model = self.train_model(dataset_config)
            
            test_results = self.validate_model(model, dataset_config)
            
            self.save_model_for_deployment(model)
            
            console.print(Panel(f"""
🎉 [bold green]PHASE 3 TRAINING COMPLETE![/bold green]

✅ Model trained successfully
✅ Validation completed
✅ Deployment files ready for RPi4 team

📁 [bold]Results Location:[/bold] {self.models_dir}
🚀 [bold]Ready for Phase 4 deployment by RPi4 developer![/bold]
            """, style="bright_green"))
            
            return True
            
        except Exception as e:
            console.print(f"❌ [bold red]Training pipeline failed: {e}[/bold red]")
            return False

def main():
    """Main training function"""
    trainer = VTOLModelTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        console.print("\n🎯 [bold green]Ready to proceed to Phase 4 (RPi4 deployment)![/bold green]")
    else:
        console.print("\n⚠️ [bold yellow]Training issues detected - review and retry[/bold yellow]")

if __name__ == "__main__":
    main()
