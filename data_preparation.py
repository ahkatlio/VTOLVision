#!/usr/bin/env python3
"""
VTOL Vision - Enhanced Data Preparation
=====================================
Creates YOLO annotation format with bold, colorful training data matching the mixed dataset style.

Author: VTOL Vision Team (Ahkatlio) 
Date: 2025-08-09
"""

import os
import cv2
import numpy as np
import json
import yaml
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
import random
import string
from sklearn.model_selection import train_test_split

console = Console()

class VTOLDataPreparator:
    def __init__(self):
        self.datasets_dir = Path("Datasets")
        self.yolo_dataset_dir = Path("YOLO_Dataset")
        
        # Remove old dataset
        if self.yolo_dataset_dir.exists():
            shutil.rmtree(self.yolo_dataset_dir)
        
        # Create directory structure
        self.yolo_dataset_dir.mkdir()
        (self.yolo_dataset_dir / "images" / "train").mkdir(parents=True)
        (self.yolo_dataset_dir / "images" / "val").mkdir(parents=True)
        (self.yolo_dataset_dir / "images" / "test").mkdir(parents=True)
        (self.yolo_dataset_dir / "labels" / "train").mkdir(parents=True)
        (self.yolo_dataset_dir / "labels" / "val").mkdir(parents=True)
        (self.yolo_dataset_dir / "labels" / "test").mkdir(parents=True)
        
        # Class definitions (48 total classes)
        self.shape_classes = {
            'arrow': 0, 'circle': 1, 'cross': 2, 'ellipse': 3, 'heart': 4,
            'hexagon': 5, 'octagon': 6, 'pentagon': 7, 'rectangle': 8,
            'star': 9, 'trapezoid': 10, 'triangle': 11
        }
        self.number_classes = {str(i): i + 12 for i in range(10)}  # Classes 12-21
        self.letter_classes = {chr(65 + i): i + 22 for i in range(26)}  # Classes 22-47
        self.all_classes = {**self.shape_classes, **self.number_classes, **self.letter_classes}
        
        console.print(f"📊 Total classes: {len(self.all_classes)} (12 shapes + 10 numbers + 26 letters)")
        
        # Colors matching the mixed dataset style
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (255, 192, 203), (0, 128, 255), (128, 255, 0), (255, 20, 147),
            (255, 165, 0), (128, 0, 128), (0, 128, 0), (255, 69, 0), (70, 130, 180), (220, 20, 60)
        ]
        
        self.shapes = ['circle', 'rectangle', 'triangle', 'pentagon', 'hexagon', 'star', 
                      'trapezoid', 'octagon', 'ellipse', 'cross', 'arrow', 'heart']
        self.letters = list(string.ascii_uppercase)
        self.numbers = list('0123456789')
        
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'annotations_created': 0
        }
    
    def create_realistic_background(self, img_size=640):
        """Create realistic outdoor-like backgrounds (from mixed dataset style)"""
        bg = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        bg_type = random.choice(['sky', 'grass', 'concrete', 'mixed'])
        
        if bg_type == 'sky':
            for i in range(img_size):
                intensity = 135 + int(120 * i / img_size)
                bg[i, :] = [intensity, intensity, 255]
        elif bg_type == 'grass':
            base_green = random.randint(60, 100)
            for i in range(img_size):
                for j in range(img_size):
                    noise = random.randint(-20, 20)
                    green_val = max(0, min(255, base_green + noise))
                    bg[i, j] = [random.randint(10, 30), green_val, random.randint(10, 40)]
        elif bg_type == 'concrete':
            base_gray = random.randint(80, 120)
            for i in range(img_size):
                for j in range(img_size):
                    noise = random.randint(-15, 15)
                    gray_val = max(0, min(255, base_gray + noise))
                    bg[i, j] = [gray_val, gray_val, gray_val]
        else:  # mixed
            patch_size = img_size // 4
            for i in range(0, img_size, patch_size):
                for j in range(0, img_size, patch_size):
                    patch_color = random.choice([(100, 150, 80), (120, 120, 120), (180, 200, 255)])
                    end_i = min(i + patch_size, img_size)
                    end_j = min(j + patch_size, img_size)
                    bg[i:end_i, j:end_j] = patch_color
        
        return bg
    
    def add_text_element(self, img, text, position, color, size_scale=1.0):
        """Add text (letter or number) to image - EXACT same style as mixed dataset"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = size_scale * random.uniform(0.8, 1.5)
        thickness = random.randint(2, 4)
        
        # First layer: White outline/border (thicker)
        cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness + 2)
        # Second layer: Colored text on top (thinner)
        cv2.putText(img, text, position, font, font_scale, color, thickness)
    
    def rotate_shape(self, points, angle, center):
        """Rotate shape points around center"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        translated = points - center
        rotated = np.dot(translated, rotation_matrix.T)
        return rotated + center
    
    def draw_shape(self, img, shape, center, size, color, rotation=0):
        """Draw a shape at specified position - EXACT same style as mixed dataset"""
        center_x, center_y = center
        
        if shape == 'circle':
            cv2.circle(img, (center_x, center_y), size, color, -1)
            cv2.circle(img, (center_x, center_y), size, (255, 255, 255), 2)
            
        elif shape == 'rectangle':
            half_w, half_h = size, int(size * random.uniform(0.6, 1.4))
            rect_pts = np.array([
                [-half_w, -half_h], [half_w, -half_h], 
                [half_w, half_h], [-half_w, half_h]
            ], dtype=np.float32)
            
            rotated_pts = self.rotate_shape(rect_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'triangle':
            tri_pts = np.array([
                [0, -size], [-size * 0.866, size * 0.5], 
                [size * 0.866, size * 0.5]
            ], dtype=np.float32)
            
            rotated_pts = self.rotate_shape(tri_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'pentagon':
            # Create pentagon points
            pentagon_pts = []
            for i in range(5):
                angle = i * 2 * np.pi / 5 - np.pi/2
                x = size * np.cos(angle)
                y = size * np.sin(angle)
                pentagon_pts.append([x, y])
            pentagon_pts = np.array(pentagon_pts, dtype=np.float32)
            
            rotated_pts = self.rotate_shape(pentagon_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'hexagon':
            # Create hexagon points
            hexagon_pts = []
            for i in range(6):
                angle = i * 2 * np.pi / 6
                x = size * np.cos(angle)
                y = size * np.sin(angle)
                hexagon_pts.append([x, y])
            hexagon_pts = np.array(hexagon_pts, dtype=np.float32)
            
            rotated_pts = self.rotate_shape(hexagon_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'star':
            # Create 5-pointed star
            star_pts = []
            for i in range(10):
                angle = i * np.pi / 5
                if i % 2 == 0:
                    radius = size
                else:
                    radius = size * 0.4
                x = radius * np.cos(angle - np.pi/2)
                y = radius * np.sin(angle - np.pi/2)
                star_pts.append([x, y])
            star_pts = np.array(star_pts, dtype=np.float32)
            
            rotated_pts = self.rotate_shape(star_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'ellipse':
            axes = (size, int(size * 0.6))
            cv2.ellipse(img, (center_x, center_y), axes, rotation * 180/np.pi, 0, 360, color, -1)
            cv2.ellipse(img, (center_x, center_y), axes, rotation * 180/np.pi, 0, 360, (255, 255, 255), 2)
            
        elif shape == 'heart':
            # Simple heart approximation using circles and triangle
            heart_size = int(size * 0.7)
            cv2.circle(img, (center_x - heart_size//2, center_y - heart_size//2), heart_size//2, color, -1)
            cv2.circle(img, (center_x + heart_size//2, center_y - heart_size//2), heart_size//2, color, -1)
            
            # Triangle bottom
            tri_pts = np.array([
                [center_x - heart_size, center_y],
                [center_x + heart_size, center_y],
                [center_x, center_y + heart_size]
            ], dtype=np.int32)
            cv2.fillPoly(img, [tri_pts], color)
            
        elif shape == 'cross':
            # Draw cross as two rectangles
            thickness = size // 3
            # Vertical bar
            cv2.rectangle(img, (center_x - thickness//2, center_y - size), 
                         (center_x + thickness//2, center_y + size), color, -1)
            # Horizontal bar  
            cv2.rectangle(img, (center_x - size, center_y - thickness//2),
                         (center_x + size, center_y + thickness//2), color, -1)
            # White outlines
            cv2.rectangle(img, (center_x - thickness//2, center_y - size), 
                         (center_x + thickness//2, center_y + size), (255, 255, 255), 2)
            cv2.rectangle(img, (center_x - size, center_y - thickness//2),
                         (center_x + size, center_y + thickness//2), (255, 255, 255), 2)
            
        elif shape == 'arrow':
            # Draw arrow pointing up
            arrow_pts = np.array([
                [center_x, center_y - size],  # Top point
                [center_x - size//2, center_y - size//2],  # Left
                [center_x - size//4, center_y - size//2],  # Left inner
                [center_x - size//4, center_y + size//2],  # Left bottom
                [center_x + size//4, center_y + size//2],  # Right bottom
                [center_x + size//4, center_y - size//2],  # Right inner
                [center_x + size//2, center_y - size//2],  # Right
            ], dtype=np.int32)
            
            cv2.fillPoly(img, [arrow_pts], color)
            cv2.polylines(img, [arrow_pts], True, (255, 255, 255), 2)
            
        elif shape == 'trapezoid':
            # Create trapezoid
            trap_pts = np.array([
                [-size * 0.8, -size * 0.5],  # Top left
                [size * 0.8, -size * 0.5],   # Top right
                [size, size * 0.5],          # Bottom right
                [-size, size * 0.5]          # Bottom left
            ], dtype=np.float32)
            
            rotated_pts = self.rotate_shape(trap_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'octagon':
            # Create octagon points
            octagon_pts = []
            for i in range(8):
                angle = i * 2 * np.pi / 8
                x = size * np.cos(angle)
                y = size * np.sin(angle)
                octagon_pts.append([x, y])
            octagon_pts = np.array(octagon_pts, dtype=np.float32)
            
            rotated_pts = self.rotate_shape(octagon_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
    
    def generate_training_data(self, num_samples=800):
        """Generate training data matching mixed dataset style"""
        console.print("🎨 [bold green]Generating Style-Matched Training Data[/bold green]")
        
        all_annotations = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Creating training samples...", total=num_samples)
            
            for i in range(num_samples):
                try:
                    # Create realistic background
                    img_size = 640
                    img = self.create_realistic_background(img_size)
                    
                    # Random number of objects (1-3)
                    num_objects = random.randint(1, 3)
                    
                    image_annotations = []
                    used_positions = []  # Track positions to avoid overlap
                    
                    for j in range(num_objects):
                        # Random object type
                        obj_category = random.choice(['shape', 'letter', 'number'])
                        
                        if obj_category == 'shape':
                            obj_class = random.choice(self.shapes)
                            class_id = self.shape_classes[obj_class]
                            obj_type = obj_class
                        elif obj_category == 'letter':
                            obj_class = random.choice(self.letters)
                            class_id = self.letter_classes[obj_class]
                            obj_type = 'letter'
                        else:  # number
                            obj_class = random.choice(self.numbers)
                            class_id = self.number_classes[obj_class]
                            obj_type = 'number'
                        
                        # Find non-overlapping position
                        attempts = 0
                        while attempts < 20:
                            margin = 80
                            x = random.randint(margin, img_size - margin)
                            y = random.randint(margin, img_size - margin)
                            
                            # Check for overlap
                            too_close = False
                            for prev_x, prev_y in used_positions:
                                if abs(x - prev_x) < 120 and abs(y - prev_y) < 120:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                break
                            attempts += 1
                        
                        used_positions.append((x, y))
                        
                        # Size and color
                        size = random.randint(40, 70)
                        color = random.choice(self.colors)
                        rotation = random.uniform(0, 2 * np.pi)
                        
                        # Draw object
                        if obj_type in self.shapes:
                            self.draw_shape(img, obj_type, (x, y), size, color, rotation)
                            bbox_size = size * 2.5
                        else:
                            # Text rendering with EXACT same style as mixed dataset
                            text_size = random.uniform(1.5, 2.5)
                            text_x = x - 20  # Adjust for text centering
                            text_y = y + 15
                            self.add_text_element(img, obj_class, (text_x, text_y), color, text_size)
                            bbox_size = size * 2.0
                        
                        # Create YOLO annotation
                        x_center = x / img_size
                        y_center = y / img_size
                        width = min(bbox_size / img_size, 1.0)
                        height = min(bbox_size / img_size, 1.0)
                        
                        annotation = {
                            'image_path': f"train_sample_{i:04d}.jpg",
                            'class_id': class_id,
                            'class_name': obj_class,
                            'bbox': [x_center, y_center, width, height]
                        }
                        
                        image_annotations.append(annotation)
                    
                    # Save image
                    img_filename = f"train_sample_{i:04d}.jpg"
                    temp_path = Path("temp_training") / img_filename
                    temp_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(temp_path), img)
                    
                    # Update annotations with correct path
                    for ann in image_annotations:
                        ann['image_path'] = temp_path
                        all_annotations.append(ann)
                    
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"⚠️ Error generating sample {i}: {e}")
                    continue
        
        console.print(f"✅ Generated {len(all_annotations)} annotations from {num_samples} images")
        return all_annotations
    
    def process_mixed_dataset(self):
        """Process mixed test dataset"""
        console.print("🔀 [bold magenta]Processing Mixed Test Dataset[/bold magenta]")
        
        mixed_dir = self.datasets_dir / "mixed_test"
        mixed_files = list(mixed_dir.glob("*.png"))
        
        processed_mixed = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing mixed images...", total=len(mixed_files))
            
            for img_file in mixed_files:
                try:
                    filename_parts = img_file.stem.split('_')[2:]
                    
                    objects = []
                    for part in filename_parts:
                        if part in self.all_classes:
                            objects.append(part)
                    
                    if not objects:
                        continue
                    
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    height, width = img.shape[:2]
                    
                    for i, obj_name in enumerate(objects):
                        class_id = self.all_classes[obj_name]
                        
                        # Estimate bounding box (simplified)
                        bbox_width = 0.8 / len(objects)
                        bbox_height = 0.8
                        x_center = 0.2 + (i * bbox_width) + (bbox_width / 2)
                        y_center = 0.5
                        
                        annotation = {
                            'image_path': img_file,
                            'class_id': class_id,
                            'class_name': obj_name,
                            'bbox': [x_center, y_center, bbox_width, bbox_height],
                            'object_index': i
                        }
                        
                        processed_mixed.append(annotation)
                    
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"⚠️ Error processing {img_file}: {e}")
                    continue
        
        console.print(f"✅ Processed {len(mixed_files)} mixed images with {len(processed_mixed)} annotations")
        return processed_mixed
    
    def create_train_val_test_splits(self, all_annotations, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train/validation/test sets"""
        console.print("📊 [bold yellow]Creating Train/Validation/Test Splits[/bold yellow]")
        
        # Group annotations by image
        image_groups = {}
        for ann in all_annotations:
            img_path = str(ann['image_path'])
            if img_path not in image_groups:
                image_groups[img_path] = []
            image_groups[img_path].append(ann)
        
        # Split image groups
        image_paths = list(image_groups.keys())
        random.shuffle(image_paths)
        
        n_total = len(image_paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = image_paths[:n_train]
        val_images = image_paths[n_train:n_train + n_val]
        test_images = image_paths[n_train + n_val:]
        
        # Create split annotations
        train_annotations = []
        val_annotations = []
        test_annotations = []
        
        for img_path in train_images:
            train_annotations.extend(image_groups[img_path])
        for img_path in val_images:
            val_annotations.extend(image_groups[img_path])
        for img_path in test_images:
            test_annotations.extend(image_groups[img_path])
        
        console.print(f"""
📈 [bold]Dataset Split Statistics:[/bold]
• Total images: {n_total}
• Train: {len(train_images)} images ({len(train_annotations)} annotations)
• Validation: {len(val_images)} images ({len(val_annotations)} annotations)
• Test: {len(test_images)} images ({len(test_annotations)} annotations)
        """)
        
        return {
            'train': train_annotations,
            'val': val_annotations,
            'test': test_annotations
        }
    
    def copy_images_and_create_labels(self, split_data):
        """Copy images and create YOLO label files"""
        console.print("📂 [bold cyan]Copying Images and Creating Labels[/bold cyan]")
        
        for split_name, image_annotations in split_data.items():
            console.print(f"\nProcessing {split_name} split...")
            
            # Group by image
            image_groups = {}
            for ann in image_annotations:
                img_path = str(ann['image_path'])
                if img_path not in image_groups:
                    image_groups[img_path] = []
                image_groups[img_path].append(ann)
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[progress.description]Processing {split_name}..."),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Copying {split_name} images...", total=len(image_groups))
                
                for img_path, img_annotations in image_groups.items():
                    try:
                        # Copy image
                        img_path = Path(img_path)
                        img_filename = f"{split_name}_{img_path.stem}_{random.randint(1000, 9999)}.jpg"
                        
                        dest_img_path = self.yolo_dataset_dir / "images" / split_name / img_filename
                        shutil.copy2(img_path, dest_img_path)
                        
                        # Create YOLO label file
                        label_filename = img_filename.replace('.jpg', '.txt')
                        label_path = self.yolo_dataset_dir / "labels" / split_name / label_filename
                        
                        with open(label_path, 'w') as f:
                            for ann in img_annotations:
                                class_id = ann['class_id']
                                bbox = ann['bbox']
                                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                        
                        self.stats['annotations_created'] += len(img_annotations)
                        progress.advance(task)
                        
                    except Exception as e:
                        console.print(f"⚠️ Error processing {img_path}: {e}")
                        continue
            
            # Update stats
            if split_name == 'train':
                self.stats['train_images'] = len(image_groups)
            elif split_name == 'val':
                self.stats['val_images'] = len(image_groups)
            elif split_name == 'test':
                self.stats['test_images'] = len(image_groups)
        
        self.stats['total_images'] = self.stats['train_images'] + self.stats['val_images'] + self.stats['test_images']
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        console.print("📄 [bold blue]Creating Dataset Configuration[/bold blue]")
        
        dataset_config = {
            'path': str(self.yolo_dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.all_classes),
            'names': list(self.all_classes.keys())
        }
        
        yaml_path = self.yolo_dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        console.print(f"✅ Dataset configuration saved: {yaml_path}")
    
    def run_data_preparation(self):
        """Run the complete data preparation pipeline"""
        console.print(Panel("""
🔧 [bold]VTOL Vision - Enhanced Data Preparation[/bold]

Creating YOLO dataset with training data that EXACTLY matches 
the style of the mixed test dataset:

• Same text rendering technique (white outline + colored text)
• Same shape drawing style (filled + white borders)
• Same background patterns and colors
• Multi-object scenes matching test complexity
        """, style="green"))
        
        try:
            # Generate training data matching mixed dataset style
            training_annotations = self.generate_training_data(800)
            
            # Process mixed dataset for test
            mixed_annotations = self.process_mixed_dataset()
            
            # Combine all annotations
            all_annotations = training_annotations + mixed_annotations
            console.print(f"\n📊 Total annotations: {len(all_annotations)}")
            
            # Create splits
            split_data = self.create_train_val_test_splits(all_annotations)
            
            # Copy images and create labels
            self.copy_images_and_create_labels(split_data)
            
            # Create YAML config
            self.create_dataset_yaml()
            
            # Clean up temp directory
            temp_dir = Path("temp_training")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            self.display_final_stats()
            
        except Exception as e:
            console.print(f"❌ Error in data preparation: {e}")
            raise
    
    def display_final_stats(self):
        """Display final statistics"""
        console.print("\n" + "="*80)
        console.print(Panel(f"""
🎉 [bold green]DATA PREPARATION COMPLETE![/bold green]

📊 [bold]Final Statistics:[/bold]
• Total Images: {self.stats['total_images']}
• Train Images: {self.stats['train_images']} (Style-matched to mixed dataset)
• Validation Images: {self.stats['val_images']}
• Test Images: {self.stats['test_images']} (Mixed test data)
• Total Annotations: {self.stats['annotations_created']}
• Classes: {len(self.all_classes)} (48 total)

📁 [bold]Dataset Location:[/bold] {self.yolo_dataset_dir.absolute()}

🎯 [bold]Key Feature:[/bold] Training data uses EXACT same rendering style as mixed dataset!
Text with white outlines + colored text, shapes with white borders.
        """, style="bright_green"))

def main():
    """Main function"""
    preparator = VTOLDataPreparator()
    preparator.run_data_preparation()

if __name__ == "__main__":
    main()
