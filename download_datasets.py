import os
import subprocess
import shutil
import stat
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import requests
import pandas as pd

console = Console()
DATASETS_DIR = "Datasets"

os.makedirs(DATASETS_DIR, exist_ok=True)

def remove_readonly(func, path, _):
    """Error handler for Windows readonly files"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_remove_tree(path):
    """Safely remove directory tree, handling Windows permission issues"""
    if os.path.exists(path):
        console.print(Panel(f"Cleaning up {os.path.basename(path)} repository...", style="yellow"))
        try:
            # On Windows, use onerror to handle readonly files
            if os.name == 'nt':  # Windows
                shutil.rmtree(path, onerror=remove_readonly)
            else:
                shutil.rmtree(path)
            console.print(f"✅ Successfully removed {os.path.basename(path)}", style="green")
        except Exception as e:
            console.print(f"⚠️ Warning: Could not fully remove {path}: {e}", style="yellow")

def download_shapes():
    console.print(Panel("🎨 Generating SUPER COOL shape dataset! 🎨", style="magenta"))
    
    dest_dir = os.path.join(DATASETS_DIR, "shapes")
    os.makedirs(dest_dir, exist_ok=True)
    
    create_simple_shapes(dest_dir)
    
    console.print(Panel("✅ Shape dataset ready!", style="green"))

def create_simple_shapes(dest_dir):
    """Create an advanced shape dataset with random positions, rotations, and effects"""
    import cv2
    import numpy as np
    import random
    
    console.print(Panel("🎨 Creating SUPER COOL shape dataset with random variations! 🎨", style="magenta"))
    
    shapes = ['circle', 'rectangle', 'triangle', 'pentagon', 'hexagon', 'star', 
              'trapezoid', 'octagon', 'ellipse', 'cross', 'arrow', 'heart']
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (255, 192, 203), (0, 128, 255), (128, 255, 0), (255, 20, 147)
    ]
    
    def add_noise(img):
        """Add subtle noise for realism"""
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    
    def create_background_pattern(img_size):
        """Create interesting background patterns"""
        bg = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        pattern_type = random.choice(['gradient', 'dots', 'lines', 'solid'])
        
        if pattern_type == 'gradient':
            for i in range(img_size):
                intensity = int(30 * i / img_size)
                bg[i, :] = [intensity, intensity//2, intensity//3]
        elif pattern_type == 'dots':
            for _ in range(20):
                x, y = random.randint(0, img_size-1), random.randint(0, img_size-1)
                cv2.circle(bg, (x, y), 3, (20, 20, 20), -1)
        elif pattern_type == 'lines':
            for _ in range(5):
                x1, y1 = random.randint(0, img_size), random.randint(0, img_size)
                x2, y2 = random.randint(0, img_size), random.randint(0, img_size)
                cv2.line(bg, (x1, y1), (x2, y2), (15, 15, 15), 1)
        
        return bg
    
    def rotate_shape(points, angle, center):
        """Rotate shape points around center"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        translated = points - center
        rotated = np.dot(translated, rotation_matrix.T)
        return rotated + center
    
    for i in range(500):  
        img_size = random.randint(150, 250)  
        img = create_background_pattern(img_size)
        
        shape = shapes[i % len(shapes)]
        color = random.choice(colors)
        
        margin = 50
        center_x = random.randint(margin, img_size - margin)
        center_y = random.randint(margin, img_size - margin)
      
        base_size = random.randint(20, 40)

        rotation = random.uniform(0, 2 * np.pi)
        
        if shape == 'circle':
            radius = base_size
            cv2.circle(img, (center_x, center_y), radius, color, -1)
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 2)
            
        elif shape == 'rectangle':
            half_w, half_h = base_size, int(base_size * random.uniform(0.5, 1.5))
            rect_pts = np.array([
                [-half_w, -half_h], [half_w, -half_h], 
                [half_w, half_h], [-half_w, half_h]
            ], dtype=np.float32)
            
            rotated_pts = rotate_shape(rect_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'triangle':
            tri_pts = np.array([
                [0, -base_size], [-base_size * 0.866, base_size * 0.5], 
                [base_size * 0.866, base_size * 0.5]
            ], dtype=np.float32)
            
            rotated_pts = rotate_shape(tri_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'pentagon':
            angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2  # Start from top
            pent_pts = np.array([[base_size * np.cos(a), base_size * np.sin(a)] for a in angles])
            
            rotated_pts = rotate_shape(pent_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'hexagon':
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_pts = np.array([[base_size * np.cos(a), base_size * np.sin(a)] for a in angles])
            
            rotated_pts = rotate_shape(hex_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'star':
            outer_radius = base_size
            inner_radius = base_size * 0.4
            star_pts = []
            
            for j in range(10):
                angle = j * np.pi / 5 - np.pi/2
                if j % 2 == 0:  # Outer points
                    r = outer_radius
                else:  # Inner points
                    r = inner_radius
                star_pts.append([r * np.cos(angle), r * np.sin(angle)])
            
            star_pts = np.array(star_pts, dtype=np.float32)
            
            rotated_pts = rotate_shape(star_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'trapezoid':
            top_width = base_size * 0.6
            bottom_width = base_size * 1.2
            height = base_size
            
            trap_pts = np.array([
                [-top_width, -height/2],      # top left
                [top_width, -height/2],       # top right
                [bottom_width, height/2],     # bottom right
                [-bottom_width, height/2]     # bottom left
            ], dtype=np.float32)
            
            rotated_pts = rotate_shape(trap_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'octagon':
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            oct_pts = np.array([[base_size * np.cos(a), base_size * np.sin(a)] for a in angles])
            
            rotated_pts = rotate_shape(oct_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'ellipse':
            axes_a = base_size
            axes_b = int(base_size * random.uniform(0.5, 1.5))
            angle_deg = int(np.degrees(rotation))
            cv2.ellipse(img, (center_x, center_y), (axes_a, axes_b), angle_deg, 0, 360, color, -1)
            cv2.ellipse(img, (center_x, center_y), (axes_a, axes_b), angle_deg, 0, 360, (255, 255, 255), 2)
            
        elif shape == 'cross':
            thickness = max(8, base_size // 3)
            arm_length = base_size
            
            h_pts = np.array([
                [-arm_length, -thickness//2], [arm_length, -thickness//2],
                [arm_length, thickness//2], [-arm_length, thickness//2]
            ], dtype=np.float32)
            
            v_pts = np.array([
                [-thickness//2, -arm_length], [thickness//2, -arm_length],
                [thickness//2, arm_length], [-thickness//2, arm_length]
            ], dtype=np.float32)
            
            h_rotated = rotate_shape(h_pts, rotation, np.array([0, 0]))
            h_rotated += np.array([center_x, center_y])
            h_rotated = h_rotated.astype(np.int32)
            
            v_rotated = rotate_shape(v_pts, rotation, np.array([0, 0]))
            v_rotated += np.array([center_x, center_y])
            v_rotated = v_rotated.astype(np.int32)
            
            cv2.fillPoly(img, [h_rotated], color)
            cv2.fillPoly(img, [v_rotated], color)
            cv2.polylines(img, [h_rotated], True, (255, 255, 255), 2)
            cv2.polylines(img, [v_rotated], True, (255, 255, 255), 2)
            
        elif shape == 'arrow':
            arrow_pts = np.array([
                [0, -base_size],  # tip
                [-base_size*0.5, -base_size*0.3],  # left wing
                [-base_size*0.2, -base_size*0.3],  # left shaft top
                [-base_size*0.2, base_size*0.5],   # left shaft bottom
                [base_size*0.2, base_size*0.5],    # right shaft bottom
                [base_size*0.2, -base_size*0.3],   # right shaft top
                [base_size*0.5, -base_size*0.3]    # right wing
            ], dtype=np.float32)
            
            rotated_pts = rotate_shape(arrow_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'heart':
            heart_pts = []
            for t in np.linspace(0, 2*np.pi, 20):
                x = base_size * 0.8 * (16 * np.sin(t)**3) / 16
                y = -base_size * 0.8 * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) / 16
                heart_pts.append([x, y])
            
            heart_pts = np.array(heart_pts, dtype=np.float32)
            rotated_pts = rotate_shape(heart_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)

        img = add_noise(img)
        
        if random.random() < 0.3:  # 30% chance
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(0, img_size), random.randint(0, img_size)
                x2, y2 = random.randint(x1-10, x1+10), random.randint(y1-10, y1+10)
                cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        filename = f"{shape}_{i:03d}.png"
        cv2.imwrite(os.path.join(dest_dir, filename), img)
        
        if (i + 1) % 50 == 0:
            console.print(f"🎯 Generated {i + 1}/500 super cool shapes!", style="green")
    
    console.print(Panel("🚀 SUPER COOL shape dataset created (500 diverse images with random rotations, positions, and effects)! 🚀", style="green"))

def download_colors():
    url = "https://raw.githubusercontent.com/codebrainz/color-names/master/output/colors.csv"
    dest_path = os.path.join(DATASETS_DIR, "colors.csv")
    if not os.path.exists(dest_path):
        console.print(Panel("Downloading color names CSV...", style="cyan"))
        df = pd.read_csv(url)
        df.to_csv(dest_path, index=False)
        console.print(Panel("Color names CSV downloaded!", style="green"))
    else:
        console.print(Panel("Color names CSV already exists.", style="green"))

def download_emnist():
    emnist_urls = [
        "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    ]
    
    dest_dir = os.path.join(DATASETS_DIR, "emnist")
    os.makedirs(dest_dir, exist_ok=True)
    dest_zip = os.path.join(dest_dir, "emnist_gzip.zip")
    
    if not os.path.exists(dest_zip):
        console.print(Panel("Downloading EMNIST dataset...", style="cyan"))
        
        for url in emnist_urls:
            try:
                console.print(f"Trying URL: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(dest_zip, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                console.print(Panel("EMNIST dataset downloaded successfully!", style="green"))
                return
                
            except Exception as e:
                console.print(f"Failed with {url}: {e}")
                continue
        
        console.print(Panel("EMNIST auto-download failed. Check DOWNLOAD_INSTRUCTIONS.txt", style="yellow"))
    else:
        console.print(Panel("EMNIST dataset already exists.", style="green"))

def create_mixed_dataset():
    """Create mixed dataset with shapes, letters, numbers, and colors for realistic camera testing"""
    import cv2
    import numpy as np
    import random
    import string
    
    console.print(Panel("🎭 Creating MIXED DATASET for realistic camera testing! 🎭", style="bold magenta"))
    
    dest_dir = os.path.join(DATASETS_DIR, "mixed_test")
    os.makedirs(dest_dir, exist_ok=True)

    shapes = ['circle', 'rectangle', 'triangle', 'pentagon', 'hexagon', 'star', 
              'trapezoid', 'octagon', 'ellipse', 'cross', 'arrow', 'heart']
    letters = list(string.ascii_uppercase)
    numbers = list('0123456789')
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (255, 192, 203), (0, 128, 255), (128, 255, 0), (255, 20, 147),
        (255, 165, 0), (128, 0, 128), (0, 128, 0), (255, 69, 0), (70, 130, 180), (220, 20, 60)
    ]
    
    def create_realistic_background(img_size):
        """Create realistic outdoor-like backgrounds"""
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
    
    def add_text_element(img, text, position, color, size_scale=1.0):
        """Add text (letter or number) to image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = size_scale * random.uniform(0.8, 1.5)
        thickness = random.randint(2, 4)
        
        cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(img, text, position, font, font_scale, color, thickness)
    
    def rotate_shape(points, angle, center):
        """Rotate shape points around center"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        translated = points - center
        rotated = np.dot(translated, rotation_matrix.T)
        return rotated + center
    
    def draw_shape(img, shape, center, size, color, rotation=0):
        """Draw a shape at specified position"""
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
            
            rotated_pts = rotate_shape(rect_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'triangle':
            tri_pts = np.array([
                [0, -size], [-size * 0.866, size * 0.5], 
                [size * 0.866, size * 0.5]
            ], dtype=np.float32)
            
            rotated_pts = rotate_shape(tri_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'pentagon':
            angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2
            pent_pts = np.array([[size * np.cos(a), size * np.sin(a)] for a in angles])
            
            rotated_pts = rotate_shape(pent_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'hexagon':
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_pts = np.array([[size * np.cos(a), size * np.sin(a)] for a in angles])
            
            rotated_pts = rotate_shape(hex_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'star':
            outer_radius = size
            inner_radius = size * 0.4
            star_pts = []
            
            for j in range(10):
                angle = j * np.pi / 5 - np.pi/2 + rotation
                if j % 2 == 0:
                    r = outer_radius
                else:
                    r = inner_radius
                star_pts.append([r * np.cos(angle), r * np.sin(angle)])
            
            star_pts = np.array(star_pts, dtype=np.float32)
            star_pts += np.array([center_x, center_y])
            star_pts = star_pts.astype(np.int32)
            
            cv2.fillPoly(img, [star_pts], color)
            cv2.polylines(img, [star_pts], True, (255, 255, 255), 2)
            
        elif shape == 'trapezoid':
            top_width = size * 0.6
            bottom_width = size * 1.2
            height = size
            
            trap_pts = np.array([
                [-top_width, -height/2],
                [top_width, -height/2],
                [bottom_width, height/2],
                [-bottom_width, height/2]
            ], dtype=np.float32)
            
            rotated_pts = rotate_shape(trap_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'octagon':
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            oct_pts = np.array([[size * np.cos(a), size * np.sin(a)] for a in angles])
            
            rotated_pts = rotate_shape(oct_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'ellipse':
            axes_a = size
            axes_b = int(size * random.uniform(0.5, 1.5))
            angle_deg = int(np.degrees(rotation))
            cv2.ellipse(img, (center_x, center_y), (axes_a, axes_b), angle_deg, 0, 360, color, -1)
            cv2.ellipse(img, (center_x, center_y), (axes_a, axes_b), angle_deg, 0, 360, (255, 255, 255), 2)
            
        elif shape == 'cross':
            thickness = max(8, size // 3)
            arm_length = size
            
            h_pts = np.array([
                [-arm_length, -thickness//2], [arm_length, -thickness//2],
                [arm_length, thickness//2], [-arm_length, thickness//2]
            ], dtype=np.float32)
            
            v_pts = np.array([
                [-thickness//2, -arm_length], [thickness//2, -arm_length],
                [thickness//2, arm_length], [-thickness//2, arm_length]
            ], dtype=np.float32)
            
            h_rotated = rotate_shape(h_pts, rotation, np.array([0, 0]))
            h_rotated += np.array([center_x, center_y])
            h_rotated = h_rotated.astype(np.int32)
            
            v_rotated = rotate_shape(v_pts, rotation, np.array([0, 0]))
            v_rotated += np.array([center_x, center_y])
            v_rotated = v_rotated.astype(np.int32)
            
            cv2.fillPoly(img, [h_rotated], color)
            cv2.fillPoly(img, [v_rotated], color)
            cv2.polylines(img, [h_rotated], True, (255, 255, 255), 2)
            cv2.polylines(img, [v_rotated], True, (255, 255, 255), 2)
            
        elif shape == 'arrow':
            arrow_pts = np.array([
                [0, -size],
                [-size*0.5, -size*0.3],
                [-size*0.2, -size*0.3],
                [-size*0.2, size*0.5],
                [size*0.2, size*0.5],
                [size*0.2, -size*0.3],
                [size*0.5, -size*0.3]
            ], dtype=np.float32)
            
            rotated_pts = rotate_shape(arrow_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'heart':
            heart_pts = []
            for t in np.linspace(0, 2*np.pi, 20):
                x = size * 0.8 * (16 * np.sin(t)**3) / 16
                y = -size * 0.8 * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) / 16
                heart_pts.append([x, y])
            
            heart_pts = np.array(heart_pts, dtype=np.float32)
            rotated_pts = rotate_shape(heart_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
    
    for i in range(200):
        img_size = random.randint(300, 500)  
        img = create_realistic_background(img_size)
        
        num_elements = random.randint(2, 5)
        used_positions = []
        
        elements_info = []
        
        for elem_idx in range(num_elements):
            element_type = random.choice(['shape', 'letter', 'number'])
            color = random.choice(colors)
            
            attempts = 0
            while attempts < 50:  
                if element_type == 'shape':
                    size = random.randint(25, 45)
                    margin = size + 10
                else:  # text elements
                    size = random.randint(30, 50)
                    margin = size + 15
                
                center_x = random.randint(margin, img_size - margin)
                center_y = random.randint(margin, img_size - margin)
                
                overlap = False
                for used_pos in used_positions:
                    dist = np.sqrt((center_x - used_pos[0])**2 + (center_y - used_pos[1])**2)
                    if dist < (margin + used_pos[2]):  
                        overlap = True
                        break
                
                if not overlap:
                    used_positions.append((center_x, center_y, margin))
                    break
                attempts += 1
            
            if attempts >= 50:  
                continue
            rotation = random.uniform(0, 2 * np.pi)
            
            if element_type == 'shape':
                shape = random.choice(shapes)
                draw_shape(img, shape, (center_x, center_y), size, color, rotation)
                elements_info.append(f"{shape}")
                
            elif element_type == 'letter':
                letter = random.choice(letters)

                text_pos = (center_x - 15, center_y + 15)
                add_text_element(img, letter, text_pos, color, size/50.0)
                elements_info.append(f"L{letter}")
                
            elif element_type == 'number':
                number = random.choice(numbers)
                
                text_pos = (center_x - 15, center_y + 15)
                add_text_element(img, number, text_pos, color, size/50.0)
                elements_info.append(f"N{number}")
        
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        if random.random() < 0.4:  # 40% chance
            for _ in range(random.randint(1, 3)):
                pt1 = (random.randint(0, img_size), random.randint(0, img_size))
                pt2 = (random.randint(0, img_size), random.randint(0, img_size))
                cv2.line(img, pt1, pt2, (random.randint(50, 150),) * 3, 1)
        
        elements_str = "_".join(elements_info[:3])  
        filename = f"mixed_{i:03d}_{elements_str}.png"
        cv2.imwrite(os.path.join(dest_dir, filename), img)
        
        if (i + 1) % 25 == 0:
            console.print(f"🎯 Generated {i + 1}/200 mixed test images!", style="green")
    
    console.print(Panel("🚀 MIXED TEST DATASET created (200 realistic images with multiple elements)! 🚀", style="green"))

if __name__ == "__main__":
    console.print(Panel.fit("🎯 VTOL VISION DATASET DOWNLOADER 🎯", style="bold magenta"))
    console.print("🚀 Preparing datasets for shape, color, letter, and number detection!\n")
    
    with Progress(
        SpinnerColumn(), 
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Shape Dataset
        shape_task = progress.add_task(description="[bold cyan]🎨 Creating Super Cool Shapes...", total=None)
        try:
            download_shapes()
            progress.update(shape_task, description="[bold green]✅ Shape Dataset Complete!")
        except Exception as e:
            console.print(Panel(f"[red]❌ Error generating shapes: {e}", style="red"))
            progress.update(shape_task, description="[bold red]❌ Shape Dataset Failed!")
        
        # Color Dataset  
        color_task = progress.add_task(description="[bold yellow]🌈 Downloading Color Names...", total=None)
        try:
            download_colors()
            progress.update(color_task, description="[bold green]✅ Color Dataset Complete!")
        except Exception as e:
            console.print(Panel(f"[red]❌ Error downloading colors: {e}", style="red"))
            progress.update(color_task, description="[bold red]❌ Color Dataset Failed!")
        
        # EMNIST Dataset
        emnist_task = progress.add_task(description="[bold blue]🔤 Downloading EMNIST (Letters & Numbers)...", total=None)
        try:
            download_emnist()
            progress.update(emnist_task, description="[bold green]✅ EMNIST Dataset Complete!")
        except Exception as e:
            console.print(Panel(f"[red]❌ Error downloading EMNIST: {e}", style="red"))
            progress.update(emnist_task, description="[bold red]❌ EMNIST Dataset Failed!")
        
        # Mixed Test Dataset
        mixed_task = progress.add_task(description="[bold purple]🎭 Creating Mixed Test Dataset...", total=None)
        try:
            create_mixed_dataset()
            progress.update(mixed_task, description="[bold green]✅ Mixed Test Dataset Complete!")
        except Exception as e:
            console.print(Panel(f"[red]❌ Error creating mixed dataset: {e}", style="red"))
            progress.update(mixed_task, description="[bold red]❌ Mixed Test Dataset Failed!")
    
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]🎉 ALL DATASETS ARE READY! 🎉\n"
        "📁 Check the Datasets folder for:\n"
        "   🔺 Shapes (500 diverse images with 12 shape types + rotations)\n"
        "   🌈 Colors (CSV for OpenCV detection)\n"
        "   🔤 EMNIST (Letters & Numbers)\n"
        "   🎭 Mixed Test (200 realistic multi-element images)\n\n"
        "🎯 Shape Types: Circle, Rectangle, Triangle, Pentagon, Hexagon,\n"
        "               Star, Trapezoid, Octagon, Ellipse, Cross, Arrow, Heart\n\n"
        "🚀 Ready for YOLO training on Raspberry Pi!\n"
        "📸 Perfect for camera testing with mixed elements!", 
        style="green"
    ))
    console.print("="*60)
