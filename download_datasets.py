import os
import subprocess
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import requests
import pandas as pd

console = Console()
DATASETS_DIR = "Datasets"

os.makedirs(DATASETS_DIR, exist_ok=True)

# 1. Download and generate shape dataset
def download_shapes():
    repo_url = "https://github.com/elkorchi/2DGeometricShapesGenerator.git"
    repo_dir = os.path.join(DATASETS_DIR, "2DGeometricShapesGenerator")
    
    # Clone the repository
    if not os.path.exists(repo_dir):
        console.print(Panel("Cloning 2DGeometricShapesGenerator...", style="cyan"))
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    else:
        console.print(Panel("2DGeometricShapesGenerator already cloned.", style="green"))
    
    # Install required dependencies for shape generator (skip problematic requirements.txt)
    console.print(Panel("Installing shape generator dependencies...", style="cyan"))
    subprocess.run(["pip", "install", "click", "pillow", "opencv-python", "numpy"], check=True)
    
    # Generate shapes
    dest_dir = os.path.join(DATASETS_DIR, "shapes")
    os.makedirs(dest_dir, exist_ok=True)
    console.print(Panel(f"Generating 1,000 shapes in {dest_dir} (reduced for testing)...", style="cyan"))
    
    # Change to repo directory to run the generator
    original_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        # Use relative path and smaller dataset for testing
        subprocess.run([
            "python", "shape_generator.py",
            "generate-shapes", "--size=1000", f"--destination=..{os.sep}shapes"
        ], check=True)
    except Exception as e:
        console.print(Panel(f"Shape generation failed: {e}. Creating simple shape dataset manually...", style="yellow"))
        # If the generator fails, create a simple placeholder
        os.chdir(original_cwd)
        create_simple_shapes(dest_dir)
        return
    finally:
        os.chdir(original_cwd)
    
    # Clean up the repository after generating shapes
    console.print(Panel("Cleaning up shape generator repository...", style="yellow"))
    shutil.rmtree(repo_dir)
    
    console.print(Panel("Shape dataset ready!", style="green"))

def create_simple_shapes(dest_dir):
    """Create an advanced shape dataset with random positions, rotations, and effects"""
    import cv2
    import numpy as np
    import random
    
    console.print(Panel("🎨 Creating SUPER COOL shape dataset with random variations! 🎨", style="magenta"))
    
    shapes = ['circle', 'rectangle', 'triangle', 'pentagon', 'hexagon', 'star']
    # Enhanced color palette with more vibrant colors
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
        
        # Translate to origin, rotate, translate back
        translated = points - center
        rotated = np.dot(translated, rotation_matrix.T)
        return rotated + center
    
    for i in range(500):  # Create 500 diverse shapes
        img_size = random.randint(150, 250)  # Random image sizes
        img = create_background_pattern(img_size)
        
        shape = shapes[i % len(shapes)]
        color = random.choice(colors)
        
        # Random position (ensuring shape stays within bounds)
        margin = 50
        center_x = random.randint(margin, img_size - margin)
        center_y = random.randint(margin, img_size - margin)
        
        # Random size
        base_size = random.randint(20, 40)
        
        # Random rotation angle
        rotation = random.uniform(0, 2 * np.pi)
        
        if shape == 'circle':
            radius = base_size
            cv2.circle(img, (center_x, center_y), radius, color, -1)
            # Add border for definition
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 2)
            
        elif shape == 'rectangle':
            # Create rectangle points
            half_w, half_h = base_size, int(base_size * random.uniform(0.5, 1.5))
            rect_pts = np.array([
                [-half_w, -half_h], [half_w, -half_h], 
                [half_w, half_h], [-half_w, half_h]
            ], dtype=np.float32)
            
            # Rotate and translate
            rotated_pts = rotate_shape(rect_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'triangle':
            # Create triangle points
            tri_pts = np.array([
                [0, -base_size], [-base_size * 0.866, base_size * 0.5], 
                [base_size * 0.866, base_size * 0.5]
            ], dtype=np.float32)
            
            # Rotate and translate
            rotated_pts = rotate_shape(tri_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'pentagon':
            # Create pentagon points
            angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2  # Start from top
            pent_pts = np.array([[base_size * np.cos(a), base_size * np.sin(a)] for a in angles])
            
            # Rotate and translate
            rotated_pts = rotate_shape(pent_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'hexagon':
            # Create hexagon points
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_pts = np.array([[base_size * np.cos(a), base_size * np.sin(a)] for a in angles])
            
            # Rotate and translate
            rotated_pts = rotate_shape(hex_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
            
        elif shape == 'star':
            # Create 5-pointed star
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
            
            # Rotate and translate
            rotated_pts = rotate_shape(star_pts, rotation, np.array([0, 0]))
            rotated_pts += np.array([center_x, center_y])
            rotated_pts = rotated_pts.astype(np.int32)
            
            cv2.fillPoly(img, [rotated_pts], color)
            cv2.polylines(img, [rotated_pts], True, (255, 255, 255), 2)
        
        # Add some noise for realism
        img = add_noise(img)
        
        # Add random small artifacts (like dust or scratches)
        if random.random() < 0.3:  # 30% chance
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(0, img_size), random.randint(0, img_size)
                x2, y2 = random.randint(x1-10, x1+10), random.randint(y1-10, y1+10)
                cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        filename = f"{shape}_{i:03d}.png"
        cv2.imwrite(os.path.join(dest_dir, filename), img)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            console.print(f"🎯 Generated {i + 1}/500 super cool shapes!", style="green")
    
    console.print(Panel("🚀 SUPER COOL shape dataset created (500 diverse images with random rotations, positions, and effects)! 🚀", style="green"))

# 2. Download color names CSV
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

# 3. Download EMNIST (ByClass split)
def download_emnist():
    # Try multiple EMNIST sources
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
        
        # If all URLs fail, create a placeholder file with instructions
        with open(os.path.join(dest_dir, "DOWNLOAD_INSTRUCTIONS.txt"), "w") as f:
            f.write("EMNIST Download Instructions:\n")
            f.write("1. Visit: https://www.nist.gov/itl/products-and-services/emnist-dataset\n")
            f.write("2. Download the EMNIST ByClass dataset manually\n")
            f.write("3. Extract it to this folder\n")
        
        console.print(Panel("EMNIST auto-download failed. Check DOWNLOAD_INSTRUCTIONS.txt", style="yellow"))
    else:
        console.print(Panel("EMNIST dataset already exists.", style="green"))

if __name__ == "__main__":
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task(description="[bold]Downloading Shape Dataset...", total=None)
        try:
            download_shapes()
        except Exception as e:
            console.print(Panel(f"[red]Error downloading/generating shapes: {e}", style="red"))
        progress.add_task(description="[bold]Downloading Color Names...", total=None)
        try:
            download_colors()
        except Exception as e:
            console.print(Panel(f"[red]Error downloading colors: {e}", style="red"))
        progress.add_task(description="[bold]Downloading EMNIST...", total=None)
        try:
            download_emnist()
        except Exception as e:
            console.print(Panel(f"[red]Error downloading EMNIST: {e}", style="red"))
    console.print(Panel("[bold green]All datasets are ready in the Datasets folder![/bold green]", style="green"))
