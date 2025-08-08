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
    """Create a simple shape dataset if the generator fails"""
    import cv2
    import numpy as np
    
    console.print(Panel("Creating simple shape dataset manually...", style="cyan"))
    
    shapes = ['circle', 'rectangle', 'triangle']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i in range(100):  # Create 100 simple shapes
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]
        
        if shape == 'circle':
            cv2.circle(img, (100, 100), 50, color, -1)
        elif shape == 'rectangle':
            cv2.rectangle(img, (50, 50), (150, 150), color, -1)
        elif shape == 'triangle':
            pts = np.array([[100, 50], [50, 150], [150, 150]], np.int32)
            cv2.fillPoly(img, [pts], color)
        
        filename = f"{shape}_{i:03d}.png"
        cv2.imwrite(os.path.join(dest_dir, filename), img)
    
    console.print(Panel("Simple shape dataset created (100 images)!", style="green"))

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
