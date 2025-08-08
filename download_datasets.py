import os
import subprocess
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
    if not os.path.exists(repo_dir):
        console.print(Panel("Cloning 2DGeometricShapesGenerator...", style="cyan"))
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    else:
        console.print(Panel("2DGeometricShapesGenerator already cloned.", style="green"))
    
    # Generate shapes
    dest_dir = os.path.join(DATASETS_DIR, "shapes")
    os.makedirs(dest_dir, exist_ok=True)
    console.print(Panel(f"Generating 10,000 shapes in {dest_dir}...", style="cyan"))
    subprocess.run([
        "python", os.path.join(repo_dir, "shape_generator.py"),
        "generate-shapes", "--size=10000", f"--destination={dest_dir}"
    ], check=True)
    console.print(Panel("Shape dataset ready!", style="green"))

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
    emnist_url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
    dest_zip = os.path.join(DATASETS_DIR, "emnist_gzip.zip")
    if not os.path.exists(dest_zip):
        console.print(Panel("Downloading EMNIST ByClass dataset...", style="cyan"))
        with requests.get(emnist_url, stream=True) as r:
            r.raise_for_status()
            with open(dest_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        console.print(Panel("EMNIST dataset downloaded!", style="green"))
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
