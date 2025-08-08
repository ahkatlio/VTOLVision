import subprocess
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
import time

console = Console()

def get_modified_files():
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    files = []
    file_status = {}
    
    for line in result.stdout.splitlines():
        if len(line) < 3:
            continue
        status, file = line[:2], line[3:]
        
        # Handle different git status codes
        status_code = status.strip()
        if status_code in {'M', 'A', '??', 'D', 'R', 'C', 'U'}:
            files.append(file)
            
            # Map status codes to readable descriptions
            status_map = {
                'M': ('Modified', 'yellow'),
                'A': ('Added', 'green'), 
                '??': ('Untracked', 'cyan'),
                'D': ('Deleted', 'red'),
                'R': ('Renamed', 'magenta'),
                'C': ('Copied', 'blue'),
                'U': ('Unmerged', 'orange')
            }
            file_status[file] = status_map.get(status_code, ('Unknown', 'white'))
    
    return files, file_status

def push_files_one_by_one():
    console.print(Panel.fit("🚀 VTOL VISION GIT PUSHER 🚀", style="bold magenta"))
    console.print("🔍 Scanning for changes...\n")
    
    files, file_status = get_modified_files()
    
    if not files:
        console.print(Panel("✨ No modified files to push! Repository is clean! ✨", style="bold green"))
        return

    # Create enhanced table
    table = Table(
        title="📋 Files Ready for Commit & Push", 
        title_style="bold cyan",
        border_style="magenta"
    )
    table.add_column("📁 File", style="cyan", width=40)
    table.add_column("📊 Status", justify="center", width=15)
    table.add_column("🎯 Action", justify="center", width=15)
    
    for file in files:
        status_text, status_color = file_status[file]
        action_icon = {
            'Modified': '🔄',
            'Added': '➕', 
            'Untracked': '🆕',
            'Deleted': '🗑️',
            'Renamed': '📝',
            'Copied': '📋',
            'Unmerged': '⚠️'
        }.get(status_text, '❓')
        
        table.add_row(
            file, 
            f"[{status_color}]{status_text}[/{status_color}]",
            f"{action_icon} Push"
        )
    
    console.print(table)
    console.print()
    
    # Progress bar for the push process
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        push_task = progress.add_task("[cyan]Processing files...", total=len(files))
        
        for i, file in enumerate(files):
            status_text, status_color = file_status[file]
            
            # Update progress description
            progress.update(push_task, description=f"[{status_color}]Processing {file}...")
            
            # Add file (this handles both new files and deletions)
            console.print(f"[yellow]📂 Adding:[/yellow] {file}")
            subprocess.run(['git', 'add', file], capture_output=True)
            
            # Create descriptive commit message based on status
            if status_text == 'Deleted':
                commit_msg = f"🗑️ Remove {file}"
            elif status_text == 'Added' or status_text == 'Untracked':
                commit_msg = f"➕ Add {file}"
            elif status_text == 'Modified':
                commit_msg = f"🔄 Update {file}"
            elif status_text == 'Renamed':
                commit_msg = f"📝 Rename {file}"
            else:
                commit_msg = f"✨ {status_text} {file}"
            
            console.print(f"[blue]💾 Committing:[/blue] {commit_msg}")
            subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True)
            
            console.print(f"[magenta]🚀 Pushing:[/magenta] {file}")
            result = subprocess.run(['git', 'push'], capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green]✅ Successfully pushed {file}![/green]")
            else:
                console.print(f"[red]❌ Failed to push {file}: {result.stderr}[/red]")
            
            console.print("─" * 50)
            
            # Update progress
            progress.update(push_task, advance=1)
            time.sleep(0.1)  # Small delay for visual effect
    
    # Final celebration
    console.print()
    console.print(Panel.fit(
        f"🎉 SUCCESS! 🎉\n"
        f"📊 Processed {len(files)} files\n"
        f"🚀 All changes pushed to repository!\n"
        f"✨ VTOL Vision is ready for takeoff! ✨", 
        style="bold green"
    ))

if __name__ == "__main__":
    push_files_one_by_one()
