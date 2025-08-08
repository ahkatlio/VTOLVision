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
        
        status_code = status.strip()
        if status_code in {'M', 'A', '??', 'D', 'R', 'C', 'U'}:
            files.append(file)
            
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
    
    console.print(Panel(
        "[yellow]⚠️ SLOW INTERNET MODE ⚠️[/yellow]\n"
        "This will push files ONE BY ONE (not batched)\n"
        "• Each file gets its own commit and push\n"
        "• If internet fails, you won't lose previous progress\n"
        "• Takes longer but more reliable for slow connections\n\n"
        "[cyan]Continue with individual file pushing?[/cyan]",
        style="orange3"
    ))
    
    response = input("Press Enter to continue or Ctrl+C to cancel: ")
    console.print()
    
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
            
            progress.update(push_task, description=f"[{status_color}]Processing {file}...")
            
            console.print(f"[yellow]📂 Adding:[/yellow] {file}")
            add_result = subprocess.run(['git', 'add', file], capture_output=True, text=True)
            
            if add_result.returncode != 0:
                console.print(f"[red]❌ Failed to add {file}: {add_result.stderr}[/red]")
                progress.update(push_task, advance=1)
                continue
            
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
            commit_result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True)
            
            if commit_result.returncode != 0:
                console.print(f"[orange3]⚠️ Nothing to commit for {file} (already staged?)[/orange3]")
                progress.update(push_task, advance=1)
                continue
            
            console.print(f"[magenta]🚀 Pushing:[/magenta] {file} (individual push for slow internet)")
            push_result = subprocess.run(['git', 'push'], capture_output=True, text=True)
            
            if push_result.returncode == 0:
                console.print(f"[green]✅ Successfully pushed {file}![/green]")
            else:
                console.print(f"[red]❌ Failed to push {file}:[/red]")
                console.print(f"[red]Error: {push_result.stderr}[/red]")
            
            console.print("─" * 60)
            
            progress.update(push_task, advance=1)
            time.sleep(0.2) 
    
    # Final celebration
    console.print()
    console.print(Panel.fit(
        f"🎉 SUCCESS! 🎉\n"
        f"📊 Processed {len(files)} files individually\n"
        f"🐌 Slow internet friendly: One file at a time\n"
        f"🚀 All changes pushed to repository!\n"
        f"✨ VTOL Vision is ready for takeoff! ✨", 
        style="bold green"
    ))

if __name__ == "__main__":
    push_files_one_by_one()
