import subprocess
from rich.console import Console
from rich.table import Table

console = Console()

def get_modified_files():
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    files = []
    for line in result.stdout.splitlines():
        status, file = line[:2], line[3:]
        if status.strip() in {'M', 'A', '??'}:
            files.append(file)
    return files

def push_files_one_by_one():
    files = get_modified_files()
    if not files:
        console.print('[bold green]No modified files to push![/bold green]')
        return

    table = Table(title="Modified Files to Push")
    table.add_column("File", style="cyan")
    for file in files:
        table.add_row(file)
    console.print(table)

    for file in files:
        console.print(f"[yellow]Adding:[/yellow] {file}")
        subprocess.run(['git', 'add', file])
        commit_msg = f"Update {file}"
        console.print(f"[blue]Committing:[/blue] {commit_msg}")
        subprocess.run(['git', 'commit', '-m', commit_msg])
        console.print(f"[magenta]Pushing:[/magenta] {file}")
        subprocess.run(['git', 'push'])
        console.print(f"[green]Pushed {file}![/green]\n")

if __name__ == "__main__":
    push_files_one_by_one()
