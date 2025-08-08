#!/usr/bin/env python3
"""
VTOL Vision Project Action Plan Tracker

This script generates and manages the action plan markdown file.
It tracks completion status and provides progress reports.

Usage:
    python action_plan_tracker.py
"""

import os
import re
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

class ActionPlanTracker:
    def __init__(self, markdown_file="action_plan.md"):
        self.markdown_file = markdown_file
        self.action_plan_template = """# 📋 VTOL Vision Project Action Plan

**Generated on:** {timestamp}

## 🎯 PHASE 1: Model Selection & Setup (Week 1)
- [x] Dataset generation complete
- [x] Analysis complete
- [x] Install YOLOv8 (Ultralytics)
- [x] Test model variants on development machine
- [x] Benchmark performance vs accuracy
- [x] Select final model (recommended: YOLOv8n)

## 🔧 PHASE 2: Data Preparation (Week 2)
- [ ] Create YOLO annotation format
- [ ] Generate training/validation/test splits
- [ ] Implement data augmentation pipeline
- [ ] Create dataset YAML configuration
- [ ] Validate data quality and format

## 🚀 PHASE 3: Model Training (Week 3)
- [ ] Set up training environment
- [ ] Configure hyperparameters
- [ ] Train initial model (100 epochs)
- [ ] Monitor training metrics
- [ ] Validate on mixed test dataset
- [ ] Fine-tune if needed

## 📱 PHASE 4: Raspberry Pi Deployment (Week 4)
- [ ] Set up Raspberry Pi environment
- [ ] Install optimized PyTorch/YOLO
- [ ] Port model to RPi
- [ ] Implement real-time detection pipeline
- [ ] Test with camera
- [ ] Optimize performance

## 🏁 PHASE 5: Integration & Testing (Week 5)
- [ ] Integrate with VTOL control system
- [ ] Field testing
- [ ] Performance tuning
- [ ] Competition preparation
- [ ] Documentation

## 📊 Success Metrics
- Detection accuracy >85% on test set
- Real-time performance >10 FPS on RPi4
- Memory usage <400MB
- Robust outdoor performance
- Multi-object detection capability

## 🔍 Next Immediate Actions
1. Research and install YOLOv8
2. Create annotation generation script
3. Set up training environment
4. Begin model training experiments

---
**Progress Tracking:** Run `python action_plan_tracker.py` to see current completion status.
"""

    def generate_markdown(self):
        """Generate the initial action plan markdown file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = self.action_plan_template.format(timestamp=timestamp)
        
        with open(self.markdown_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        console.print(f"✅ Action plan generated: [bold cyan]{self.markdown_file}[/bold cyan]")
        return True

    def parse_markdown(self):
        """Parse the markdown file and extract task information"""
        if not os.path.exists(self.markdown_file):
            console.print(f"❌ File not found: {self.markdown_file}")
            console.print("💡 Run with --generate to create the file first")
            return None
        
        with open(self.markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all checkbox items
        checkbox_pattern = r'^- \[([ x])\] (.+)$'
        matches = re.findall(checkbox_pattern, content, re.MULTILINE)
        
        # Extract phase information
        phase_pattern = r'^## ([🎯🔧🚀📱🏁].*?) \(Week \d+\)$'
        phases = re.findall(phase_pattern, content, re.MULTILINE)
        
        tasks = []
        current_phase = "Unknown"
        
        lines = content.split('\n')
        for line in lines:
            phase_match = re.match(phase_pattern, line)
            if phase_match:
                current_phase = phase_match.group(1)
            
            checkbox_match = re.match(checkbox_pattern, line)
            if checkbox_match:
                status, task = checkbox_match.groups()
                completed = status == 'x'
                tasks.append({
                    'phase': current_phase,
                    'task': task.strip(),
                    'completed': completed
                })
        
        return tasks

    def calculate_progress(self, tasks):
        """Calculate overall and per-phase progress"""
        if not tasks:
            return {}
        
        progress = {
            'overall': {'completed': 0, 'total': 0, 'percentage': 0},
            'phases': {}
        }
        
        # Group tasks by phase
        phase_tasks = {}
        for task in tasks:
            phase = task['phase']
            if phase not in phase_tasks:
                phase_tasks[phase] = []
            phase_tasks[phase].append(task)
        
        # Calculate progress for each phase
        total_completed = 0
        total_tasks = 0
        
        for phase, phase_task_list in phase_tasks.items():
            completed = sum(1 for t in phase_task_list if t['completed'])
            total = len(phase_task_list)
            percentage = (completed / total * 100) if total > 0 else 0
            
            progress['phases'][phase] = {
                'completed': completed,
                'total': total,
                'percentage': percentage
            }
            
            total_completed += completed
            total_tasks += total
        
        # Calculate overall progress
        overall_percentage = (total_completed / total_tasks * 100) if total_tasks > 0 else 0
        progress['overall'] = {
            'completed': total_completed,
            'total': total_tasks,
            'percentage': overall_percentage
        }
        
        return progress

    def display_progress(self):
        """Display the current progress in a nice format"""
        tasks = self.parse_markdown()
        if not tasks:
            return False
        
        progress = self.calculate_progress(tasks)
        
        # Overall progress panel
        overall = progress['overall']
        console.print(Panel(
            f"[bold green]Overall Progress: {overall['completed']}/{overall['total']} tasks completed[/bold green]\n"
            f"[bold cyan]Completion: {overall['percentage']:.1f}%[/bold cyan]",
            title="🎯 VTOL Vision Project Status",
            border_style="green"
        ))
        
        # Phase-wise progress table
        table = Table(title="📊 Phase-wise Progress")
        table.add_column("Phase", style="cyan", width=40)
        table.add_column("Progress", style="green", width=15)
        table.add_column("Status Bar", style="blue", width=30)
        table.add_column("Percentage", style="yellow", width=10)
        
        for phase, phase_progress in progress['phases'].items():
            completed = phase_progress['completed']
            total = phase_progress['total']
            percentage = phase_progress['percentage']
            
            # Create a simple progress bar
            bar_length = 20
            filled_length = int(bar_length * percentage / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            # Color coding based on completion
            if percentage == 100:
                status_style = "bold green"
            elif percentage >= 50:
                status_style = "bold yellow"
            else:
                status_style = "bold red"
            
            table.add_row(
                phase,
                f"{completed}/{total}",
                f"[{status_style}]{bar}[/{status_style}]",
                f"{percentage:.1f}%"
            )
        
        console.print(table)
        
        # Detailed task breakdown
        console.print("\n📋 [bold]Detailed Task Status:[/bold]")
        
        current_phase = None
        for task in tasks:
            if task['phase'] != current_phase:
                current_phase = task['phase']
                console.print(f"\n[bold cyan]{current_phase}[/bold cyan]")
            
            status_icon = "✅" if task['completed'] else "⭕"
            status_color = "green" if task['completed'] else "red"
            console.print(f"  {status_icon} [{status_color}]{task['task']}[/{status_color}]")
        
        # Next actions suggestion
        incomplete_tasks = [t for t in tasks if not t['completed']]
        if incomplete_tasks:
            console.print(f"\n🔍 [bold]Next Action:[/bold] [yellow]{incomplete_tasks[0]['task']}[/yellow]")
        else:
            console.print(f"\n🎉 [bold green]All tasks completed! Great job![/bold green]")
        
        return True

    def update_file_timestamp(self):
        """Update the timestamp in the markdown file"""
        if not os.path.exists(self.markdown_file):
            return False
        
        with open(self.markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_pattern = r'\*\*Generated on:\*\* .+'
        updated_content = re.sub(
            timestamp_pattern,
            f"**Generated on:** {timestamp}",
            content
        )
        
        # Add last updated line if not present
        if "**Last updated:**" not in updated_content:
            updated_content = updated_content.replace(
                f"**Generated on:** {timestamp}",
                f"**Generated on:** {timestamp}  \n**Last updated:** {timestamp}"
            )
        else:
            updated_content = re.sub(
                r'\*\*Last updated:\*\* .+',
                f"**Last updated:** {timestamp}",
                updated_content
            )
        
        with open(self.markdown_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VTOL Vision Project Action Plan Tracker')
    parser.add_argument('--generate', action='store_true', 
                       help='Generate a new action plan markdown file')
    parser.add_argument('--file', default='action_plan.md',
                       help='Markdown file to use (default: action_plan.md)')
    
    args = parser.parse_args()
    
    tracker = ActionPlanTracker(args.file)
    
    if args.generate:
        tracker.generate_markdown()
        console.print("📝 [yellow]Edit the file and mark tasks as complete with [x][/yellow]")
        console.print("🔄 [cyan]Run the script again (without --generate) to see progress[/cyan]")
    else:
        success = tracker.display_progress()
        if success:
            tracker.update_file_timestamp()
        else:
            console.print("💡 [yellow]Use --generate flag to create the action plan file first[/yellow]")

if __name__ == "__main__":
    main()
