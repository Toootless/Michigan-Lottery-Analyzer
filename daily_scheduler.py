#!/usr/bin/env python3
"""
Daily Scheduler for Michigan Lottery Data System
Automatically runs daily updates and can be scheduled via Windows Task Scheduler
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from complete_lottery_system import CompleteLotterySystem

def setup_logging():
    """Setup logging for scheduled runs"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'daily_scheduler.log'),
            logging.StreamHandler()
        ]
    )

def run_daily_update():
    """Run the daily lottery update"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting daily lottery update")
        
        # Initialize system
        system = CompleteLotterySystem()
        
        # Run daily update
        update_result = system.daily_update()
        
        # Log results
        logger.info(f"✅ Daily update completed successfully")
        logger.info(f"Total added: {update_result['total_added']}")
        logger.info(f"Total updated: {update_result['total_updated']}")
        
        # Check for any failures
        failed_games = [game for game, stats in update_result['games'].items() 
                       if stats['status'] in ['error', 'no_data']]
        
        if failed_games:
            logger.warning(f"⚠️ Some games failed: {failed_games}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Daily update failed: {e}")
        return False

def create_windows_task_command():
    """Generate Windows Task Scheduler command"""
    current_file = Path(__file__).absolute()
    python_exe = sys.executable
    
    command = f'"{python_exe}" "{current_file}"'
    
    print("🔧 Windows Task Scheduler Setup:")
    print("=" * 50)
    print("1. Open Task Scheduler (taskschd.msc)")
    print("2. Create Basic Task...")
    print("3. Name: 'Michigan Lottery Daily Update'")
    print("4. Trigger: Daily at 10:00 PM")
    print("5. Action: Start a program")
    print(f"6. Program: {python_exe}")
    print(f"7. Arguments: \"{current_file}\"")
    print(f"8. Start in: {current_dir}")
    print()
    print("Or use this PowerShell command to create the task:")
    print("-" * 50)
    
    powershell_command = f'''
$Action = New-ScheduledTaskAction -Execute "{python_exe}" -Argument '"{current_file}"' -WorkingDirectory "{current_dir}"
$Trigger = New-ScheduledTaskTrigger -Daily -At "10:00 PM"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName "Michigan Lottery Daily Update" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force
'''
    
    print(powershell_command.strip())

def main():
    """Main function"""
    setup_logging()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        create_windows_task_command()
        return
    
    # Run daily update
    success = run_daily_update()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()