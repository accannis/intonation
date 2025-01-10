"""
Main entry point for the application
"""

import sys
import signal
import psutil
import os
import logging
from PyQt6.QtWidgets import QApplication
from src.ui.setup_window import SetupWindow
from src.ui.session_manager import SessionManager

def kill_other_instances():
    """Kill other running instances of this application"""
    current_pid = os.getpid()
    current_process = psutil.Process(current_pid)
    app_name = current_process.name()
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # If it's a Python process
            if proc.info['name'] == app_name and proc.pid != current_pid:
                # Check if it's running our script
                cmdline = proc.cmdline()
                if any('src.main' in arg for arg in cmdline):
                    logging.info(f"Killing previous instance (PID: {proc.pid})")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def main():
    """Main entry point"""
    try:
        # Kill other instances first
        kill_other_instances()
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for more verbose output
            format='%(levelname)s:%(name)s:%(message)s'
        )
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create session manager
        session_manager = SessionManager()
        
        # Create and show setup window
        setup_window = SetupWindow(session_manager)
        setup_window.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)

if __name__ == '__main__':
    main()
