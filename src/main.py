"""
Main entry point for the application
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication
from src.ui.setup_window import SetupWindow
from src.ui.session_manager import SessionManager

def main():
    """Main entry point"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
