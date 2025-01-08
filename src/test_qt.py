import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

def main():
    # Create the application
    app = QApplication(sys.argv)
    
    # Create the main window
    window = QMainWindow()
    window.setWindowTitle("Test Window")
    window.setGeometry(100, 100, 400, 200)
    
    # Create central widget and layout
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    # Add a label
    label = QLabel("Hello World!")
    layout.addWidget(label)
    
    # Show the window
    window.show()
    
    # Start the event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
