#!/usr/bin/env python3

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
import pyqtgraph as pg
import numpy as np

def main():
    # Create the application
    app = QApplication(sys.argv)
    
    # Configure pyqtgraph
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    
    # Create the main window
    window = QMainWindow()
    window.setWindowTitle("Minimal Qt Test")
    window.setGeometry(100, 100, 800, 600)
    
    # Create a central widget and layout
    central = QWidget()
    window.setCentralWidget(central)
    
    # Create a layout
    layout = QVBoxLayout(central)
    
    # Add a label
    label = QLabel("Test Plot")
    label.setStyleSheet("font-size: 24pt; color: #2c3e50;")
    layout.addWidget(label)
    
    # Create a plot
    plot = pg.PlotWidget(title="Test Plot")
    plot.setBackground('w')
    plot.setLabel('left', 'Value')
    plot.setLabel('bottom', 'Time (s)')
    plot.showGrid(x=True, y=True)
    
    # Generate some test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Plot the data with a red line
    plot.plot(x, y, pen=pg.mkPen(color='r', width=2))
    
    # Add plot to layout
    layout.addWidget(plot)
    
    # Show the window
    window.show()
    
    # Start the event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
