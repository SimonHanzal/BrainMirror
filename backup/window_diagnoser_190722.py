# Importing core packages and needed PyQt Widgets

import sys

from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6.QtCore import Qt, QTimer

# Establishing the colour palette
colours = [0] * 256
for x in range(0, 256):
    colours[x] = "#{:02x}{:02x}{:02x}".format(10, 10, x)
# This variables specifies how many colours will be used, +1
refinement = 255
background_text = "background-color: "


# The main window
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Formatting
        self.setStyleSheet(''.join([background_text, colours[0]]))
        self.setWindowTitle("BrainMirror")
        self.showMaximized()
        self.showFullScreen()
        self.show()
        # Counter variable
        self.clrs = 0
        # Counter
        self.timer = QTimer()
        self.timer.setInterval(20)  # 20 = 1/50 second
        self.timer.timeout.connect(self.update_time)
        self.timer.start()

    # Exit by left mouse click
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.close()

    # Main procedural function
    def update_time(self):
        if self.clrs < refinement:
            self.setStyleSheet(''.join([background_text, colours[self.clrs]]))
            self.clrs += 1
        # This is to loop back to the start
        elif self.clrs <= 2 * refinement:
            self.setStyleSheet(''.join([background_text, colours[refinement - self.clrs]]))
            self.clrs += 1
        else:
            self.setStyleSheet(''.join([background_text, colours[0]]))
            self.clrs = 0


# Needed to run the app
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
