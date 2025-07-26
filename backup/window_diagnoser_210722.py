import sys
import numpy as np
import time
import math

# Establishing the colour palette
from pylsl import StreamInlet, resolve_stream
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget, QPushButton

from PyQt6.QtCore import Qt, QTimer
from scipy.fft import fft

# Establishing colours
colours = [0] * 256
for x in range(0, 256):
    colours[x] = "#{:02x}{:02x}{:02x}".format(10, 10, x)
background_text = "background-color: "

# This variables specifies how many colours will be used, +1
refinement = 255

# Stream variables, including time confusion
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
channel_1 = []
channel_2 = []
channel_3 = []
start_time = time.time()
time = []


# The display window

class Menu(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Another Window")
        layout.addWidget(self.label)
        self.setLayout(layout)

class Window(QMainWindow):

    def __init__(self):
        # Formatting
        super().__init__()
        self.setStyleSheet(''.join([background_text, colours[0]]))
        self.setWindowTitle("BrainMirror")
        self.showMaximized()
        #self.showFullScreen()
        self.show()
        # Counter
        self.clrs = 0
        # Timer
        self.timer = QTimer()
        self.timer.setInterval(20)  # msecs 100 = 1/10th sec
        self.timer.timeout.connect(self.update_time)
        self.timer.start()
        self.counter = 0
        #Figuring out menu
        self.button = QPushButton("Push for Window")
        self.button.clicked.connect(self.show_new_window)
        self.setCentralWidget(self.button)

    def show_new_window(self, checked):
        w = Menu()
        w.show()

    # Exit by left mouse click
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.close()

    # Main procedular function, does its job but is quite mushy at the moment
    def update_time(self):
        sample, timestamp = inlet.pull_sample()

        if len(time) <= 1:
            time.append(timestamp)
        else:
            time.append(timestamp - time[0])
        channel_1.append(sample[0])
        channel_2.append(sample[1])
        channel_3.append(sample[2])
        self.counter += 1
        if self.counter >= 2:
            if len(channel_1) < 5000:
                N = len(channel_1)
                freq_data = fft(channel_1)
                freq_data_2 = fft(channel_2)
                self.clrs = 30 + round(math.sqrt(abs(60 * sample[0])))
                # self.counter == 20
            else:
                N = 5000
                freq_data = fft(channel_1[-5000:])
                freq_data_2 = fft(channel_2[-5000:])
            y = 2 / N * np.abs(freq_data[0: int(N / 2)])
            y_2 = 2 / N * np.abs(freq_data_2[0: int(N / 2)])
            y_3 = channel_3
            np.savetxt("save_window_y.txt", y, delimiter=';')
            np.savetxt("save_window_y2.txt", y_2, delimiter=';')
            np.savetxt("save_window_y3.txt", y_3, delimiter=';')
            self.counter = 0

        if self.clrs < refinement:
            self.setStyleSheet(''.join([background_text, colours[self.clrs]]))
            # self.clrs += 1
        elif self.clrs <= 2 * refinement:
            self.setStyleSheet(''.join([background_text, colours[refinement - self.clrs]]))
            # self.clrs += 1
        else:
            self.setStyleSheet(''.join([background_text, colours[0]]))
            # self.clrs = 0

# Needed to run the app
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
