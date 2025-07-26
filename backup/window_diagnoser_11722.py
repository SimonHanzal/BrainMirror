# bugs - load makes text bolder, baseline duration will probably break it, settings categories

import sys
import time
import numpy as np
import math

from statistics import mean
from pylsl import StreamInlet, resolve_stream
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QGridLayout, \
  QWidget, QPushButton, QProgressBar, QComboBox, QDoubleSpinBox
from PyQt6.QtCore import Qt, QTimer
from scipy.fft import fft
from functools import partial
import matplotlib.pyplot as plt

import SendStable

# Stream variables, including several timestamps. This may need refactoring.
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
channel_1 = []
channel_2 = []
channel_3 = []
start_time = time.time()
time_now = []
background_text = "background-color: "


class Parameters:

    settings_settings = {
        "feedback_colour": ["Feedback Colour:", "feedback_colour",
                            0, 0, "no",  # row, column, number
                            0, 0, 0],  # min,max,step
        "regulation": ["Regulation Direction:", "regulation",
                       2, 0, "no",  # row, column, number
                       0, 0, 0],  # min,max,step
        "experimental_condition": ["Experimental Condition:", "experimental_condition",
                                   4, 0, "no",  # row, column, number
                                   0, 0, 0],  # min,max,step
        "feedback_duration": ["Feedback Duration (milliseconds):", "feedback_duration",
                              6, 0, "yes",  # row, column, number
                              10, 5000, 10],  # min,max,step
        "smoothing_intensity": ["Smoothing Intensity:", "smoothing_intensity",
                                8, 0, "yes",  # row, column, number
                                0, 10, 1],  # min,max,step
        "participant_id": ["Participant ID:", "participant_id",
                           10, 0, "yes",  # row, column, number
                           0, 100000, 1],  # min,max,step
        "session_id": ["Session ID:", "session_id",
                       12, 0, "yes",  # row, column, number
                       0, 100000, 1],  # min,max,step
        "participant_left_peak": ["Participant Left Peak:", "participant_left_peak",
                                  14, 0, "yes",  # row, column, number
                                  0, 100, 0.01],  # min,max,step
        "participant_right_peak": ["Participant Right Peak:", "participant_right_peak",
                                   16, 0, "yes",  # row, column, number
                                   0, 100, 0.01],  # min,max,step
        "participant_left_frequency": ["Participant Left Frequency:", "participant_left_frequency",
                                       0, 1, "yes",  # row, column, number
                                       0.50, 70, 0.01],  # min,max,step
        "participant_right_frequency": ["Participant Right Frequency:", "participant_right_frequency",
                                        2, 1, "yes",  # row, column, number
                                        0.50, 70, 0.01],  # min,max,step
        "threshold": ["Reward Threshold:", "threshold",
                      4, 1, "yes",  # position, number
                      0, 1, 0.01],  # min,max,step
        "lower_filter": ["Lower Filter:", "lower_filter",
                         6, 1, "yes",  # position, number
                         0.5, 40, 0.05],  # min,max,step
        "upper_filter": ["Lower Filter:", "lower_filter",
                         8, 1, "yes",  # position, number
                         1, 100, 0.05],  # min,max,step
        "filter_type": ["Filter Type:", "filter_type",
                        10, 1, "no",  # position, number
                        0, 0, 0],  # min,max,step
        "baseline_duration": ["Baseline Duration (milliseconds):", "baseline_duration",
                              12, 1, "yes",  # position, number
                              10, 5000, 10],  # min,max,step
        "channel_measure": ["Channel Measure:", "channel_measure",
                            14, 1, "no",  # position, number
                            0.5, 40, 0.05],  # min,max,step
        "downsampling": ["Downsampling Rate:", "downsampling",
                         16, 1, "yes",  # position, number
                         0, 5, 0.01],  # min,max,step
    }
    value_settings = {
        "feedback_duration": 1000,
        "smoothing_intensity": 5,
        "participant_id": 101,
        "session_id": 1,
        "participant_left_peak": 2,
        "participant_right_peak": 2,
        "participant_left_frequency": 10.0,
        "participant_right_frequency": 10.0,
        "threshold": 10.0,
        "lower_filter": 5,
        "upper_filter": 40,
        "baseline_duration": 5000,
        "downsampling": 1.00
    }
    index_settings = {
        "feedback_colour": 0,
        "regulation": 0,
        "experimental_condition": 0,
        "filter_type": 0,
        "channel_measure": 0
    }
    text_settings = {
        "feedback_colour": "Blue",
        "regulation": "Upregulate",
        "experimental_condition": "Placebo",
         "filter_type": "Butter",
         "channel_measure": "Average"
    }
    all_options = {
        "feedback_colour": ["Blue", "Green", "Red"],
        "regulation": ["Upregulate", "Downregulate"],
        "experimental_condition": ["Placebo", "Experimental", "Blind"],
        "filter_type": ["Butter"],
        "channel_measure": ["Average"]
    }
    # Advanced ideas
    # show_latency = 'no'
    # downsampling = 'no'
    # special_algorhithm = 'none'
    # task_alongisde = 'no'


# Main menu
class Menu(QMainWindow):
    # Initialisation of the main menu using main_menu_button.
    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        self.main_menu_button("Mirror", "self.show_new_window")
        self.main_menu_button("Baseline", "self.acquire_baseline")
        self.main_menu_button("Settings", "self.show_settings")
        self.main_menu_button("Exit", "self.finish")
        main_menu = QWidget()
        main_menu.setLayout(self.main_layout)
        self.setCentralWidget(main_menu)

    # Function in used to generate a menu button which executes a specified function upon clicking.
    def main_menu_button(self, label="Label", function="self.show_new_window"):
        widget = QPushButton(label)
        widget.clicked.connect(partial(eval(function)))
        self.main_layout.addWidget(widget)

    # Run feedback
    def show_new_window(self):
        self.user_window = Feedback()
        self.user_window.show()

    # Will acquire, diagnose and store a participant profile to use for the neurofeedback
    def acquire_baseline(self):
        self.user_window = Baseliner()
        self.user_window.show()

    # Runs settings as they are.
    def show_settings(self):
        self.user_window = Settings()
        self.user_window.show()

    # Simply exits the programme without saving.
    def finish(self):
        self.close()


# The main feedback interface which allows itself to be customised by settings.
class Feedback(QWidget):

    def __init__(self):
        # Formatting
        super().__init__()
        # Get Colours

        self.feedback_dataset_channel_1 = []
        self.feedback_dataset_channel_2 = []
        self.feedback_dataset_channel_3 = []
        self.feedback_dataset_timestamp = []

        self.refinement = 255  # how many colours will be used, +1
        self.colours = [0] * 256
        if Parameters.index_settings["feedback_colour"] == 0:
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(10, 10, x)
        elif Parameters.index_settings["feedback_colour"] == 1:
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(10, x, 10)
        elif Parameters.index_settings["feedback_colour"] == 2:
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(x, 10, 10)
        else:
            raise ValueError('Something broke with colours.')
        self.setStyleSheet(''.join([background_text, self.colours[0]]))
        self.setWindowTitle("BrainMirror")
        self.showMaximized()
        self.showFullScreen()
        # Counter
        self.clrs = 128
        # Timer
        self.timer = QTimer()
        self.timer.setInterval(1)  # msecs 100 = 1/10th sec
        self.timer.timeout.connect(self.update_time_2)
        self.timer.start()
        self.counter = 0

    # Exit by left mouse click
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.close()

    # Main procedural function, does its job but is quite mushy at the moment
    def update_info(self):

        sample, timestamp = inlet.pull_sample()

        if len(time_now) <= 1:
            time_now.append(timestamp)
        else:
            time_now.append(timestamp - time_now[0])

        self.feedback_dataset_channel_1 = np.append(self.feedback_dataset_channel_1, sample[0])
        self.feedback_dataset_channel_2 = np.append(self.feedback_dataset_channel_2, sample[1])
        self.feedback_dataset_channel_3 = np.append(self.feedback_dataset_channel_3, sample[2])
        self.feedback_dataset_timestamp = np.append(self.feedback_dataset_timestamp, timestamp)

        self.counter += 1

        if self.counter == 200:
            self.n = len(self.feedback_dataset_channel_1)
            self.freq_data_1 = fft(self.feedback_dataset_channel_1[-199:])
            self.y_1 = 2 / self.n * np.abs(self.freq_data_1[0: int(self.n / 2)])

            self.freq_data_2 = fft(self.feedback_dataset_channel_2[-199:])
            self.y_2 = 2 / self.n * np.abs(self.freq_data_2[0: int(self.n / 2)])
            self.linspace = np.linspace(0, 128, int(self.n / 2))

            self.current_amplitude_1 = mean(self.y_1)

            self.clrs = 30 + round(math.sqrt(abs(60 * self.current_amplitude_1)))

        if self.clrs < self.refinement:
            self.setStyleSheet(''.join([background_text, self.colours[self.clrs]]))
            # self.clrs += 1
        elif self.clrs <= 2 * self.refinement:
            self.setStyleSheet(''.join([background_text, self.colours[self.refinement - self.clrs]]))
            # self.clrs += 1
        else:
            self.setStyleSheet(''.join([background_text, self.colours[0]]))

    def update_time_2(self):
        sample, timestamp = inlet.pull_sample()
        if len(time_now) <= 1:
            time_now.append(timestamp)
        else:
            time_now.append(timestamp - time_now[0])
        # channel_1.append(sample[0])
        self.feedback_dataset_channel_1 = np.append(self.feedback_dataset_channel_1, sample[0])
        channel_2.append(sample[1])
        channel_3.append(sample[2])
        self.counter += 1
        """
        if self.counter >= 2:
            if len(channel_1) < 5000:
                n = len(channel_1)
                freq_data = fft(channel_1)
                freq_data_2 = fft(channel_2)
                self.clrs = 30 + round(math.sqrt(abs(60 * sample[0])))
                # self.counter == 20
            else:
                n = 5000
                freq_data = fft(channel_1[-5000:])
                freq_data_2 = fft(channel_2[-5000:])
                self.clrs = 30 + round(math.sqrt(abs(60 * sample[0])))
            y = 2 / n * np.abs(freq_data[0: int(n / 2)])
            y_2 = 2 / n * np.abs(freq_data_2[0: int(n / 2)])
            y_3 = channel_3
            self.counter = 0
        """
        if self.counter >= 50:
            channel_1_calculation = self.feedback_dataset_channel_1[-99:]
            freq_data_1 = fft(channel_1_calculation)
            n_1 = len(channel_1_calculation)
            y_1 = 2 / n_1 * np.abs(freq_data_1[0: int((n_1 / 2))])
            freq_data_1 = np.linspace(0, 128, int(n_1 / 2))
            peak_1_frequency = freq_data_1[max(range(len(y_1)), key=y_1.__getitem__)]
            peak_1_amplitude = max(y_1)

            """
            plt.figure(figsize=(8, 4))
            plt.ion()
            plt.show()
            plt.subplot(1, 2, 2)
            plt.plot(freq_data_1, y_1, label="channel_1", color="green")
            plt.xlim([0, 20])
            plt.title('Frequency domain Signal')
            plt.xlabel('Frequency in Hz')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.draw()
            """

            print(peak_1_amplitude)

            if peak_1_amplitude <= Parameters.participant_left_peak and self.clrs < 253 and self.clrs > 3:
                self.clrs -= 3
                self.setStyleSheet(''.join([background_text, self.colours[self.clrs]]))
            elif peak_1_amplitude > Parameters.participant_left_peak * 1.1 and self.clrs < 253 and self.clrs > 3:
                self.clrs += 3
                self.setStyleSheet(''.join([background_text, self.colours[self.clrs]]))

            self.counter = 0

        """
        if self.clrs < self.refinement:
            self.setStyleSheet(''.join([background_text, self.colours[self.clrs]]))
            # self.clrs += 1
        elif self.clrs <= 2 * self.refinement:
            self.setStyleSheet(''.join([background_text, self.colours[self.refinement - self.clrs]]))
            # self.clrs += 1
        else:
            self.setStyleSheet(''.join([background_text, self.colours[0]]))
            # self.clrs = 0
        """

    def update_time(self):
        sample, timestamp = inlet.pull_sample()

        if len(time_now) <= 1:
            time_now.append(timestamp)
        else:
            time_now.append(timestamp - time_now[0])
        channel_1.append(sample[0])
        channel_2.append(sample[1])
        channel_3.append(sample[2])
        self.counter += 1
        if self.counter >= 2:
            if len(channel_1) < 5000:
                n = len(channel_1)
                freq_data = fft(channel_1)
                freq_data_2 = fft(channel_2)
                self.clrs = 30 + round(math.sqrt(abs(60 * sample[0])))
                # self.counter == 20
            else:
                n = 5000
                freq_data = fft(channel_1[-5000:])
                freq_data_2 = fft(channel_2[-5000:])
            y = 2 / n * np.abs(freq_data[0: int(n / 2)])
            y_2 = 2 / n * np.abs(freq_data_2[0: int(n / 2)])
            y_3 = channel_3
            np.savetxt("save_window_y.txt", y, delimiter=';')
            np.savetxt("save_window_y2.txt", y_2, delimiter=';')
            np.savetxt("save_window_y3.txt", y_3, delimiter=';')
            self.counter = 0

        if self.clrs < self.refinement:
            self.setStyleSheet(''.join([background_text, self.colours[self.clrs]]))
            # self.clrs += 1
        elif self.clrs <= 2 * self.refinement:
            self.setStyleSheet(''.join([background_text, self.colours[self.refinement - self.clrs]]))
            # self.clrs += 1
        else:
            self.setStyleSheet(''.join([background_text, self.colours[0]]))
            # self.clrs = 0


class Baseliner(QWidget):

    def __init__(self):
        # Formatting
        self.limit = 1200
        self.baseline_dataset_channel_1 = []
        self.baseline_dataset_channel_2 = []
        self.baseline_dataset_channel_3 = []
        self.baseline_dataset_timestamp = []

        super().__init__()
        self.setWindowTitle("BrainMirror")
        self.showMaximized()
        self.showFullScreen()
        # Timer
        self.timer = QTimer()
        self.timer.setInterval(1)  # msecs 100 = 1/10th sec
        self.timer.timeout.connect(self.baseline_read)
        self.timer.start()
        self.counter = 0
        self.saver = 0
        self.progress = QProgressBar(self, maximum = Parameters.baseline_duration)
        self.progress.setGeometry(700, 500, 500, 20)
        self.progress.show()

    def baseline_read(self):
        # actual baseline
        sample, timestamp = inlet.pull_sample()
        if len(time_now) <= 1:
            time_now.append(timestamp)
        else:
            time_now.append(timestamp - time_now[0])

        self.baseline_dataset_channel_1 = np.append(self.baseline_dataset_channel_1, sample[0])
        self.baseline_dataset_channel_2 = np.append(self.baseline_dataset_channel_2, sample[1])
        self.baseline_dataset_channel_3 = np.append(self.baseline_dataset_channel_3, sample[2])
        self.baseline_dataset_timestamp = np.append(self.baseline_dataset_timestamp, timestamp)

        self.counter += 1
        self.saver += 1
        self.progress.setValue(self.counter)
        if self.saver == 100:
            np.savetxt(Parameters.participant_id_text + "_" + Parameters.session_id_text + "_baseline.csv",
                        np.transpose([self.baseline_dataset_channel_1,
                        self.baseline_dataset_channel_2,
                        self.baseline_dataset_channel_3,
                        self.baseline_dataset_timestamp,]),
                       header="channel_1,channel_2,time,timestamp",
                       comments="", delimiter=',')
            self.saver = 0
        #self.progress.setValue(round(self.counter / self.limit))
        if self.counter == Parameters.baseline_duration:
            np.savetxt(Parameters.participant_id_text + "_" + Parameters.session_id_text + "_baseline.csv",
                        np.transpose([self.baseline_dataset_channel_1,
                        self.baseline_dataset_channel_2,
                        self.baseline_dataset_channel_3,
                        self.baseline_dataset_timestamp,]),
                       header="channel_1,channel_2,time,timestamp",
                       comments="", delimiter=',')
            self.close()

    # Exit by left mouse click
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.close()


    # Sort of acquires something, a lot of work still to be done
    def update_time(self):
        sample, timestamp = inlet.pull_sample()

        if len(time) <= 1:
            time_now.append(timestamp)
        else:
            time_now.append(timestamp - time[0])
        channel_1.append(sample[0])
        channel_2.append(sample[1])
        channel_3.append(sample[2])
        self.counter += 1
        if self.counter == Parameters.baseline_duration:
            N = len(channel_1)
            freq_data = fft(channel_1)
            freq_data_2 = fft(channel_2)
            y = 2 / N * np.abs(freq_data[0: N])
            y_2 = 2 / N * np.abs(freq_data_2[0: N])
            y_3 = channel_3
            y_4 = [y, y_2, y_3]
            np.savetxt("save_window_y.txt", y, delimiter=';')
            np.savetxt("save_window_y2.txt", y_2, delimiter=';')
            np.savetxt("save_window_y3.txt", y_3, delimiter=';')
            np.savetxt("save_window_y4.txt", y_4, delimiter=";", fmt='%s')
            self.progress.hide()
            self.close()


class Settings(QWidget):

    def __init__(self):
        # Formatting
        super(Settings, self).__init__()
        self.layout = QGridLayout(self)
        self.display_menu()

    def make_settings_item(self, name="Widget Name", setting="feedback_colour",
                           order=0, column=0, number="yes",
                           mini=0, maxi=1, step=1):
        self.label = QLabel(name)
        self.layout.addWidget(self.label, order, column)
        if number == "yes":
            widget = QDoubleSpinBox()
            widget.setMinimum(mini)
            widget.setMaximum(maxi)
            widget.setSingleStep(step)
            widget.setValue(Parameters.value_settings[setting])
            widget.valueChanged.connect(partial(self.value_changed, value=setting))
            self.layout.addWidget(widget, order+1, column)
            self.setLayout(self.layout)
        elif number == "no":
            widget = QComboBox()
            widget.addItems(Parameters.all_options[setting])
            widget.setCurrentIndex(Parameters.index_settings[setting])
            widget.currentIndexChanged.connect(partial(self.index_changed, value=setting))
            widget.editTextChanged.connect(partial(self.text_changed, value=setting))
            self.layout.addWidget(widget, order+1, column)
            self.setLayout(self.layout)

    def display_menu(self):
        for menu_item in Parameters.settings_settings:
            self.make_settings_item(*Parameters.settings_settings[menu_item])

        button_exit = QPushButton("Save")
        button_exit.clicked.connect(self.finish)
        self.layout.addWidget(button_exit, 18, 0)

        button_load = QPushButton("Load")
        button_load.clicked.connect(self.load)
        self.layout.addWidget(button_load, 18, 1)

    def index_changed(self, i, value):  # i is an int
        Parameters.index_settings[value] = i

    def text_changed(self, s, value):  # s is a str
        Parameters.text_settings[value] = s

    def value_changed(self, i, value):  # i is an int
        Parameters.value_settings[value] = i

    def load(self):
        self.channel_1_calculation = np.genfromtxt(str(Parameters.value_settings["participant_id"]) + "_" +
                                     str(Parameters.value_settings["session_id"]) + "_baseline.csv", delimiter=',')
        self.channel_1_calculation = np.delete(self.channel_1_calculation, 0, 0)
        self.freq_data_1 = fft(self.channel_1_calculation[:, 0])
        self.N_1 = len(self.channel_1_calculation)
        self.y_1 = 2 / self.N_1 * np.abs(self.freq_data_1[0: int((self.N_1 / 2))])
        self.freq_data_1 = np.linspace(0, 128, int(self.N_1/ 2))
        self.peak_1_frequency = self.freq_data_1[max(range(len(self.y_1)), key=self.y_1.__getitem__)]
        self.peak_1_amplitude = max(self.y_1)

        self.channel_2_calculation = np.genfromtxt(str(Parameters.value_settings["participant_id"]) + "_" +
                                                   str(Parameters.value_settings["session_id"]) + "_baseline.csv", delimiter=',')
        self.channel_2_calculation = np.delete(self.channel_2_calculation, 0, 0)
        self.freq_data_2 = fft(self.channel_2_calculation[:, 1])
        self.N_2 = len(self.channel_2_calculation)
        self.y_2 = 2 / self.N_2 * np.abs(self.freq_data_2[0: int((self.N_2 / 2))])
        self.freq_data_2 = np.linspace(0, 128, int(self.N_2 / 2))
        self.peak_2_frequency = self.freq_data_2[max(range(len(self.y_2)), key=self.y_2.__getitem__)]
        self.peak_2_amplitude = max(self.y_2)

        """
        plt.figure(figsize=(8, 4))
        plt.ion()
        plt.show()
        plt.subplot(1, 2, 2)
        plt.plot(self.freq_data_1, self.y_1, label="channel_1", color="green")
        plt.subplot(1, 2, 1)
        plt.plot(self.freq_data_2, self.y_1, label="channel_2", color="yellow")
        plt.xlim([0, 20])
        plt.title('Frequency domain Signal')
        plt.xlabel('Frequency in Hz')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.draw()
        """
        Parameters.value_settings["participant_left_peak"] = self.peak_1_amplitude
        Parameters.value_settings["participant_right_peak"] = self.peak_2_amplitude
        Parameters.value_settings["participant_left_frequency"] = self.peak_1_frequency
        Parameters.value_settings["participant_right_frequency"] = self.peak_2_frequency
        self.display_menu()

    def finish(self):
        self.close()

# Needed to run the app
App = QApplication(sys.argv)
window = Menu()
window.show()
sys.exit(App.exec())
