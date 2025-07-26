import sys
import time
import numpy as np
import math

from statistics import mean
from pylsl import StreamInlet, resolve_stream
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QGridLayout, \
  QWidget, QPushButton, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox
from PyQt6.QtCore import Qt, QTimer
from scipy.fft import fft
from functools import partial
import matplotlib.pyplot as plt

import SendStable

# Global Variables

# Stream variables, including several timestamps.
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
channel_1 = []
channel_2 = []
channel_3 = []
start_time = time.time()
time_now = []
background_text = "background-color: "


class Parameters:

    # Make a dictionary for all the defaults
    # Make a dictionary for all the values
    # Make a dictionary for all the options
    # Make a dictionary for all the text
    all_settings = {"feedback_colour_index": 0,
                    "regulation_index": 0}

    text_settings = {"feedback_colour_text": "Blue",
                    "regulation_text": "Upregulate"}

    all_options = {"feedback_colour_options": ["Blue", "Green", "Red"],
                     "regulation_options": ['Upregulate', 'Downregulate']}

    # Feedback design
    feedback_colour_index = 0
    feedback_colour_options = ["Blue", "Green", "Red"]
    feedback_colour_text = feedback_colour_options[feedback_colour_index]

    regulation_index = 0
    regulation_options = ['Upregulate', 'Downregulate']
    regulation_text = regulation_options[regulation_index]

    experimental_condition_index = 0
    experimental_condition_options = ['Placebo', 'Experimental', 'Blind']
    experimental_condition_text = regulation_options[regulation_index]

    feedback_duration = 1000
    feedback_duration_text = []

    smoothing_intensity = 5
    smoothing_intensity_text = []

    # Profile
    participant_id = 101
    participant_id_text = "101"

    session_id = 1
    session_id_text = "1"

    participant_left_peak = 5
    participant_left_peak_text = []

    participant_right_peak = 6
    participant_right_peak_text = []

    participant_left_frequency = 10.2
    participant_left_frequency_text = []

    participant_right_frequency = 10.2
    participant_right_frequency_text = []

    # Feedback maths
    threshold = 0.25
    threshold_text = []

    downsampling = 1.00
    downsampling_text = []

    filter_bellow = 5
    filter_bellow_text = []

    filter_above = 40
    filter_above_text = []

    filter_special_index = 0
    filter_special_options = ['Butter']
    filter_special_text = filter_special_options[filter_special_index]

    channel_measure_index = 0
    channel_measure_options = ['Average']
    channel_measure_text = channel_measure_options[channel_measure_index]

    baseline_duration = 5000
    baseline_duration_text = []

    # Advanced
    #show_latency = 'no'
    #downsampling = 'no'
    #special_algorhithm = 'none'
    #task_alongisde = 'no'

# Main menu
class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Another Window")
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.show()

        # menu options
        main_layout = QVBoxLayout()
        widget = QPushButton("Mirror")
        widget.clicked.connect(self.show_new_window)
        main_layout.addWidget(widget)

        widget = QPushButton("Baseline")
        widget.clicked.connect(self.acquire_baseline)
        main_layout.addWidget(widget)

        widget = QPushButton("Settings")
        widget.clicked.connect(self.show_settings)
        main_layout.addWidget(widget)

        widget = QPushButton("Exit")
        widget.clicked.connect(self.finish)
        main_layout.addWidget(widget)

        main_menu = QWidget()
        main_menu.setLayout(main_layout)
        self.setCentralWidget(main_menu)

    # Run feedback
    def show_new_window(self, checked):
        self.w = Feedback()
        self.w.show()

    # Will acquire, diagnose and store a participant profile to use for the neurofeedback
    def acquire_baseline(self, checked):
        self.w = Baseliner()
        self.w.show()

    # Run settings, currently needs to be filled with all the different options
    def show_settings(self, checked):
        self.w = Settings()
        self.w.show()

    # Simply exit
    def finish(self):
        self.close()


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
        if Parameters.all_settings["feedback_colour_index"] == 0:
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(10, 10, x)
        elif Parameters.all_settings["feedback_colour_index"] == 1:
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(10, x, 10)
        elif Parameters.all_settings["feedback_colour_index"] == 2:
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

    def make_settings_widget(self, name="Widget Name", order=0, column=0, number="yes", setting="feedback_colour"):
        self.label = QLabel(name)
        self.layout.addWidget(self.label, order, column)
        if number == "yes":
            widget = QDoubleSpinBox()
        elif number == "no":
            widget = QComboBox()
            widget.addItems(Parameters.all_options[setting+"_options"])
            widget.setCurrentIndex(Parameters.all_settings[setting+"_index"])
            widget.currentIndexChanged.connect(partial(self.index_changed, value=setting+"_index"))
            widget.editTextChanged.connect(partial(self.text_changed, value=setting+"_text"))
        self.layout.addWidget(widget, order+1, 0)
        self.setLayout(self.layout)

    def display_menu(self):
        """
        self.label = QLabel("Feedback Colour:")
        self.layout.addWidget(self.label, 0, 0)
        colour_widget = QComboBox()
        colour_widget.addItems(Parameters.feedback_colour_options)
        colour_widget.setCurrentIndex(Parameters.all_settings["feedback_colour_index"])
        colour_widget.currentIndexChanged.connect(partial(self.colour_index_changed, value="feedback_colour_index"))
        colour_widget.editTextChanged.connect(partial(self.colour_text_changed, value="feedback_colour_text"))
        self.layout.addWidget(colour_widget, 1, 0)
        self.setLayout(self.layout)
        """
        self.make_settings_widget("Feedback Colour", order=0, column=0, number="no", setting="feedback_colour")
        self.make_settings_widget("Regulation Direction", order=2, column=0, number="no", setting="regulation")

        """
        self.label = QLabel("Regulation Direction:")
        self.layout.addWidget(self.label, 2, 0)
        regulation_widget = QComboBox()
        regulation_widget.addItems(Parameters.regulation_options)
        regulation_widget.setCurrentIndex(Parameters.all_settings["regulation_index"])
        regulation_widget.currentIndexChanged.connect(partial(self.index_changed, value="regulation_index"))
        regulation_widget.editTextChanged.connect(partial(self.text_changed, value="regulation_text"))
        self.layout.addWidget(regulation_widget, 3, 0)
        self.setLayout(self.layout)
        """
        self.label = QLabel("Experimental Condition:")
        self.layout.addWidget(self.label, 4, 0)
        experimental_condition_widget = QComboBox()
        experimental_condition_widget.addItems(Parameters.experimental_condition_options)
        experimental_condition_widget.setCurrentIndex(Parameters.experimental_condition_index)
        experimental_condition_widget.currentIndexChanged.connect(self.experimental_condition_index_changed)
        experimental_condition_widget.editTextChanged.connect(self.experimental_condition_text_changed)
        self.layout.addWidget(experimental_condition_widget, 5, 0)
        self.setLayout(self.layout)

        self.label = QLabel("Feedback Duration (milliseconds):")
        self.layout.addWidget(self.label, 6, 0)
        feedback_duration_widget = QSpinBox()
        feedback_duration_widget.setMinimum(10)
        feedback_duration_widget.setMaximum(5000)
        #feedback_duration_widget.setPrefix("$")
        #feedback_duration_widget.setSuffix("c")
        feedback_duration_widget.setSingleStep(10)
        feedback_duration_widget.setValue(Parameters.feedback_duration)
        feedback_duration_widget.valueChanged.connect(self.duration_value_changed)
        feedback_duration_widget.textChanged.connect(self.duration_value_changed_str)
        self.layout.addWidget(feedback_duration_widget, 7, 0)
        self.setLayout(self.layout)

        self.label = QLabel("Smoothing Intensity:")
        self.layout.addWidget(self.label, 8, 0)
        smoothing_widget = QSpinBox()
        smoothing_widget.setMinimum(0)
        smoothing_widget.setMaximum(10)
        smoothing_widget.setSingleStep(1)
        smoothing_widget.setValue(Parameters.smoothing_intensity)
        smoothing_widget.valueChanged.connect(self.smoothing_value_changed)
        smoothing_widget.textChanged.connect(self.smoothing_value_changed_str)
        self.layout.addWidget(smoothing_widget, 9, 0)
        self.setLayout(self.layout)

        self.label = QLabel("Participant ID:")
        self.layout.addWidget(self.label, 10, 0)
        id_widget = QSpinBox()
        id_widget.setMinimum(0)
        id_widget.setMaximum(100000)
        id_widget.setSingleStep(1)
        id_widget.setValue(Parameters.participant_id)
   #     id_widget.valueChanged.connect(self.id_value_changed)
   #     id_widget.textChanged.connect(self.id_value_changed_str)
        self.layout.addWidget(id_widget, 11, 0)
        self.setLayout(self.layout)

        self.label = QLabel("Session ID:")
        self.layout.addWidget(self.label, 12, 0)
        session_widget = QSpinBox()
        session_widget.setMinimum(0)
        session_widget.setMaximum(100000)
        session_widget.setSingleStep(1)
        session_widget.setValue(Parameters.session_id)
        session_widget.valueChanged.connect(self.session_value_changed)
        session_widget.textChanged.connect(self.session_value_changed_str)
        self.layout.addWidget(session_widget, 13, 0)
        self.setLayout(self.layout)

        self.label = QLabel("Participant Left Peak:")
        self.layout.addWidget(self.label, 14, 0)
        lp_widget = QDoubleSpinBox()
        lp_widget.setMinimum(0)
        lp_widget.setMaximum(100)
        lp_widget.setSingleStep(0.01)
        lp_widget.setValue(Parameters.participant_left_peak)
        lp_widget.valueChanged.connect(self.lp_value_changed)
        lp_widget.textChanged.connect(self.lp_value_changed_str)
        self.layout.addWidget(lp_widget, 15, 0)
        self.setLayout(self.layout)

        self.label = QLabel("Participant Right Peak:")
        self.layout.addWidget(self.label, 16, 0)
        rp_widget = QDoubleSpinBox()
        rp_widget.setMinimum(0)
        rp_widget.setMaximum(100)
        rp_widget.setSingleStep(0.01)
        rp_widget.setValue(Parameters.participant_right_peak)
        rp_widget.valueChanged.connect(self.rp_value_changed)
        rp_widget.textChanged.connect(self.rp_value_changed_str)
        self.layout.addWidget(rp_widget, 17, 0)
        self.setLayout(self.layout)

        self.label = QLabel("Participant Left Frequency:")
        self.layout.addWidget(self.label, 0, 1)
        lf_widget = QDoubleSpinBox()
        lf_widget.setMinimum(0)
        lf_widget.setMaximum(100)
        lf_widget.setSingleStep(0.01)
        lf_widget.setValue(Parameters.participant_left_frequency)
        lf_widget.valueChanged.connect(self.lf_value_changed)
        lf_widget.textChanged.connect(self.lf_value_changed_str)
        self.layout.addWidget(lf_widget, 1, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Participant Right Frequency:")
        self.layout.addWidget(self.label, 2, 1)
        rf_widget = QDoubleSpinBox()
        rf_widget.setMinimum(0)
        rf_widget.setMaximum(100)
        rf_widget.setSingleStep(0.01)
        rf_widget.setValue(Parameters.participant_right_frequency)
        rf_widget.valueChanged.connect(self.rf_value_changed)
        rf_widget.textChanged.connect(self.rf_value_changed_str)
        self.layout.addWidget(rf_widget, 3, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Reward Threshold:")
        self.layout.addWidget(self.label, 4, 1)
        threshold_widget = QDoubleSpinBox()
        threshold_widget.setMinimum(0)
        threshold_widget.setMaximum(1)
        threshold_widget.setSingleStep(0.01)
        threshold_widget.setValue(Parameters.threshold)
        threshold_widget.valueChanged.connect(self.threshold_value_changed)
        threshold_widget.textChanged.connect(self.threshold_value_changed_str)
        self.layout.addWidget(threshold_widget, 5, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Lower Filter:")
        self.layout.addWidget(self.label, 6, 1)
        lfil_widget = QDoubleSpinBox()
        lfil_widget.setMinimum(0.5)
        lfil_widget.setMaximum(40)
        lfil_widget.setSingleStep(0.5)
        lfil_widget.setValue(Parameters.filter_bellow)
        lfil_widget.valueChanged.connect(self.lfil_value_changed)
        lfil_widget.textChanged.connect(self.lfil_value_changed_str)
        self.layout.addWidget(lfil_widget, 7, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Upper Filter:")
        self.layout.addWidget(self.label, 8, 1)
        ufil_widget = QDoubleSpinBox()
        ufil_widget.setMinimum(1)
        ufil_widget.setMaximum(100)
        ufil_widget.setSingleStep(0.5)
        ufil_widget.setValue(Parameters.filter_above)
        ufil_widget.valueChanged.connect(self.ufil_value_changed)
        ufil_widget.textChanged.connect(self.ufil_value_changed_str)
        self.layout.addWidget(ufil_widget, 9, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Filter Type:")
        self.layout.addWidget(self.label, 10, 1)
        regulation_widget = QComboBox()
        regulation_widget.addItems(Parameters.filter_special_options)
        regulation_widget.setCurrentIndex(Parameters.filter_special_index)
        regulation_widget.currentIndexChanged.connect(self.filter_special_index_changed)
        regulation_widget.editTextChanged.connect(self.filter_special_text_changed)
        self.layout.addWidget(regulation_widget, 11, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Channel Measure:")
        self.layout.addWidget(self.label, 12, 1)
        measure_widget = QComboBox()
        measure_widget.addItems(Parameters.channel_measure_options)
        measure_widget.setCurrentIndex(Parameters.channel_measure_index)
        measure_widget.currentIndexChanged.connect(self.measure_index_changed)
        measure_widget.editTextChanged.connect(self.measure_text_changed)
        self.layout.addWidget(measure_widget, 13, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Baseline Duration (milliseconds):")
        self.layout.addWidget(self.label, 14, 1)
        baseline_duration_widget = QSpinBox()
        baseline_duration_widget.setMinimum(10)
        baseline_duration_widget.setMaximum(5000)
        # baseline_duration_widget.setPrefix("$")
        # baseline_duration_widget.setSuffix("c")
        baseline_duration_widget.setSingleStep(10)
        baseline_duration_widget.setValue(Parameters.baseline_duration)
        baseline_duration_widget.valueChanged.connect(self.baseline_duration_value_changed)
        baseline_duration_widget.textChanged.connect(self.baseline_duration_value_changed_str)
        self.layout.addWidget(baseline_duration_widget, 15, 1)
        self.setLayout(self.layout)

        self.label = QLabel("Downsampling Rate:")
        self.layout.addWidget(self.label, 16, 1)
        downsampling_widget = QDoubleSpinBox()
        downsampling_widget.setMinimum(0)
        downsampling_widget.setMaximum(5)
        downsampling_widget.setSingleStep(0.01)
        downsampling_widget.setValue(Parameters.threshold)
        downsampling_widget.valueChanged.connect(self.downsampling_value_changed)
        downsampling_widget.textChanged.connect(self.downsampling_value_changed_str)
        self.layout.addWidget(downsampling_widget, 17, 1)
        self.setLayout(self.layout)


        button_exit = QPushButton("Save")
        button_exit.clicked.connect(self.finish)
        self.layout.addWidget(button_exit, 18, 0)

        button_load = QPushButton("Load")
        button_load.clicked.connect(self.load)
        self.layout.addWidget(button_load, 18, 1)

    def index_changed(self, i, value):  # i is an int
        Parameters.all_settings[value] = i


    def text_changed(self, s, value):  # s is a str
        Parameters.text_settings[value] = s

    def regulation_index_changed(self, i, value):  # i is an int
        Parameters.regulation_index = i

    def regulation_text_changed(self, s):  # s is a str
        Parameters.regulation_text = s

    def experimental_condition_index_changed(self, i):  # i is an int
        Parameters.experimental_condition_index = i

    def experimental_condition_text_changed(self, s):  # s is a str
        Parameters.experimental_condition_text = s

    def duration_value_changed(self, i):  # i is an int
        Parameters.feedback_duration = i

    def duration_value_changed_str(self, s):  # s is a str
        Parameters.feedback_duration_text = s

    def smoothing_value_changed(self, i):  # i is an int
        Parameters.smoothing_intensity = i

    def smoothing_value_changed_str(self, s):  # s is a str
        Parameters.smoothing_intensity_text = s

    def id_value_changed(self, i):  # i is an int
        Parameters.participant_id = i

   # def id_value_changed_str(self, s):  # s is a str
   #     Parameters.participant_id_text = s

    def session_value_changed(self, i):  # i is an int
        Parameters.session_id = i

    def session_value_changed_str(self, s):  # s is a str
        Parameters.session_id_text = s

    def lp_value_changed(self, i):  # i is an int
        Parameters.participant_left_peak = i

    def lp_value_changed_str(self, s):  # s is a str
        Parameters.participant_left_peak_text = s

    def rp_value_changed(self, i):  # i is an int
        Parameters.participant_right_peak = i

    def rp_value_changed_str(self, s):  # s is a str
        Parameters.participant_right_peak_text = s

    def lf_value_changed(self, i):  # i is an int
        Parameters.participant_left_frequency = i

    def lf_value_changed_str(self, s):  # s is a str
        Parameters.participant_left_frequency_text_text = s

    def rf_value_changed(self, i):  # i is an int
        Parameters.participant_right_frequency = i

    def rf_value_changed_str(self, s):  # s is a str
        Parameters.participant_right_frequency_text_text = s

    def threshold_value_changed(self, i):  # i is an int
        Parameters.threshold = i

    def threshold_value_changed_str(self, s):  # s is a str
        Parameters.threshold_text = s

    def downsampling_value_changed(self, i):  # i is an int
        Parameters.downsampling = i

    def downsampling_value_changed_str(self, s):  # s is a str
        Parameters.downsampling_text = s

    def lfil_value_changed(self, i):  # i is an int
        Parameters.filter_bellow = i

    def lfil_value_changed_str(self, s):  # s is a str
        Parameters.filter_bellow_text = s

    def ufil_value_changed(self, i):  # i is an int
        Parameters.filter_above = i

    def ufil_value_changed_str(self, s):  # s is a str
        Parameters.filter_above_text = s

    def filter_special_index_changed(self, i):  # i is an int
        Parameters.filter_special_index = i

    def filter_special_text_changed(self, s):  # s is a str
        Parameters.filter_special_text = s

    def measure_index_changed(self, i):  # i is an int
        Parameters.channel_measure_index = i

    def measure_text_changed(self, s):  # s is a str
        Parameters.channel_measure_text = s

    def baseline_duration_value_changed(self, i):  # i is an int
        Parameters.baseline_duration = i

    def baseline_duration_value_changed_str(self, s):  # s is a str
        Parameters.baseline_duration_text = s

    def load(self):
        self.channel_1_calculation = np.genfromtxt(Parameters.participant_id_text + "_" +
                                     Parameters.session_id_text + "_baseline.csv", delimiter=',')
        self.channel_1_calculation = np.delete(self.channel_1_calculation, 0, 0)
        self.freq_data_1 = fft(self.channel_1_calculation[:, 0])
        self.N_1 = len(self.channel_1_calculation)
        self.y_1 = 2 / self.N_1 * np.abs(self.freq_data_1[0: int((self.N_1 / 2))])
        self.freq_data_1 = np.linspace(0, 128, int(self.N_1/ 2))
        self.peak_1_frequency = self.freq_data_1[max(range(len(self.y_1)), key=self.y_1.__getitem__)]
        self.peak_1_amplitude = max(self.y_1)

        self.channel_2_calculation = np.genfromtxt(Parameters.participant_id_text + "_" +
                                                   Parameters.session_id_text + "_baseline.csv", delimiter=',')
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

        Parameters.participant_left_peak = self.peak_1_amplitude
        Parameters.participant_left_frequency = self.peak_1_frequency
        Parameters.participant_right_peak = self.peak_2_amplitude
        Parameters.participant_right_frequency = self.peak_2_frequency
        self.display_menu()

    def finish(self):
        self.close()


# Needed to run the app
App = QApplication(sys.argv)
window = Menu()
window.show()
sys.exit(App.exec())
