# bugs - replay older recordings, do more colours, make same function, and customisable, for everywhere

import sys
import numpy as np

from os.path import exists
# From statistics import mean, this comes in when needed
from pylsl import StreamInlet, resolve_stream
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QGridLayout, \
    QWidget, QPushButton, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox
from PyQt6.QtCore import Qt, QTimer
from scipy.fft import fft
from functools import partial
import matplotlib.pyplot as plt

# Stream variables, including several timestamps. This may need refactoring.
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])


# These are the dictionary entries for settings menu items with the structure:
# Label, parameter name
# row, column, number being the menu item type
# minimum value, maximum value, step, and value is the position in value_settings storing the default value
class Parameters:
    settings_settings = {
        "participant_id": ["Participant ID:", "participant_id",
                           0, 0, "yes_single",  # row, column, number
                           0, 100000, 1, 0],  # min,max,step, value
        "session_id": ["Session ID:", "session_id",
                       2, 0, "yes_single",  # row, column, number
                       0, 100000, 1, 1],  # min,max,step, value
        "participant_left_peak": ["Participant Left Peak:", "participant_left_peak",
                                  4, 0, "yes",  # row, column, number
                                  0, 100, 0.01, 4],  # min,max,step, value
        "participant_right_peak": ["Participant Right Peak:", "participant_right_peak",
                                   6, 0, "yes",  # row, column, number
                                   0, 100, 0.01, 5],  # min,max,step, value
        "participant_left_frequency": ["Participant Left Frequency:", "participant_left_frequency",
                                       8, 0, "yes",  # row, column, number
                                       0.50, 70, 0.01, 6],  # min,max,step, value
        "participant_right_frequency": ["Participant Right Frequency:", "participant_right_frequency",
                                        10, 0, "yes",  # row, column, number
                                        0.50, 70, 0.01, 7],  # min,max,step, value
        "feedback_duration": ["Feedback Duration (s):", "feedback_duration",
                              0, 1, "yes",  # row, column, number
                              2, 5000, 1, 2],  # min,max,step, value

        "baseline_duration": ["Baseline Duration (s):", "baseline_duration",
                              2, 1, "yes",  # position, number
                              10, 100000, 1, 11],  # min,max,step
        "threshold": ["Reward Threshold:", "threshold",
                      4, 1, "yes",  # position, number
                      0, 1, 0.01, 8],  # min,max,step, value
        "feedback_colour": ["Feedback Colour:", "feedback_colour",
                            0, 2, "no",  # row, column, number
                            0, 0, 0],  # min,max,step, value
        "regulation": ["(empty) Regulation Direction:", "regulation",
                       2, 2, "no",  # row, column, number
                       0, 0, 0],  # min,max,step, value
        "experimental_condition": ["(empty) Experimental Condition:", "experimental_condition",
                                   4, 2, "no",  # row, column, number
                                   0, 0, 0],  # min,max,step, value
        "smoothing_intensity": ["(empty) Smoothing Intensity:", "smoothing_intensity",
                                6, 2, "yes",  # row, column, number
                                0, 10, 1, 3],  # min,max,step, value
        "lower_filter": ["(empty) Lower Filter:", "lower_filter",
                         8, 2, "yes",  # position, number
                         0.5, 40, 0.05, 9],  # min,max,step, value
        "upper_filter": ["(empty) Lower Filter:", "lower_filter",
                         10, 2, "yes",  # position, number
                         1, 100, 0.05, 10],  # min,max,step, value
        "filter_type": ["(empty) Filter Type:", "filter_type",
                        12, 2, "no",  # position, number
                        0, 0, 0],  # min,max,step, value
        "channel_measure": ["(empty) Channel Measure:", "channel_measure",
                            14, 2, "no",  # position, number
                            0.5, 40, 0.05],  # min,max,step, value
        "downsampling": ["(empty) Downsampling Rate:", "downsampling",
                         16, 2, "yes",  # position, number
                         0, 5, 0.01, 12],  # min,max,step, value
    }
    # This used to be a dictionary storing settings but has been deprecated and is kept only as an archive for now.
    """ value_settings = {
        "participant_id": 101,
        "session_id": 1,
        "feedback_duration": 10.0,
        "smoothing_intensity": 5.0,
        "participant_left_peak": 2,
        "participant_right_peak": 2,
        "participant_left_frequency": 10.0,
        "participant_right_frequency": 10.0,
        "threshold": 0.05,
        "lower_filter": 5,
        "upper_filter": 40,
        "baseline_duration": 20,
        "downsampling": 1.00
    } m"""

    # First checks if settings are saved and loads them if so.
    if exists("settings.npy"):
        value_settings = np.load("settings.npy")
    # Otherwise pre-generates new settings which are a simple numpy array with meaning to position as described bellow.
    # Whilst the logic seems cumbersome, it makes saving and loading very simple.
    else:
        value_settings = [
            101,  # participant id 0
            1,  # session id 1
            10.0,  # feedback duration 2
            5.0,  # smoothing intensity 3
            2,  # participant left peak 4
            2,  # participant right peak 5
            10.0,  # participant left frequency 6
            10.0,  # participant right frequency 7
            0.05,  # threshold 8
            5,  # lower filter 9
            40,  # upper filter 10
            20,  # baseline duration 11
            1.00  # downsampling 12
        ]
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
    # special_algorithm = 'none'
    # task_alongside = 'no'


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
        # Pre-initialised for PEP standard
        self.user_window = []

    # Function in used to generate a menu button which executes a specified function upon clicking.
    def main_menu_button(self, label="Label", function="self.show_new_window"):
        widget = QPushButton(label)
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(partial(eval(function)))
        self.main_layout.addWidget(widget)

    # Run feedback
    def show_new_window(self):
        self.user_window = Feedback()
        self.user_window.show()

    # Will acquire, diagnose and store a participant profile to use for the neurofeedback firstly prompting for an ID.
    def acquire_baseline(self):
        self.user_window = Usernamer()
        self.user_window.show()

    # Runs settings as they are.
    def show_settings(self):
        self.user_window = Settings()
        self.user_window.show()

    # Simply exits the programme without saving.
    def finish(self):
        self.close()


# The main feedback interface which allows customisation by settings.
class Feedback(QWidget):

    # noinspection PyTypeChecker
    def __init__(self):
        # Formatting
        super().__init__()
        # Central file for storing the incoming signal
        self.feedback_log = np.empty([0, 4])

        # The predefined colour scheme simply using x as a substitute of the modulated colours in range of 0 - 256.
        self.refinement = 255  # how many colours will be used, -1
        self.colours = [0] * 256
        if Parameters.index_settings["feedback_colour"] == 0:
            # Hinting about type
            x: int
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(0, 0, x)
        elif Parameters.index_settings["feedback_colour"] == 1:
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(0, x, 0)
        elif Parameters.index_settings["feedback_colour"] == 2:
            for x in range(0, 256):
                self.colours[x] = "#{:02x}{:02x}{:02x}".format(x, 0, 0)
        else:
            raise ValueError('Something broke with colours.')
        self.setStyleSheet(''.join(["background-color: ", self.colours[0]]))
        self.setWindowTitle("BrainMirror")
        self.showMaximized()
        self.showFullScreen()
        # Counter
        self.clrs = 128
        # Timer
        self.timer = QTimer()
        self.timer.setInterval(1)  # ms 100 = 1/10th sec
        # noinspection PyUnresolvedReferences
        self.timer.timeout.connect(self.update_window)
        self.timer.start()
        # Counter for recalculating the score
        self.counter = 0
        # Counter for changing screen colour
        self.mini_counter = 0
        # Counter for direction of screen colour change
        self.switch = 0
        # Counter for terminating
        self.macro_counter = 0

    # The user can exit at any time by left mouse click.
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # noinspection PyTypeChecker
            np.savetxt(str(Parameters.value_settings[0]) + "_" + str(
                Parameters.value_settings[1]) + "_mirror.csv",
                       self.feedback_log,
                       header="channel_1,channel_2,time,timestamp",
                       comments="", delimiter=',')
        self.close()

    def update_window(self):
        sample, timestamp = inlet.pull_sample()
        # channel_1.append(sample[0])
        self.feedback_log = np.append(self.feedback_log, [[sample[0], sample[1], sample[2], timestamp]], axis=0)
        # Update counters
        # The idea is for this to check the timings as well
        # print((self.feedback_log[-1, 3] - self.feedback_log[0, 3])-\
        # (self.feedback_log[-1, 2] - self.feedback_log[0, 2]))
        self.counter += 1
        self.mini_counter += 1
        self.macro_counter += 1
        # Currently the counter is at one second but that has absolutely no significance
        if self.counter >= 1000:
            channel_1_calculation = self.feedback_log[-1000:, 0]
            freq_data_1 = fft(channel_1_calculation)
            n_1 = len(channel_1_calculation)
            y_1 = 2 / n_1 * np.abs(freq_data_1[0: int((n_1 / 2))])
            freq_data_1 = np.linspace(0, 500, int(n_1 / 2))
            # Currently not needed to calculate the peak
            # peak_1_frequency = freq_data_1[max(range(len(y_1)), key=y_1.__getitem__)]
            # Get only a value 2 Hz within what is the pre-set peak
            peak_1_desired_frequency = Parameters.value_settings[6]
            y_lower_limit = freq_data_1 > peak_1_desired_frequency - 2
            y_lower_limit_position = np.where(y_lower_limit)[0][0]
            y_upper_limit = freq_data_1 < peak_1_desired_frequency + 2
            y_upper_limit_position = np.where(y_upper_limit)[0][-1]

            # This needs to be improved to account for extreme values likely to happen outside the range
            peak_1_amplitude = max(y_1[y_lower_limit_position:y_upper_limit_position])

            channel_2_calculation = self.feedback_log[-1000:, 1]
            freq_data_2 = fft(channel_2_calculation)
            n_2 = len(channel_2_calculation)
            y_2 = 2 / n_2 * np.abs(freq_data_2[0: int((n_2 / 2))])
            freq_data_2 = np.linspace(0, 500, int(n_2 / 2))
            # Currently not needed to calculate the peak
            # peak_2_frequency = freq_data_2[max(range(len(y_2)), key=y_2.__getitem__)]
            # Get only a value 2 Hz within what is the pre-set peak
            peak_2_desired_frequency = Parameters.value_settings[7]
            y_lower_limit = freq_data_2 > peak_2_desired_frequency - 2
            y_lower_limit_position = np.where(y_lower_limit)[0][0]
            y_upper_limit = freq_data_2 < peak_2_desired_frequency + 2
            y_upper_limit_position = np.where(y_upper_limit)[0][-1]

            # This needs to be improved to account for extreme values likely to happen outside the range
            peak_2_amplitude = max(y_2[y_lower_limit_position:y_upper_limit_position])
            # This is just to check what is happening but can be removed later
            # print(peak_2_amplitude)
            # print(peak_2_frequency)

            # Thresholds are at 5% from the middle value, currently adding up the signal, which should be changeable
            # print(peak_1_amplitude + peak_2_amplitude)
            if peak_1_amplitude + peak_2_amplitude <= (Parameters.value_settings[4] + Parameters.value_settings[5]) * \
                    (1 - Parameters.value_settings[8]):
                self.switch = -1
            elif peak_1_amplitude + peak_2_amplitude > (Parameters.value_settings[4] + Parameters.value_settings[5]) * \
                    (1 + Parameters.value_settings[8]):
                self.switch = 1
            else:
                self.switch = 0
            self.counter = 0

        if self.mini_counter >= 50:
            self.mini_counter = 0
            # Dynamically change the colour, do this more often than the calculations
            if 0 < self.clrs < 254:
                # print(self.clrs)
                # print(self.switch)
                self.clrs += self.switch
                self.setStyleSheet(''.join(["background-color: ", self.colours[self.clrs]]))
        if self.macro_counter > (1000 * Parameters.value_settings[2]):
            self.close()
        # The file needs to be likewise saved for later.
        # Also, there is currently no support for time. There needs to be another macroscopic timer for overall duration
        # and to get into any discrepancies between the timestamp and channel 3.


# This class is a simple selection of participant id and session id to assign the acquired baseline to.
# noinspection PyMethodMayBeStatic
class Usernamer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainMirror")
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Select user ID")
        self.layout.addWidget(self.label)
        widget = QSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(100000)
        widget.setSingleStep(1)
        widget.setValue(int(Parameters.value_settings[0]))
        # noinspection PyUnresolvedReferences
        widget.valueChanged.connect(self.user_change)
        self.layout.addWidget(widget)
        self.setLayout(self.layout)
        self.user_window_2 = []

        self.label = QLabel("Select session ID")
        self.layout.addWidget(self.label)
        widget = QSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(100000)
        widget.setSingleStep(1)
        widget.setValue(int(Parameters.value_settings[1]))
        # noinspection PyUnresolvedReferences
        widget.valueChanged.connect(self.session_change)
        self.layout.addWidget(widget)
        self.setLayout(self.layout)

        widget = QPushButton("Select")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.get_baseline)
        self.layout.addWidget(widget)

    def user_change(self, i):  # i is an int
        Parameters.value_settings[0] = i

    def session_change(self, i):  # i is an int
        Parameters.value_settings[1] = i

    def get_baseline(self):
        self.user_window_2 = Baseliner()
        self.user_window_2.show()
        self.close()


# This class takes the baseline of the signal in quite similar manner to infer the individual values for
# The participant. It could do with prompting the user to enter the participant id and session id.
# noinspection PyUnresolvedReferences,PyArgumentList,PyTypeChecker
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
        self.timer.setInterval(1)  # 100ms = 1/10th sec
        self.timer.timeout.connect(self.baseline_read)
        self.timer.start()
        # Used to finish
        self.macro_time = 0
        # Used to recalculate
        self.counter = 0
        # Used to save
        self.saver = 0
        self.progress = QProgressBar(self, maximum=int(Parameters.value_settings[11]) * 1000)
        self.progress.setGeometry(700, 500, 500, 20)
        self.progress.show()

    def baseline_read(self):
        # Acquires the baseline, but currently is a bit messy and could be simplified.
        sample, timestamp = inlet.pull_sample()

        self.baseline_dataset_channel_1 = np.append(self.baseline_dataset_channel_1, sample[0])
        self.baseline_dataset_channel_2 = np.append(self.baseline_dataset_channel_2, sample[1])
        self.baseline_dataset_channel_3 = np.append(self.baseline_dataset_channel_3, sample[2])
        self.baseline_dataset_timestamp = np.append(self.baseline_dataset_timestamp, timestamp)

        self.counter += 1
        self.saver += 1
        self.progress.setValue(self.counter)
        if self.saver == 100:
            np.savetxt(str(Parameters.value_settings[0]) + "_" + str(Parameters.value_settings[1]) + "_baseline.csv",
                       np.transpose([self.baseline_dataset_channel_1,
                                     self.baseline_dataset_channel_2,
                                     self.baseline_dataset_channel_3,
                                     self.baseline_dataset_timestamp]),
                       header="channel_1,channel_2,time,timestamp",
                       comments="", delimiter=',')
            self.saver = 0
        # self.progress.setValue(round(self.counter / self.limit))
        if self.counter == int(Parameters.value_settings[11] * 1000):
            np.savetxt(str(Parameters.value_settings[0]) + "_" + str(Parameters.value_settings[1]) + "_baseline.csv",
                       np.transpose([self.baseline_dataset_channel_1,
                                     self.baseline_dataset_channel_2,
                                     self.baseline_dataset_channel_3,
                                     self.baseline_dataset_timestamp]),
                       header="channel_1,channel_2,time,timestamp",
                       comments="", delimiter=',')
            self.close()

    # Exit by left mouse click
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.close()


# This is all the user predefined settings which save upon every modification.
# noinspection PyUnresolvedReferences,PyMethodMayBeStatic
class Settings(QWidget):
    # Initialisation is very simple
    def __init__(self):
        # Formatting
        super(Settings, self).__init__()
        self.layout = QGridLayout(self)
        self.display_menu()

    # This function can create a QDoubleSpinBox = "yes", QSpinBox = "yes_single"
    # or QCombo widget, it takes the arguments
    # Of tag displayed in menu, internal setting name, position where it is, whether it is a number
    # And so the minimum and maximum values as well as the step between them
    # Note, this can make it tricky with extreme peak values coming from the actual device.
    def make_settings_item(self, name="Widget Name", setting="feedback_colour",
                           order=0, column=0, number="yes",
                           mini=0, maxi=1, step=1, value=0):
        # noinspection PyAttributeOutsideInit
        self.label = QLabel(name)
        self.layout.addWidget(self.label, order, column)
        if number == "yes":
            widget = QDoubleSpinBox()
            widget.setMinimum(mini)
            widget.setMaximum(maxi)
            widget.setSingleStep(step)
            widget.setValue(Parameters.value_settings[value])
            widget.valueChanged.connect(partial(self.value_changed, value=value))
            self.layout.addWidget(widget, order + 1, column)
            self.setLayout(self.layout)
        elif number == "no":
            widget = QComboBox()
            widget.addItems(Parameters.all_options[setting])
            widget.setCurrentIndex(Parameters.index_settings[setting])
            widget.currentIndexChanged.connect(partial(self.index_changed, value=setting))
            widget.editTextChanged.connect(partial(self.text_changed, value=setting))
            self.layout.addWidget(widget, order + 1, column)
            self.setLayout(self.layout)
        elif number == "yes_single":
            widget = QSpinBox()
            widget.setMinimum(mini)
            widget.setMaximum(maxi)
            widget.setSingleStep(step)
            widget.setValue(int(Parameters.value_settings[value]))
            widget.valueChanged.connect(partial(self.value_changed, value=value))
            self.layout.addWidget(widget, order + 1, column)
            self.setLayout(self.layout)

    # Master call for the function to iterate through the menu and do it all.
    def display_menu(self):
        # This complicated code makes sure the widgets are refreshed
        for i in reversed(range(self.layout.count())):
            # noinspection PyTypeChecker
            self.layout.itemAt(i).widget().setParent(None)
        # Generate menu items anew
        for menu_item in Parameters.settings_settings:
            self.make_settings_item(*Parameters.settings_settings[menu_item])

        # The function also displays the save, check and load buttons.
        button_load = QPushButton("Load")
        button_load.clicked.connect(partial(self.load, plot=0))
        self.layout.addWidget(button_load, 18, 0)

        button_check = QPushButton("Check")
        button_check.clicked.connect(partial(self.load, plot=1))
        self.layout.addWidget(button_check, 18, 1)

        button_exit = QPushButton("Save")
        button_exit.clicked.connect(self.finish)
        self.layout.addWidget(button_exit, 18, 2)

    # These three functions are necessary to enable correct handling of setting change signals.
    def index_changed(self, i, value):  # i is an int
        Parameters.index_settings[value] = i

    def text_changed(self, s, value):  # s is a str
        Parameters.text_settings[value] = s

    def value_changed(self, i, value):  # i is an int
        Parameters.value_settings[value] = i

    # This function calculates the averages from a specified participant and session.
    # This function can be simplified to be similar to the mirror, however it does not need to
    # Calculate anything continuously, so in that respect it is different.
    # noinspection PyAttributeOutsideInit
    def process_channel(self, channel=0):
        channel_calculation = np.delete(self.all_channels, 0, 0)
        freq_data = fft(channel_calculation[:, channel])
        number = len(channel_calculation)
        y = 2 / number * np.abs(freq_data[0: int((number / 2))])
        freq_data = np.linspace(0, 500, int(number / 2))
        return [freq_data[max(range(len(y)), key=y.__getitem__)], max(y)]
        # Careful here as this is not cropped like in the feedback.

    def load(self, plot=1):
        # noinspection PyBroadException
        try:
            # This does the calculation, ultimately replace by one function shared with parameters.
            self.all_channels = np.genfromtxt(str(int(Parameters.value_settings[0])) + "_" +
                                                       str(int(Parameters.value_settings[1])) + "_baseline.csv",
                                                       delimiter=',')
            self.peak_1_frequency = self.process_channel(0)[0]
            self.peak_1_amplitude = self.process_channel(0)[1]
            self.peak_2_frequency = self.process_channel(1)[0]
            self.peak_2_amplitude = self.process_channel(1)[1]

            # This is deprecated and replaced by functions up there
            """
            self.channel_2_calculation = np.genfromtxt(str(int(Parameters.value_settings[0])) + "_" +
                                                       str(int(Parameters.value_settings[1])) + "_baseline.csv",
                                                       delimiter=',')
            self.channel_2_calculation = np.delete(self.channel_2_calculation, 0, 0)
            self.freq_data_2 = fft(self.channel_2_calculation[:, 1])
            self.N_2 = len(self.channel_2_calculation)
            self.y_2 = 2 / self.N_2 * np.abs(self.freq_data_2[0: int((self.N_2 / 2))])
            self.freq_data_2 = np.linspace(0, 500, int(self.N_2 / 2))
            self.peak_2_frequency = self.freq_data_2[max(range(len(self.y_2)), key=self.y_2.__getitem__)]
            self.peak_2_amplitude = max(self.y_2)
            """

            # An optional extra activated by parameter plot is here which plots the first 20Hz of each channel.
            if plot == 1:
                plt.figure(figsize=(8, 4))
                plt.ion()
                plt.show()
                plt.subplot(1, 2, 2)
                plt.plot(self.freq_data_2, self.y, label="channel_1", color="blue")
                plt.xlim([0, 20])
                plt.subplot(1, 2, 1)
                plt.plot(self.freq_data_2, self.y, label="channel_2", color="purple")
                plt.xlim([0, 20])
                plt.title('Frequency domain Signal')
                plt.xlabel('Frequency in Hz')
                plt.ylabel('Amplitude')
                plt.tight_layout()
                plt.draw()

            # This is key, as it displays the participant update.
            Parameters.value_settings[4] = self.peak_1_amplitude
            Parameters.value_settings[5] = self.peak_2_amplitude
            Parameters.value_settings[6] = self.peak_1_frequency
            Parameters.value_settings[7] = self.peak_2_frequency
            # However, there is a mistake here as this writes all the labels again and makes them strange.
            self.display_menu()
        except Exception:
            self.user_window = WarningWindow()
            self.user_window.show()

    # The save button currently only closes the programme.
    def finish(self):
        # This saves the settings but as they are to an array, then it would work.
        np.save("settings.npy", Parameters.value_settings, allow_pickle=True)
        self.close()


# This is a simple pop-up window warning that a participant with that id and session number does not exist.
class WarningWindow(QWidget):

    # Initialisation is very simple
    def __init__(self):
        # Formatting
        super(WarningWindow, self).__init__()
        self.setWindowTitle("BrainMirror")
        self.layout = QGridLayout(self)
        label = QLabel("Participant or Session does not exist!")
        self.layout.addWidget(label)
        widget = QPushButton("OK")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.finish)
        self.layout.addWidget(widget)

    def finish(self):
        self.close()


# These commands are needed to run the app.
App = QApplication(sys.argv)
window = Menu()
window.show()
sys.exit(App.exec())
