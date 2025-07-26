# REMOVE THIS (3 places) to switch around. Make secrets.csv in data to delineate conditions
# Matplotlib import treads backends carefully
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mne
import numpy as np
import random
import os
import sys
import threading

from cryptography.fernet import Fernet
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectrum
from functools import partial
from os.path import exists
from pylsl import StreamInlet, resolve_stream
from PyQt6.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QGridLayout, \
    QWidget, QPushButton, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
from scipy.fft import fft
from scipy.signal import detrend
from time import sleep, time

# Stream variables, including timestamps.
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0], max_buflen=1000,  max_chunklen=64)

# Parameters
# One of the potential issues slowing down the neurofeedback display.
DISPLAY_PRECISION = 100
# The sampling rate, it NEEDS to match that of the server.
SRATE = 512
# How often to recalculate feedback (times display_precision)
FEEDBACK_FREQUENCY = 10
# How quick should the fade be between the colours, based on FEEDBACK_FREQUENCY.
SCREEN_SPEED = 1
# Inspection frequency, how much of the raw signal gets shown in 1/SRATE milliseconds
EVALUATION = 200
# Maximum number of samples to take before shutting down:
RECORDING_LENGTH = 180000
# Total number of participants
PARTICIPANT_TOP = 11
# Around IAF
AROUND = 2
# Minimum alpha
LOWER_ALPHA = 7
# Maximum alpha
HIGHER_ALPHA = 14
# Savitzky window
WINDOW = SRATE*2
# Ceiling effect constant
CEILING_EFFECT = 1.75

# Global scope
# Raw signal
tracked_signal, tracked_signal_2, timestamp = np.empty(0), np.empty(0), np.empty(0)
all_signal = np.empty([0,3])
# Processed signal
clean_signal_1 = np.empty([0, 1])
clean_signal_2 = np.empty([0, 2])
all_clean_signal = np.empty([0,2])
# Left conditions?
some_conditions_left = True
# Baseline correction
base_1, base_2 = 0, 0


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    window_size = np.abs(int(window_size)) + 20
    order = np.abs(int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window size must be an odd number")
    if window_size < order + 2:
        raise TypeError("window size is too small for the polynomials order")
    polynomial_order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # coefficients
    b = np.mat([[k ** i for i in polynomial_order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad at extremes
    first_values = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    last_values = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((first_values, y, last_values))
    return np.convolve(m[::-1], y, mode='valid')


# I have to begin with this. This is a worker, who does real-time computations on top of the GUI.
# This worker hums in the background and keeps acquiring data from the server. All the other functions and menu
# Are involved in processing it and meaningfully displaying it.
class Worker(QObject):
    progress = Signal(int)
    completed = Signal(int)
    raw_signal = []
    raw_signal_2 = []
    @Slot(int)
    def do_work(self, n):
        global tracked_signal
        global tracked_signal_2
        global timestamp
        global all_signal
        # Counter for special events
        self.odd_occurred = False
        # Starts a lagging-behind tracker
        self.thread4 = threading.Thread(target=self.rebase)
        self.thread4.start()
        for i in range(1, n+1):
            # Loop at a good speed
            sleep(0.01)
            # Pulling chunks from server if there are any
            sample, timestamp_temp = inlet.pull_chunk()
            sample = np.array(sample).squeeze().flatten()
            if len(sample) != 0:
                timestamp_temp = np.zeros(len(sample))
            # This only works if something was received
            if sample.ndim > 0:
                # Test what happens if only a single value comes through
                if len(sample) == 1:
                    print("Message: Sample of length one, attempting resolve...")
                # A repair mechanism in case odd number of values comes through
                if (len(sample) % 2) != 0:
                    print("WARNING: Mismatched chunk, attempting to resolve...")
                    # First appends channels normally
                    if self.odd_occurred:
                        self.odd_occurred = False
                        tracked_signal = np.concatenate((tracked_signal, sample[::2]), axis=0)
                        tracked_signal_2 = np.concatenate((tracked_signal_2, sample[1::2]), axis=0)
                        timestamp = np.concatenate((timestamp, timestamp_temp[::2]), axis=0)
                    # And then switched around
                    else:
                        self.odd_occurred = True
                        tracked_signal = np.concatenate((tracked_signal, sample[1::2]), axis=0)
                        tracked_signal_2 = np.concatenate((tracked_signal_2, sample[::2]), axis=0)
                        timestamp = np.concatenate((timestamp, timestamp_temp[1::2]), axis=0)
                # This is the normal scenario
                elif len(sample) % 2 == 0 and not self.odd_occurred:
                    tracked_signal = np.concatenate((tracked_signal, sample[::2]), axis=0)
                    tracked_signal_2 = np.concatenate((tracked_signal_2, sample[1::2]), axis=0)
                    timestamp = np.concatenate((timestamp, timestamp_temp[::2]), axis=0)
                    if len(tracked_signal) == len(tracked_signal_2):
                        all_signal = np.column_stack((tracked_signal, tracked_signal_2, timestamp))
                    elif len(tracked_signal) > len(tracked_signal_2):
                        all_signal = np.column_stack((tracked_signal[len(tracked_signal_2)], tracked_signal_2, timestamp))
                    else:
                        print("Message: Mismatch unresolved.")
                elif len(sample) % 2 == 0 and self.odd_occurred:
                    print("WARNING: Only one odd chunk, attempting to resolve...")
                    tracked_signal = np.concatenate((tracked_signal, sample[::2]), axis=0)
                    tracked_signal_2 = np.concatenate((tracked_signal_2, 0, sample[1::2]), axis=0)
                    timestamp = np.concatenate((timestamp, timestamp_temp[::2]), axis=0)
                    if len(tracked_signal) == len(tracked_signal_2):
                        all_signal = np.column_stack((tracked_signal, tracked_signal_2, timestamp))
                    elif len(tracked_signal) > len(tracked_signal_2):
                        all_signal = np.column_stack(
                            (tracked_signal[len(tracked_signal_2)], tracked_signal_2, timestamp))
                    else:
                        print("Message: Mismatch unresolved.")
                elif len(sample[::2]) is not len(sample[1::2]):
                    print("WARNING: Mismatching samples")
                    print(sample)
                else:
                    print("ERROR: Unknown error")
            self.progress.emit(i)
        self.completed.emit(i)

    # This automatically de-baselines everything. Very strong!
    def rebase(self):
        global base_1
        global base_2
        global tracked_signal
        global tracked_signal_2
        global base_archive
        base_archive = np.empty([0,2])
        while True:
            sleep(5)
            base_1 = np.mean(tracked_signal[-512*5:])
            base_2 = np.mean(tracked_signal_2[-512*5:])

# And this is second worker looping at a slower pace over the raw signal one and acquiring a derivative
# clean signal, which should be based on the menu parameters.
class OnlineProcessor(QObject):
    # PyQT specific signals
    processing_progress = Signal(int)
    processing_completed = Signal(int)

    @Slot(int)
    def do_processing(self, n):
        # Counter detecting if a mismatch occurs
        self.mismatch_exists = False
        global tracked_signal
        global tracked_signal_2
        # Being nice and initialising the variables beforehand
        global clean_signal_1
        global clean_signal_2
        global all_clean_signal
        global all_signal
        global timestamp
        beginning = True
        current_index = len(all_signal)
        pre_processed_signal_1 = []
        pre_processed_signal_2 = []
        for i in range(1, n + 1):
            # Loop 10x slower than main worker
            # sleep(0.125)
            sleep(0.01)
            # This piece of code ensures that if the app longs for too long, it gets restarted or closes.
            if len(tracked_signal) > 10200000:
                os._exit(1)
            if len(tracked_signal) > 10100000:
                print("WARNING, DEVICE WILL SHUT OFF!")
            elif len(tracked_signal) > 10000000:
                print("WARNING, DEVICE NEEDS RESTART!")
            if beginning:
                older_index = len(all_signal)-100
                beginning = False
            else:
                older_index = current_index
            current_index = len(all_signal)
            processed_signal_1 = all_signal[older_index:current_index,0] - base_1
            processed_signal_2 = all_signal[older_index:current_index,1] - base_2
            increment = len(processed_signal_1)
            if increment > 1 and len(processed_signal_2) > 1:
                # Here comes the whole online processing pipeline, which can be further optimised.
                # Filter
                if increment != len(processed_signal_2):
                    print("MESSAGE: channel sample length mismatch of:")
                    print(increment - len(processed_signal_2))
                    self.mismatch_exists = True
                elif self.mismatch_exists and increment == len(processed_signal_2):
                    print("Problem resolved!")
                    self.mismatch_exists = False
                # Process incrementally
                pre_processed_signal_1, clean_signal_1 = self.apply_filter_and_smooth(
                    pre_processed_signal_1, processed_signal_1, clean_signal_1, WINDOW, increment)
                pre_processed_signal_2, clean_signal_2 = self.apply_filter_and_smooth(
                    pre_processed_signal_2, processed_signal_2, clean_signal_2, WINDOW, increment)
                all_clean_signal = np.column_stack((clean_signal_1, clean_signal_2))
            if len(clean_signal_1) != len(clean_signal_2):
                print("Malfunction imminent!")
            self.processing_progress.emit(i)
        self.processing_completed.emit(i)

    def apply_filter_and_smooth(self, pre_processed_signal, processed_signal, clean_signal, WINDOW, increment):
        if len(pre_processed_signal) < WINDOW:
            pre_processed_signal = mne.filter.filter_data(
                np.append(pre_processed_signal, processed_signal),
                sfreq=512,
                l_freq=Parameters.value_settings[9],
                h_freq=Parameters.value_settings[10],
                method="iir", verbose="ERROR")
        else:
            pre_processed_signal = mne.filter.filter_data(
                np.append(pre_processed_signal[-(WINDOW - increment):], processed_signal),
                sfreq=512,
                l_freq=Parameters.value_settings[9],
                h_freq=Parameters.value_settings[10],
                method="iir", verbose="ERROR")

        pre_processed_signal = pre_processed_signal.squeeze()
        processed_signal = abs(pre_processed_signal[-increment:])

        if len(clean_signal) < WINDOW:
            clean_signal = savitzky_golay(np.append(clean_signal, processed_signal), Parameters.value_settings[3], 5)
        else:
            clean_signal = np.append(
                clean_signal[0:-WINDOW],
                savitzky_golay(np.append(clean_signal[-WINDOW:], processed_signal), Parameters.value_settings[3], 5))
        return pre_processed_signal, clean_signal

# These are the dictionary entries for the settings menu with the structure:
# Label, parameter name
# row, column, number being the menu item type
# minimum value, maximum value, step, and value is the position in value_settings storing the default value
class Parameters:
    settings_settings = {
        "participant_id": ["Participant ID:", "participant_id",
                           0, 0, "yes_single",  # row, column, number
                           0, 100000000, 1, 0],  # min,max,step, value
        "session_id": ["Session ID:", "session_id",
                       2, 0, "yes_single",  # row, column, number
                       0, 100000000, 1, 1],  # min,max,step, value
        "participant_left_peak": ["Participant Left Peak:", "participant_left_peak",
                                  4, 0, "yes",  # row, column, number
                                  -100000, 100000, 0.00001, 4],  # min,max,step, value
        "participant_right_peak": ["Participant Right Peak:", "participant_right_peak",
                                   6, 0, "yes",  # row, column, number
                                   -100000, 100000, 0.00001, 5],  # min,max,step, value
        "participant_left_frequency": ["Participant Left Frequency:", "participant_left_frequency",
                                       8, 0, "yes",  # row, column, number
                                       0.50, 70, 0.01, 6],  # min,max,step, value
        "participant_right_frequency": ["Participant Right Frequency:", "participant_right_frequency",
                                        10, 0, "yes",  # row, column, number
                                        0.50, 70, 0.01, 7],  # min,max,step, value
        "feedback_duration": ["Feedback Duration:", "feedback_duration",
                              0, 1, "yes",  # row, column, number
                              10, 5000, 10, 2],  # min,max,step, value
        "baseline_duration": ["Baseline Duration:", "baseline_duration",
                              2, 1, "yes",  # position, number
                              10, 100000, 10, 11],  # min,max,step, value
        "threshold": ["Reward Threshold:", "threshold",
                      4, 1, "yes",  # position, number
                      0, 1, 0.01, 8],  # min,max,step, value
        "lower_iir": ["Low Frequency Cut-off:", "low_iir",
                      6, 1, "yes",  # position, number
                      0.5, 12, 0.5, 13],  # min,max,step, value
        "higher_iir": ["High Frequency Cut-off:", "high_iir",
                      8, 1, "yes",  # position, number
                      15, 60, 0.5, 14],  # min,max,step, value
        "feedback_colour": ["Feedback Colour:", "feedback_colour",
                            0, 2, "no",  # row, column, number
                            0, 0, 0],  # min,max,step, value
        "regulation": ["Regulation Direction:", "regulation",
                       2, 2, "no",  # row, column, number
                       0, 0, 0],  # min,max,step, value
        "experimental_condition": ["Experimental Condition:", "experimental_condition",
                                   4, 2, "no",  # row, column, number
                                   0, 0, 0],  # min,max,step, value
        "smoothing_intensity": ["Smoothing Intensity:", "smoothing_intensity",
                                6, 2, "yes",  # row, column, number
                                11, 1000, 2, 3],  # min,max,step, value
        "lower_filter": ["Alpha definition lower (Hz):", "lower_filter",
                         8, 2, "yes",  # position, number
                         0.5, 40, 0.5, 9],  # min,max,step, value
        "upper_filter": ["Alpha definition upper (Hz):", "lower_filter",
                         10, 2, "yes",  # position, number
                         1, 100, 0.5, 10],  # min,max,step, value
        "filter_type": ["Filter Type:", "filter_type",
                        12, 2, "no",  # position, number 
                        0, 0, 0],  # min,max,step, value
        "channel_measure": ["Channel Measure:", "channel_measure",
                            14, 2, "no",  # position, number
                            0.5, 40, 0.05],  # min,max,step, value
        "downsampling": ["(empy) Downsampling Rate:", "downsampling",
                         16, 2, "yes",  # position, number
                         0, 128, 1, 12]  # min,max,step, value
    }
    # First checks if settings are saved and loads them if so, otherwise resets them to defaults.
    if exists("settings.npy"):
        value_settings = np.load("settings.npy", allow_pickle=True)

    # Load colourmap
    if exists("data//colour_maps.npy"):
        colormap = (np.load("data//colour_maps.npy", allow_pickle=True))
    # Otherwise, pre-generates new settings which are a simple numpy array with meaning to position as described bellow.
    # Whilst the logic seems cumbersome, it makes saving and loading very simple.
    else:
        value_settings = [
            101,  # participant id 0
            1,  # session id 1
            10.0,  # feedback duration 2
            211.0,  # smoothing intensity 3
            2,  # participant left peak 4
            2,  # participant right peak 5
            10.0,  # participant left frequency 6
            10.0,  # participant right frequency 7
            0.25,  # threshold 8
            8,  # lower filter 9
            12,  # upper filter 10
            20,  # baseline duration 11
            16.00,  # downsampling 12
            8,  # low iir 13
            30  # high iir 14
        ]
    # Multiple option settings menus, will be manually reset, so preferences should be here.
    # In time, the indexes can be migrated to value_settings.
    index_settings = {
        "feedback_colour": 3,
        "regulation": 1,  # regulation 1
        "experimental_condition": 2,  # experimental condition 2
        "filter_type": 0,
        "channel_measure": 0,
        "pretend_stability": 0,
        "session_extract": 0,
        "rebaseline": 1
    }
    text_settings = {
        "feedback_colour": "Balanced",
        "regulation": "Downregulate",
        "experimental_condition": "One",
        "filter_type": "Butter",
        "channel_measure": "Sum"
    }
    all_options = {
        "feedback_colour": ["Blue", "Green", "Red", "Balanced"],
        "regulation": ["Upregulate", "Downregulate"],
        "experimental_condition": ["One", "Two", "Double-blind"], # Blind?
        "filter_type": ["Butter"],
        "channel_measure": ["Average"]
    }


# Main menu
class Menu(QMainWindow):
    work_requested = Signal(int)
    processing_requested = Signal(int)
    # Initialisation of the main menu using the main_menu_button.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_layout = QVBoxLayout()
        self.main_menu_button("Mirror", "self.show_new_window")
        self.main_menu_button("Baseline", "self.acquire_baseline")
        self.main_menu_button("Raw", "self.raw")
        self.main_menu_button("Processed", "self.alpha")
        self.main_menu_button("Encrypt", "self.encrypt_warning")
        self.main_menu_button("Settings", "self.show_settings")
        self.main_menu_button("Exit", "self.finish")
        main_menu = QWidget()
        main_menu.setLayout(self.main_layout)
        self.setCentralWidget(main_menu)

        # Start the slotting here
        self.user_window = []
        # Threading
        self.worker = Worker()
        self.worker_thread = QThread()

        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.complete)

        self.work_requested.connect(self.worker.do_work)

        # move worker to the worker thread
        self.worker.moveToThread(self.worker_thread)

        # start the thread
        self.worker_thread.start()
        self.start()

        # This is the processor
        # Threading
        self.online_processor = OnlineProcessor()
        self.online_processor_thread = QThread()

        self.online_processor.processing_progress.connect(self.processing_update_progress)
        self.online_processor.processing_completed.connect(self.processing_complete)

        self.processing_requested.connect(self.online_processor.do_processing)

        # move worker to the worker thread
        self.online_processor.moveToThread(self.online_processor_thread)

        # Processor temporarily disabled not to upset the code
        # start the thread
        self.online_processor_thread.start()
        self.processing_start()

    # This function is used to generate a menu button which executes a specified function upon clicking.
    def main_menu_button(self, label="Label", function="self.show_new_window"):
        widget = QPushButton(label)
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(partial(eval(function)))
        self.main_layout.addWidget(widget)

    # Runs feedback mirror, first prompting for the ID.
    def show_new_window(self):
        self.user_window = UsernamerMirror()
        self.user_window.show()

    # Will acquire, diagnose and store a participant profile to use for the neurofeedback, firstly prompting for an ID.
    def acquire_baseline(self):
        self.user_window = Usernamer()
        self.user_window.show()

    # Used to display the raw signal
    def raw(self):
        self.user_window = Raw()
        self.user_window.show()

    # Used to display the preprocessed signal
    def alpha(self):
        self.user_window = Alpha()
        self.user_window.show()

    # Runs settings as they are.
    def show_settings(self):
        self.user_window = Settings()
        self.user_window.show()

    # Shows changelog.
    def encrypt_warning(self):
        self.user_window = EncryptionWarning()
        self.user_window.show()

    # Simply exits the programme without saving.
    def finish(self):
        global inlet
        inlet.close_stream()
        self.close()

    def start(self):
        n = RECORDING_LENGTH
        self.work_requested.emit(n)

    def update_progress(self, v):
        pass

    def complete(self, v):
        print("Warning: Recording limit reached! Please, restart brain mirror. ")

    def processing_update_progress(self, v):
        pass

    def processing_complete(self, v):
        print("Parallel process finished. ")

    def processing_start(self):
        n = RECORDING_LENGTH
        self.processing_requested.emit(n)


# This is a window that issues a warning before participant conditions get encrypted
class EncryptionWarning(QWidget):
    # Initialisation is very simple.
    def __init__(self):
        # Formatting
        super(EncryptionWarning, self).__init__()
        self.setWindowTitle("BrainMirror")
        self.layout = QGridLayout(self)
        label = QLabel("Warning! This will irreversibly (for data collectors) encrypt the experimental design!")
        self.layout.addWidget(label)
        widget = QPushButton("Continue")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.go_ahead)
        self.layout.addWidget(widget)
        widget = QPushButton("Cancel")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.finish)
        self.layout.addWidget(widget)

    def go_ahead(self):
        reading = np.genfromtxt('data//secrets.csv', delimiter=',', dtype=int)
        np.save("conditions//encrypted_secrets.npy", reading, allow_pickle=True)
        if os.path.isfile('data//secrets.csv'):
            os.remove('data//secrets.csv')
        self.close()

    def finish(self):
        self.close()


# This will just show a plot of the raw signal to check that it is coming in at all and what it is like.
# It shows the signal pre-filtered at a specified rate.
# This is by far the buggiest part of the programme.
class Raw(QWidget):

    # noinspection PyTypeChecker
    def __init__(self):
        # Formatting
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Start to inspect signal, only exit by pressing finish.")
        self.layout.addWidget(self.label)
        self.setWindowTitle("BrainMirror")
        widget = QPushButton("Start")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.display)
        self.layout.addWidget(widget)
        widget = QPushButton("Finish")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.exit)
        self.layout.addWidget(widget)
        # Display window in top left corner.
        self.topLeftPoint = self.screen().availableGeometry().topLeft()
        self.move(self.topLeftPoint)
        # This is a counter for acquiring samples, resets upon display.
        self.counter = 0
        # Only show signal starting at the moment of button press
        self.starting_point = len(tracked_signal)
        # Closing mechanism
        self.view_complete = False

    def exit(self):
        self.view_complete = True
        self.close()
        plt.close()

    def animate(self, i, xs: list, ys1: list, ys2: list):
        global base_1
        global base_2
        # By default, the animation progresses slightly ahead of the sampling rate not to fall behind
        # The first frame must be different to start off at the point when the viewer was initialised
        if self.beginning:
            change = 80
            self.addition1 = tracked_signal[self.starting_point - change:self.starting_point]
            self.addition2 = tracked_signal_2[self.starting_point - change:self.starting_point]
            self.previous_signal = self.starting_point
        else:
            # Default behaviour
            if (self.counter - len(tracked_signal) + self.starting_point) < -100:
                change = 80
                self.counter += change
                self.addition1 = tracked_signal[self.starting_point - change + self.counter:self.starting_point + self.counter]
                self.addition2 = tracked_signal_2[self.starting_point - change + self.counter:self.starting_point + self.counter]
            else:
                # Readjust the change score to account for any delays
                change = len(tracked_signal) - self.previous_signal
                self.counter += change
                self.addition1 = tracked_signal[self.starting_point + self.counter - change:self.starting_point + self.counter]
                self.addition2 = tracked_signal_2[self.starting_point + self.counter - change:self.starting_point + self.counter]
        if change == 0:
            pass
        # Readjusts the plot window only when needed to maximise frame rate
        elif max(self.addition1) - base_1 > self.axlim:
            self.axlim = max(self.addition1) - base_1
        elif max(self.addition1) - base_1 < -self.axlim:
            self.axlim = abs(max(self.addition1) - base_1)
        if change == 0:
            pass
        elif max(self.addition2) - base_2 > self.axlim:
            self.axlim = max(self.addition2) - base_2
        elif max(self.addition2) - base_2 < -self.axlim:
            self.axlim = abs(max(self.addition2) - base_2)
        self.stretch_back += 1
        if self.stretch_back > 250:
            tracked_max = np.max(np.append(abs(tracked_signal[-2000:]-base_1), abs(tracked_signal_2[-2000:]-base_2)))
            self.axlim = tracked_max+tracked_max*0.1
            self.stretch_back = 0
        # Add x and y to lists
        self.xs.extend(list(range(self.counter-(change-1), self.counter+1)))
        self.ys1.extend(self.addition1 - base_1)
        self.ys2.extend(self.addition2 - base_2)
        # Limit x and y lists to sampling rate items
        xs = self.xs[-SRATE:]
        ys1 = self.ys1[-SRATE:]
        ys2 = self.ys2[-SRATE:]
        # Draw x and y lists
        self.ax.clear()
        self.ax.plot(xs, ys1)
        self.ax.plot(xs, ys2)
        # Format plot
        self.ax.set_ylim([-self.axlim, self.axlim])#-self.axlim, self.axlim])
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.20)
        self.ax.set_title('Plot of server signal')
        self.ax.set_xlabel('Sample no')
        self.ax.set_ylabel('mV')
        # Mark that the first frame run
        self.beginning = False
        # Log the length of the signal at this point
        self.previous_signal = len(tracked_signal)
        # Plot exiting behaviour
        if self.view_complete:
            self.ani.event_source.stop()

    # Displaying function
    def display(self):
        # Carefully knitted high-performance animation
        global tracked_signal
        self.beginning = True
        self.counter = 0
        self.axlim = 0.00001
        self.fig, self.ax = plt.subplots()
        self.xs = []
        self.ys1 = []
        self.ys2 = []
        self.stretch_back = 0
        # Starts off the animation
        self.ani = animation.FuncAnimation(self.fig, self.animate, fargs=(self.xs, self.ys1, self.ys2), interval=30)
        plt.show()


# This will just show a plot of the raw signal to check that it is coming in at all and what it is like.
# It shows the signal pre-filtered at the rate of quarter a second, with 1 second rate.
# This is by far the buggiest part of the programme. Not sure what I am doing wrong.
class Alpha(QWidget):

    # noinspection PyTypeChecker
    def __init__(self):
        # Formatting
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.label = QLabel("This inspects a transformed version of the signal, press finish to stop!")
        self.layout.addWidget(self.label)
        self.setWindowTitle("BrainMirror")
        widget = QPushButton("Start")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.display)
        self.layout.addWidget(widget)
        widget = QPushButton("Finish")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.exit)
        self.layout.addWidget(widget)
        # Display window in top left corner.
        self.topLeftPoint = self.screen().availableGeometry().topLeft()
        self.move(self.topLeftPoint)
        # This is a counter for acquiring samples, resets upon display.
        self.counter = 0
        self.starting_point = len(clean_signal_1)
        self.view_complete = False

    def exit(self):
        self.view_complete = True
        self.close()
        plt.close()

    def animate(self, i, xs: list, ys1: list, ys2: list):
        global current_time
        global previous_time
        # Different first window
        if self.beginning:
            current_time = 0
            change = 80
            self.addition1 = clean_signal_1[self.starting_point - change:self.starting_point]
            self.addition2 = clean_signal_2[self.starting_point - change:self.starting_point]
            self.previous_signal = self.starting_point
        else:
            # Normal behaviour

            if (len(clean_signal_1) - self.starting_point - self.counter) < 100:
                change = 80
                self.counter += change
                self.addition1 = clean_signal_1[self.starting_point - change + self.counter:self.starting_point + self.counter]
                self.addition2 = clean_signal_2[self.starting_point - change + self.counter:self.starting_point + self.counter]
            else:
                # Behaviour if lagging behind
                change = len(clean_signal_1) - self.previous_signal
                self.counter += change
                self.addition1 = clean_signal_1[self.starting_point + self.counter - change:self.starting_point + self.counter]
                self.addition2 = clean_signal_2[self.starting_point + self.counter - change:self.starting_point + self.counter]
        if change == 0:
            pass
        # Adjust window size
        elif max(self.addition1) > self.axlim:
            self.axlim = max(self.addition1)
        elif max(self.addition1) < -self.axlim:
            self.axlim = abs(max(self.addition1))
        if change == 0:
            pass
        elif max(self.addition2) > self.axlim:
            self.axlim = max(self.addition2)
        elif max(self.addition2) < -self.axlim:
            self.axlim = abs(max(self.addition2))
        self.stretch_back += 1
        if self.stretch_back > 250:
            tracked_max = np.max(np.append(abs(clean_signal_1[-2000:]), abs(clean_signal_2[-2000:])))
            self.axlim = tracked_max+tracked_max*0.1
            self.stretch_back = 0
        # Some conditionals can be added here if the data has too high a value
        # Add x and y to lists
        self.xs.extend(list(range(self.counter-(change-1), self.counter+1)))
        self.ys1.extend(self.addition1)
        self.ys2.extend(self.addition2)
        # Limit x and y lists to 1536 items
        xs, ys1, ys2 = self.xs[-1536:], self.ys1[-1536:], self.ys2[-1536:]
        # Draw x and y lists
        previous_time = time()
        self.ax.clear()
        self.ax.plot(xs, ys1)
        self.ax.plot(xs, ys2)
        # current_time += (time() - previous_time)
        # print(current_time)
        self.beginning = False
        self.previous_signal = len(clean_signal_1)
        # Format plot
        self.ax.set_ylim([-self.axlim/10, self.axlim])
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.20)
        self.ax.set_title('Plot of processed signal')
        self.ax.set_xlabel('Sample no')
        self.ax.set_ylabel('power')
        # Closing behaviour
        if self.view_complete:
            self.ani.event_source.stop()
        previous_time = time()

    def do_check(self):
        global clean_signal_1
        while True:
            sleep(2)
            #print("Delay (s): " + str((len(clean_signal_1)-len(tracked_signal))/512))
            if self.view_complete:
                sys.exit()

    def display(self):
        global clean_signal_1
        self.beginning = True
        self.counter = 0
        self.axlim = 0.00001
        self.fig, self.ax = plt.subplots()
        self.xs = []
        self.ys1 = []
        self.ys2 = []
        self.stretch_back = 0
        # Threading a logger of being behind
        self.thread2 = threading.Thread(target=self.do_check)
        self.thread2.start()
        self.ani = animation.FuncAnimation(self.fig, self.animate, fargs=(self.xs, self.ys1, self.ys2), interval=100)#30
        plt.show()


# The main feedback interface which allows customisation by settings.
class Feedback(QWidget):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        # global file for storing the incoming signal
        global all_clean_signal
        global some_conditions_left
        # Initialise the local widget storage of feedback
        self.online_feedback = np.empty([0, 3])
        # The predefined colour scheme with x for modulated colours in range of 0 - 256.
        self.refinement = 255  # how many colours will be used, -1
        self.colours = [0] * 256
        # Can be modified to do more advanced combinations.
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
        elif Parameters.index_settings["feedback_colour"] == 3:
            self.colours = Parameters.colormap
        else:
            raise ValueError('Something broke with colours.')
        # Counter, starts in the middle of screen colour
        self.clrs = 128
        # Determining the condition
        # If setting is One or Two, picks the colour according to the setting
        self.load_encryption()
        if Parameters.index_settings["experimental_condition"] < 2:
            self.condition = Parameters.index_settings["experimental_condition"]
            self.assigned = True
        # Otherwise, the setting should be two
        elif Parameters.index_settings["experimental_condition"] >= 2:
            self.assign = True
            path = "conditions//" + str(int(Parameters.value_settings[0])) + "_secret.npy"
            if os.path.isfile(path):
                self.condition = np.load(path, allow_pickle=True)
                print(self.condition)
            # First tries to load the participant setting
            # if os.path.isfile(path):
            #     self.file = open(path, 'rb')
            #     self.encrypted_condition = self.file.read()
            #     self.condition = self.decryptify(self.encrypted_condition)
            #     # self.condition = np.genfromtxt(path)
                print("Loaded successfully!")
            # If that does not work, check if there are these secrets files
            elif os.path.isfile("conditions//encrypted_secrets.npy"):
                conditions_left = np.load("conditions//encrypted_secrets.npy", allow_pickle=True)
                if len(conditions_left) == 0:
                    self.assign = False
                else:
                    # Finally, if passed the tests, assign a new condition
                    # Get a number in the settings range
                    self.condition = int(np.round(random.choice(range(2, PARTICIPANT_TOP + 3))))
                    # Only try to get it right 50 times
                    iterator = 0
                    # If the number is already taken try again
                    while not np.any(conditions_left == self.condition):
                        iterator += 1
                        print("Trying again!")
                        self.condition = int(np.round(random.choice(range(2, PARTICIPANT_TOP + 3))))
                        # But don't go crazy
                        if iterator >= PARTICIPANT_TOP * 5:
                            print("Run out of secrets to assign!")
                            self.close()
                    # Save the participant-specific secret in a file, not to be open until finished
                    self.save_value = np.empty([0])
                    self.save_value = np.append(self.save_value, np.array([self.condition]))
                    # self.save_value_encrypt = self.encryptify(str(self.save_value))
                    # Open the file as wb to write bytes
                    # file = open("conditions//" + str(int(Parameters.value_settings[0])) + "_secret.csv", 'wb')
                    # file.write(self.save_value_encrypt)  # The key is type bytes still
                    # file.close()
                    np.save("conditions//" + str(int(Parameters.value_settings[0])) + "_secret.npy", self.save_value, allow_pickle=True)
                    conditions_left = conditions_left[conditions_left != self.save_value]
                    # REMOVE THIS at the end, to make fully double-blind for students
                    #np.savetxt("data//secrets.csv",
                    #           conditions_left,
                    #           comments="", delimiter=',')
                    np.save("conditions//encrypted_secrets.npy", conditions_left, allow_pickle=True)
            # If no encrypted experimental files are present, close down the thing
            else:
                print("Unable to read secrets")
                some_conditions_left = False
                self.close()
            # Next, check if there are secrets left to allocate
            if not self.assign:
                print("Run out of all the secrets to assign!")
                self.assigned = False
                some_conditions_left = False
                self.close()
            elif self.assign:
                # derive from the loaded number which condition the participant is in
                if (self.condition % 2) == 0:
                    self.condition = 0
                elif (self.condition % 2) == 1:
                    self.condition = 1
                self.assigned = True
        if self.assigned:
            # Show window
            self.assigned = False
            self.setStyleSheet(''.join(["background-color: ", self.colours[0]]))
            self.setWindowTitle("BrainMirror")
            self.showMaximized()
            self.showFullScreen()
            # Timer
            self.timer = QTimer()
            # 1ms = 1/10th sec
            self.timer.setInterval(1000)
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
            # noinspection PyUnresolvedReferences
            self.timer.timeout.connect(self.update_window)
            self.starting_time = time()
            self.timer.start()
            # Counter for direction of screen colour change
            self.switch = 0
            # Ending counter
            self.ending_counter = 0
            # Starting time
            self.old_time = len(all_clean_signal)
            self.start_time = self.old_time
            self.finishing = False
            self.increment = 1
            self.thread3 = threading.Thread(target=self.colour_changer)
            self.thread3.start()

    # Needs to be loaded before for encryption to work
    def load_encryption(self):
        file = open('key.key', 'rb')
        key = file.read()
        file.close()
        self.fernet = Fernet(key)

    # Wraps around and encrypts
    def encryptify(self, value):
        var_test = str(value)
        encryption = self.fernet.encrypt(var_test.encode())
        return encryption

    # Wraps around and decrypts
    def decryptify(self, encryption):
        decryption = self.fernet.decrypt(encryption)
        decryption = int.from_bytes(decryption, 'big')
        return decryption

    # The user can exit at any time by left mouse click.
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.finishing = True

    # Gets basic time-frequency data.
    def recalculate_channel(self, channel=0):
        # This is just to get values.
        if len(all_clean_signal[self.old_time:self.new_time,:]) > 0:
            threshold_value = np.mean(all_clean_signal[self.old_time:self.new_time, channel])
        else:
            print("Error!")
        return threshold_value

    def colour_changer(self):
        # Dynamically change the colour, do this more often than the calculations.
        # If the colour is settable:
        while not self.finishing:
            sleep(0.1)
            if 0 < self.clrs < 255 - self.increment:
                # Based on switch, start changing the colour
                if self.switch == -2:
                    # Make sure colours don't drop under zero
                    if self.clrs > 1 + self.increment:
                        self.clrs -= 1 + self.increment
                elif self.switch == -1:
                    # Here, the options are either to increase or decrease it.
                    if self.clrs > 63:
                        self.clrs -= 1 + self.increment
                    if self.clrs <= 63:
                        self.clrs += 1 + self.increment
                elif self.switch == 0:
                    if self.clrs > 127:
                        self.clrs -= 1 + self.increment
                    if self.clrs <= 127:
                        self.clrs += 1 + self.increment
                elif self.switch == 1:
                    if self.clrs > 191:
                        self.clrs -= 1 + self.increment
                    if self.clrs <= 191:
                        self.clrs += 1 + self.increment
                elif self.switch == 2:
                    if self.clrs < 254 - self.increment:
                        self.clrs += 1
                else:
                    print("Unknown issue with screen colour change.")
        # And write it into the style sheet.
            self.setStyleSheet(''.join(["background-color: ", self.colours[self.clrs]]))

    def update_window(self):
        global all_clean_signal
        global CEILING_EFFECT
        self.ending_counter += 1
        # Generates a number between 0 - 1 to be used in the placebo condition.
        test_number = random.random()
        # And this pulls out the actual peaks from the channels.
        self.new_time = len(all_clean_signal)
        peak_1_amplitude = self.recalculate_channel(channel=0)
        peak_2_amplitude = self.recalculate_channel(channel=1)
        #print(peak_1_amplitude + peak_2_amplitude)
        self.old_time = len(all_clean_signal)
        # Currently, the thresholding principle distinguishes specified increment and has 5 steps
        # in the experimental conditions, calculating the amplitudes. This needs to be changed to smooth it.
        if self.condition == 0:
            # When upregulating
            if Parameters.index_settings["regulation"] == 0:
                if (peak_1_amplitude + peak_2_amplitude) * (1 + 2*Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 + Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -1
                elif (peak_1_amplitude + peak_2_amplitude) * (1 - 2*Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 - Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 1
                else:
                    self.switch = 0
            elif Parameters.index_settings["regulation"] == 1:
                if (peak_1_amplitude + peak_2_amplitude) * (1 - 2*Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 - Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -1
                elif (peak_1_amplitude + peak_2_amplitude) * (1 + 2*Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 + Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 1
                else:
                    self.switch = 0
            else:
                print("Something wrong with the direction of regulation!")
        # The placebo condition decides this based on a 5-sided dice-roll.
        elif self.condition == 1:
            """
            if test_number < 0.2:
                self.switch = -2
            elif test_number < 0.4:
                self.switch = -1
            elif test_number < 0.6:
                self.switch = 0
            elif test_number < 0.8:
                self.switch = 1
            else:
                self.switch = 2
            """
            # When downregulating
            if Parameters.index_settings["regulation"] == 0:
                if (peak_1_amplitude + peak_2_amplitude) * (1 - 2 * Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 - Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -1
                elif (peak_1_amplitude + peak_2_amplitude) * (1 + 2 * Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 + Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 1
                else:
                    self.switch = 0
            elif Parameters.index_settings["regulation"] == 1:
                if (peak_1_amplitude + peak_2_amplitude) * (1 + 2 * Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 + Parameters.value_settings[8]) <= \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = -1
                elif (peak_1_amplitude + peak_2_amplitude) * (1 - 2 * Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 2
                elif (peak_1_amplitude + peak_2_amplitude) * (1 - Parameters.value_settings[8]) > \
                        (Parameters.value_settings[4] + Parameters.value_settings[5]):
                    self.switch = 1
                else:
                    self.switch = 0
            else:
                print("Something wrong with the direction of regulation!")
        else:
            print("Something went wrong with the experimental condition!")
        # This saves the feedback values given, in both conditions, to a _score.csv file for later analysis.
        self.online_feedback = np.append(self.online_feedback, [[peak_1_amplitude, peak_2_amplitude, self.switch]], 0)
    #   Then, save data.
        if self.ending_counter >= Parameters.value_settings[2]:
            self.finishing = True
        if self.finishing:
            # noinspection PyTypeChecker
            self.elapsed_time = time() - self.starting_time
            print(self.elapsed_time)
            np.savetxt("data//" + str(int(Parameters.value_settings[0])) + "_" + str(
                int(Parameters.value_settings[1])) + "_mirror.csv",
                       all_clean_signal[self.start_time:,:],
                       header="channel_1,channel_2",
                       comments="", delimiter=',')
            # Scores are stored as an .npy to be indecipherable
            np.save("data//" + str(int(Parameters.value_settings[0])) + "_" + str(
                int(Parameters.value_settings[1])) + "_score.npy", self.online_feedback, allow_pickle=True)
            #np.savetxt("data//" + str(int(Parameters.value_settings[0])) + "_" + str(
            #    int(Parameters.value_settings[1])) + "_score.csv",
            #           self.online_feedback,
            #           header="channel_1,channel_2,feedback",
            #           comments="", delimiter=',')
            # Calculate average score to determine whether re-baselining is necessary
            self.timer.stop()
            overall_performance = np.mean(self.online_feedback[:,2])
            if abs(overall_performance) > CEILING_EFFECT:
                self.user_window = BaselineWindow()
                self.user_window.show()
            self.close()


# This is a simple pop-up window warning that a participant is ceiling level and must be dealt with.
class BaselineWindow(QWidget):
    # Initialisation is very simple.
    def __init__(self):
        # Formatting
        super(BaselineWindow, self).__init__()
        self.setWindowTitle("BrainMirror")
        self.layout = QGridLayout(self)
        label = QLabel("Ceiling effects are occuring!")
        self.layout.addWidget(label)
        widget = QPushButton("OK")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.finish)
        self.layout.addWidget(widget)

    def finish(self):
        self.close()


# This class is a simple selection of participant id and session id to assign the acquired baseline to.
# noinspection PyMethodMayBeStatic
class Usernamer(QWidget):
    def __init__(self):
        super().__init__()
        # This small window just asks for whom to get the baseline, buttons coded manually to save time.
        self.setWindowTitle("BrainMirror")
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Select user ID")
        self.layout.addWidget(self.label)
        widget = QSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(100000000)
        widget.setSingleStep(1)
        widget.setValue(int(Parameters.value_settings[0]))
        # noinspection PyUnresolvedReferences
        widget.valueChanged.connect(self.user_change)
        self.layout.addWidget(widget)
        self.setLayout(self.layout)
        self.user_window_2 = []
        # And the sessions is chosen in this one.
        self.label = QLabel("Select session ID")
        self.layout.addWidget(self.label)
        widget = QSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(100000000)
        widget.setSingleStep(1)
        widget.setValue(int(Parameters.value_settings[1]))
        # noinspection PyUnresolvedReferences
        widget.valueChanged.connect(self.session_change)
        self.layout.addWidget(widget)
        self.setLayout(self.layout)
        # And Finally here is an exit button.
        widget = QPushButton("Select")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.get_baseline)
        self.layout.addWidget(widget)

    # These are functions to process the signal.
    def user_change(self, i):  # i is an int
        Parameters.value_settings[0] = i

    def session_change(self, i):  # i is an int
        Parameters.value_settings[1] = i

    def get_baseline(self):
        self.user_window_2 = Baseliner()
        self.user_window_2.show()
        self.close()


# This class is its sister selection of participant id and session id to assign to the mirrorring session.
# It also performs quick rebaselining
# noinspection PyMethodMayBeStatic
class UsernamerMirror(QWidget):
    def __init__(self):
        super().__init__()
        # This small window just asks for whom to get the baseline.
        self.setWindowTitle("BrainMirror")
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Select user ID")
        self.layout.addWidget(self.label)
        widget = QSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(100000000)
        widget.setSingleStep(1)
        widget.setValue(int(Parameters.value_settings[0]))
        # noinspection PyUnresolvedReferences
        widget.valueChanged.connect(self.user_change)
        self.layout.addWidget(widget)
        self.setLayout(self.layout)
        self.user_window_2 = []
        # And the sessions is chosen in this one.
        self.label = QLabel("Select session ID")
        self.layout.addWidget(self.label)
        widget = QSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(100000000)
        widget.setSingleStep(1)
        widget.setValue(int(Parameters.value_settings[1]))
        # noinspection PyUnresolvedReferences
        widget.valueChanged.connect(self.session_change)
        self.layout.addWidget(widget)
        self.setLayout(self.layout)
        # The function also displays the save, check and load buttons.
        pretend_stability = QCheckBox("Rebaseline to previous")
        pretend_stability.setChecked(Parameters.index_settings["rebaseline"])
        pretend_stability.toggled.connect(self.rebaseline)
        self.layout.addWidget(pretend_stability)
        # And here is an exit button.
        widget = QPushButton("Select")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.get_baseline)
        self.layout.addWidget(widget)
        # And finally display the number of conditions left to assign.
        widget = QPushButton("Display amount of internal IDs")
        # noinspection PyUnresolvedReferences
        widget.clicked.connect(self.display_left_conditions)
        self.layout.addWidget(widget)


    # These are functions to process the signal.
    def user_change(self, i):  # i is an int
        Parameters.value_settings[0] = i

    def session_change(self, i):  # i is an int
        Parameters.value_settings[1] = i

    # Check is switched around
    def rebaseline(self, i):
        Parameters.index_settings["rebaseline"] = i

    # Start of chain of functions to launch mirroring
    def get_baseline(self):
        self.load()

    def display_left_conditions(self):
        conditions_left = np.load("conditions//encrypted_secrets.npy", allow_pickle=True)
        print(len(conditions_left))

    # This function subdivides the extraction from the two individual channels.
    def recalculate_identically(self, dataset, channel=0):
        # The analysis is done both for detecting individual peaks and power
        semi_processed_signal = dataset[:, channel]
        semi_processed_signal = semi_processed_signal.squeeze()
        processed_signal_length = int(np.round(len(semi_processed_signal) / 128, 0) - 1)
        means_vector = []
        for i in range(processed_signal_length):
            window_mean = np.mean(semi_processed_signal[i:i + 128])
        means_vector = np.append(means_vector, window_mean)
        mean_amplitude = np.mean(means_vector)
        return mean_amplitude

    # This function rebaselines to a simple average of channels from the previous block
    def load(self):
        global some_conditions_left
        if Parameters.index_settings["rebaseline"]:
            try:
                # Uses the previous block, that's why it is -1
                self.all_channels_raw = np.genfromtxt("data//" + str(int(Parameters.value_settings[0])) + "_" +
                                                          str(int(Parameters.value_settings[1]-1)) + "_mirror.csv",
                                                          delimiter=',')
                self.all_channels_raw = np.delete(self.all_channels_raw, 0, 0)
                self.peak_1_amplitude = self.recalculate_identically(dataset=self.all_channels_raw, channel=0)
                self.peak_2_amplitude = self.recalculate_identically(dataset=self.all_channels_raw, channel=1)

                # This is key, as it displays the participant update.
                Parameters.value_settings[4] = self.peak_1_amplitude
                Parameters.value_settings[5] = self.peak_2_amplitude
                # REMOVE THIS: To be deleted
                #
                #
                # print("left: " + str(Parameters.value_settings[4]))
                print("Thresholds set.")
                # print("right: " + str(Parameters.value_settings[5]))
                #
                #
                self.user_window_2 = Feedback()
                if some_conditions_left:
                    self.user_window_2.show()
                self.close()
            except Exception:
                # noinspection PyAttributeOutsideInit
                self.user_window = WarningWindow()
                self.user_window.show()
        else:
            self.user_window_2 = Feedback()
            if some_conditions_left:
                self.user_window_2.show()
            self.close()


# This class takes the baseline of the signal in quite similar manner to infer the individual values for
# noinspection PyUnresolvedReferences,PyArgumentList,PyTypeChecker
class Baseliner(QWidget):

    def __init__(self):
        # Formatting
        super().__init__()
        self.setWindowTitle("BrainMirror")
        self.showMaximized()
        self.showFullScreen()
        # Timer
        self.timer = QTimer()
        # Recommendation from Suji Satkunarajah, thanks!
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(1000)  # 100ms = 1/10th sec
        self.timer.timeout.connect(self.baseline_read)
        self.timer.start()
        # Used to recalculate
        self.counter = 0
        self.starting_time = len(all_clean_signal[:,0])
        self.progress = QProgressBar(self, maximum=int(Parameters.value_settings[11])) #* SRATE * 2)#SRATE)
        self.progress.setGeometry(700, 500, 500, 20)
        self.progress.show()

    def baseline_read(self):
        global all_signal
        self.counter += 1
        # Updates the counter
        self.progress.setValue(self.counter)
        # Saves as back up what is recorded every 5 seconds in case of sudden disruption.
        # Closes and saves at the end.
        if self.counter == int(Parameters.value_settings[11]):
            # print(len(clean_signal_1)-self.starting_time)
            # print(len(clean_signal_2)-self.starting_time)
            np.savetxt("data//" + str(int(Parameters.value_settings[0])) + "_" +
                       str(int(Parameters.value_settings[1])) + "_baseline_processed.csv",
                       all_clean_signal[self.starting_time:,:],
                       header="channel_1,channel_2",
                       comments="", delimiter=',')
            np.savetxt("data//" + str(int(Parameters.value_settings[0])) + "_" +
                       str(int(Parameters.value_settings[1])) + "_baseline_raw.csv",
                       all_signal[self.starting_time:,:],
                       header="channel_1,channel_2,timestamp",
                       comments="", delimiter=',')
            self.close()

    # Exit by left mouse click.
    # noinspection PyPep8Naming
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
        pretend_stability = QCheckBox("Remove 1/f")
        pretend_stability.setChecked(Parameters.index_settings["pretend_stability"])
        pretend_stability.toggled.connect(self.stability_pretended)
        self.layout.addWidget(pretend_stability, 16, 0)
        # This function enables to directly rebaseline from a feedback signal
        session_extraction = QCheckBox("Extract from session")
        session_extraction.setChecked(Parameters.index_settings["session_extract"])
        session_extraction.toggled.connect(self.session_extracted)
        self.layout.addWidget(session_extraction, 17, 0)
        # Displays a statement that double-blinding prevents thresholds from being displayed
        self.label = QLabel("DOUBLE-BLINDED")
        self.layout.addWidget(self.label, 16, 1)
        self.label = QLabel("(Peaks shown as 1)")
        self.layout.addWidget(self.label, 17, 1)
        self.blind_this()

        button_load = QPushButton("Calculate IAF")
        button_load.clicked.connect(partial(self.load, plot=1, get_iaf=1))
        self.layout.addWidget(button_load, 18, 0)

        button_load = QPushButton("Detect threshold")
        button_load.clicked.connect(partial(self.load, plot=0, get_iaf=0))
        self.layout.addWidget(button_load, 18, 1)

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

    def stability_pretended(self, i):
        Parameters.index_settings["pretend_stability"] = i

    def session_extracted(self, i):
        Parameters.index_settings["session_extract"] = i

    # Simply masks the menu items showing the participant peak values with 1, gets called each time all is redrawn
    def blind_this(self):
        widget = QDoubleSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(1)
        widget.setSingleStep(1)
        widget.setValue(1)
        self.layout.addWidget(widget, 5, 0)
        self.setLayout(self.layout)
        widget = QDoubleSpinBox()
        widget.setMinimum(0)
        widget.setMaximum(1)
        widget.setSingleStep(1)
        widget.setValue(1)
        self.layout.addWidget(widget, 7, 0)
        self.setLayout(self.layout)

    # This function calculates the averages from a specified participant and session.
    # noinspection PyAttributeOutsideInit
    def process_channel(self, dataset, channel=0):
        threshold_value = np.mean(dataset[:,channel])
        return threshold_value

    # This functions sets up calculations for IAF
    def get_threshold(self, dataset, channel=0, plot=0):
        # The analysis is done both for detecting individual peaks and power
        processed_signal = dataset[:,channel].squeeze()
        processed_signal = detrend(processed_signal)
        processed_signal = mne.filter.filter_data(processed_signal, sfreq=SRATE,
                                                  l_freq=Parameters.value_settings[13],
                                                  h_freq=Parameters.value_settings[14],
                                                  method="iir", verbose="ERROR")
        freq_data = fft(processed_signal)  # (** 2)
        # Same duplicate equations as when giving feedback, over more data and so with different constants.
        number = len(processed_signal)
        y = 2 / number * np.abs(freq_data[0: int(number / 2)])
        y = savitzky_golay(y, Parameters.value_settings[3], 5)
        freq_data = np.linspace(0, SRATE / 2, int(number / 2))  # * 2
        index_low = np.where(freq_data > LOWER_ALPHA)[0][0]
        index_high = np.where(freq_data > HIGHER_ALPHA)[0][0]
        threshold_value = np.max(y[index_low:index_high])
        peak_frequency = freq_data[np.where(y == threshold_value)]
        # Filter
        semi_processed_signal = dataset[:,channel]
        semi_processed_signal = semi_processed_signal.squeeze()
        processed_signal_length = int(np.round(len(semi_processed_signal)/128, 0)-1)
        means_vector = []
        pre_rolling_average = np.empty([0])
        # Debaselined away
        save_average = np.empty([0])
        averaging_counter = 25
        for i in range(processed_signal_length):
            # Matched up to baseline recalculation every 5 seconds
            if averaging_counter == 25:
                averaging_counter = -1
                substraction_mean = np.mean(semi_processed_signal[i:i+2560])
            averaging_counter += 1
            rolling_average = semi_processed_signal[i:i+128] - substraction_mean
            if len(pre_rolling_average) < WINDOW:
                pre_rolling_average = mne.filter.filter_data(np.append(pre_rolling_average, rolling_average),
                                                             sfreq=512,
                                                             l_freq=Parameters.value_settings[9],
                                                             h_freq=Parameters.value_settings[10],
                                                             method="iir", verbose="ERROR")
            elif len(pre_rolling_average) > WINDOW:
                pre_rolling_average = mne.filter.filter_data(np.append(pre_rolling_average[-WINDOW:], rolling_average),
                                                             sfreq=512,
                                                             l_freq=Parameters.value_settings[9],
                                                             h_freq=Parameters.value_settings[10],
                                                             method="iir", verbose="ERROR")
            rolling_average = abs(pre_rolling_average[-128:])
            if len(save_average) < WINDOW:
                save_average = savitzky_golay(np.append(save_average, rolling_average), Parameters.value_settings[3], 5)
            # Only apply the smoothing on a portion of the signal to avoid slowing
            # Or when it is getting longer to the last 2048 samples, the number may be changed.
            if len(save_average) > WINDOW:
                save_average = np.append(save_average[0:-WINDOW], savitzky_golay(
                    np.append(save_average[-WINDOW:], rolling_average), Parameters.value_settings[3], 5))
            # When computation is done from mirroring data, signal is just averaged and not pre-processed.
            if Parameters.index_settings["session_extract"]:
                window_mean = np.mean(semi_processed_signal[i:i + 128])
            elif not Parameters.index_settings["session_extract"]:
                window_mean = np.mean(save_average[-128:])
            means_vector = np.append(means_vector, window_mean)
        mean_amplitude = np.mean(means_vector)
        if plot == 0:
            return peak_frequency
        elif plot == 2:
            return mean_amplitude
        else:
            return [freq_data, y]

    # noinspection PyAttributeOutsideInit
    def load(self, plot=1, get_iaf=0):
        try:
            # This does the calculation.
            fm = FOOOF()
            if not Parameters.index_settings["session_extract"]:
                self.all_channels = np.genfromtxt("data//" + str(int(Parameters.value_settings[0])) + "_" +
                                                  str(int(Parameters.value_settings[1])) + "_baseline_processed.csv",
                                                  delimiter=',')
                self.all_channels = np.delete(self.all_channels, 0, 0)
                self.all_channels_raw = np.genfromtxt("data//" + str(int(Parameters.value_settings[0])) + "_" +
                                                  str(int(Parameters.value_settings[1])) + "_baseline_raw.csv",
                                                  delimiter=',')
            elif Parameters.index_settings["session_extract"]:
                self.all_channels_raw = np.genfromtxt("data//" + str(int(Parameters.value_settings[0])) + "_" +
                                                      str(int(Parameters.value_settings[1])) + "_mirror.csv",
                                                      delimiter=',')
            self.all_channels_raw = np.delete(self.all_channels_raw, 0, 0)
            if get_iaf == 1:
                self.peak_1_frequency = self.get_threshold(dataset=self.all_channels_raw, channel=0)
                self.peak_2_frequency = self.get_threshold(dataset=self.all_channels_raw, channel=1)
                Parameters.value_settings[6] = self.peak_1_frequency
                Parameters.value_settings[7] = self.peak_2_frequency
                Parameters.value_settings[9] = ((self.peak_1_frequency + self.peak_2_frequency) / 2) - AROUND
                Parameters.value_settings[10] = ((self.peak_1_frequency + self.peak_2_frequency) / 2) + AROUND
            self.peak_1_amplitude = self.get_threshold(dataset=self.all_channels_raw, channel=0, plot=2)
            self.peak_2_amplitude = self.get_threshold(dataset=self.all_channels_raw, channel=1, plot=2)
            # This is key, as it displays the participant update.
            Parameters.value_settings[4] = self.peak_1_amplitude
            Parameters.value_settings[5] = self.peak_2_amplitude
            self.display_menu()
            if plot == 1:
                matplotlib.use('QTAgg')
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
                plt.ion()
                plt.show()
                self.temp = self.get_threshold(channel=0, dataset=self.all_channels_raw, plot=1)
                axs[0, 0].set_xlim([0, 55])
                axs[0, 0].plot(self.temp[0], self.temp[1], label="channel_1", color="mediumblue")
                # However, 55 is a natural choice here as that shows how the electricity noise is faring.
                plt.xlim([0, 55])
                # There is an issue with how standard this parameter is.
                if Parameters.index_settings["pretend_stability"]:
                    fm.fit(self.temp[0], self.temp[1], [2, 40])
                    fm.power_spectrum[np.isnan(fm.power_spectrum)] = 0
                    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
                    init_flat_spec = fm.power_spectrum - init_ap_fit
                    plt_log = False
                    plot_spectrum(fm.freqs, init_flat_spec, plt_log,
                                  label='Flattened Spectrum', color='mediumblue', ax=axs[0,1])
                    # Extract peak frequency
                    self.peak_1_frequency = fm.freqs[np.argmax(init_flat_spec)]
                    # Check if the peak frequency falls outside a broad alpha range
                    self.peak_1_frequency = 0
                    i = int(1)
                    while self.peak_1_frequency < 8 or self.peak_1_frequency > 20:
                        self.peak_1_frequency = fm.freqs[np.where(init_flat_spec == np.partition(init_flat_spec, -i)[-i])]#fm.freqs[np.argmax(init_flat_spec)]
                        # print(self.peak_1_frequency)
                        i += 1
                    Parameters.value_settings[6] = self.peak_1_frequency
                self.temp = self.get_threshold(channel=1, dataset=self.all_channels_raw, plot=1)
                axs[1,0].set_xlim([0, 55])
                axs[1,0].plot(self.temp[0], self.temp[1], label="channel_2", color="plum")
                plt.xlim([0, 55])
                plt.title('Frequency domain Signal')
                plt.xlabel('Frequency in Hz')
                plt.ylabel('Amplitude')
                plt.tight_layout()
                plt.draw()
                if Parameters.index_settings["pretend_stability"]:
                    fm.fit(self.temp[0], self.temp[1], [2, 40])
                    fm.power_spectrum[np.isnan(fm.power_spectrum)] = 0
                    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
                    init_flat_spec = fm.power_spectrum - init_ap_fit
                    plt_log = False
                    plot_spectrum(fm.freqs, init_flat_spec, plt_log,
                                  label='Flattened Spectrum', color='plum', ax=axs[1,1])
                    self.peak_2_frequency = fm.freqs[np.argmax(init_flat_spec)]
                    self.peak_2_frequency = 0
                    i = int(1)
                    while self.peak_2_frequency < LOWER_ALPHA or self.peak_2_frequency > HIGHER_ALPHA:
                        self.peak_2_frequency = fm.freqs[np.where(init_flat_spec == np.partition(init_flat_spec, -i)[-i])]
                        # print(self.peak_2_frequency)
                        i += 1
                    # print(i)
                    Parameters.value_settings[7] = self.peak_2_frequency
                Parameters.value_settings[9] = ((self.peak_1_frequency + self.peak_2_frequency)/2) - AROUND
                Parameters.value_settings[10] = ((self.peak_1_frequency + self.peak_2_frequency)/ 2) + AROUND
                self.display_menu()
            # REMOVE THIS: To be deleted
            #
            #
            # print("left: " + str(Parameters.value_settings[4]))
            print("Thresholds set.")
            # print("right: " + str(Parameters.value_settings[5]))
            #
            #
        except Exception:
            # noinspection PyAttributeOutsideInit
            self.user_window = WarningWindow()
            self.user_window.show()

    # The save button currently only closes the programme.
    def finish(self):
        # This saves the settings but as they are to an array.
        np.save("settings.npy", Parameters.value_settings, allow_pickle=True)
        self.close()


# This is a simple pop-up window warning that a participant with that id and session number does not exist.
class WarningWindow(QWidget):
    # Initialisation is very simple.
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

######################################################################
# To-do list:
######################################################################

# Test with device
# TODO: Test with blindness

# Before real experiment
# TODO: Perhaps add in an actual deep learning algorhythm
# TODO: Add in more colours
# TODO: Considering Chris's IAF algorhythm

# After real experiment
# TODO: Run through chatgpt

# Cosmetics
##################
# Timestamp