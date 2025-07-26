"""
This is a simple server generating a fluctuating sine wave
and sending it over via LSL as three channels.
"""

import sys
import getopt
import time

import numpy as np
from random import random as rand
from pylsl import StreamInfo, StreamOutlet, local_clock


# The output is described and generic LSL formatting is used to reach the goal to enable communication.
def main(argv):
    sampling_rate = 512
    name = 'Nexus-10'
    type_of_signal = 'EEG'
    n_channels = 3
    help_string = 'SendStable.py -s <sampling_rate> -n <stream_name> -t <stream_type>'
    try:
        opts, args = getopt.getopt(argv, "hs:c:n:t:", longopts=["srate=", "channels=", "name=", "type"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--srate"):
            sampling_rate = float(arg)
        elif opt in ("-c", "--channels"):
            n_channels = int(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-t", "--type"):
            type_of_signal = arg

    info = StreamInfo(name, type_of_signal, n_channels, sampling_rate, 'float32', 'myuid34234')
    outlet = StreamOutlet(info)

    print("now sending data...")

    # Time is set up for timestamping
    start_time = local_clock()
    sent_samples = 0
    n = 0
    # Sampling rate and initial random y
    sample = 512
    amplitude = 1.5 + 0.05 * rand()
    variation = 0.5
    f = 9.8 + 0.4 * rand()
    frames_per_second = sample
    y = np.random.normal(0, 0, sample)
    counter = 0
    while True:
        elapsed_time = local_clock() - start_time
        # Required samples is always 1 at the moment, but may change later on.
        required_samples = int(sampling_rate * elapsed_time) - sent_samples
        # The loop samples a new wave to send.
        for sample_ix in range(required_samples):
            # This here is just to introduce noisiness for better simulation
            # The wave is sent thrice slightly misaligned.
            if counter < 1000:
                sent_sample = [y[n], y[n+1]*2, elapsed_time]
                counter += 1
            elif counter < 2000:
                sent_sample = [y[n], y[n+1]/2, elapsed_time]
                counter += 1
            else:
                counter = 0
            # Number 3 was chosen because the sample is sent thrice.
            if n >= sample - 3:
                n = 0
                x = np.arange(sample)
                # Generates a sine wave with some varying amplitude each time and a bit of noise on top.
                y = amplitude * (1 + (rand() * variation)) * np.sin(2 * np.pi * f * x / frames_per_second) +\
                    np.random.normal(0, 0.1, frames_per_second)
            else:
                n = n + 1
            # The information is sent.
            outlet.push_sample(sent_sample)
        sent_samples += required_samples
        time.sleep(1/sample)


# This here is generic LSL initialisation.
if __name__ == '__main__':
    main(sys.argv[1:])
