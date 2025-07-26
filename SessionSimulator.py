import getopt
import numpy as np
import sys
import threading

from pylsl import StreamInfo, StreamOutlet, local_clock
from time import sleep

# Global constants
sample = 512
# Generates an alpha-like oscillation with some natural variations each time
def make_a_wave(bonus = 1):
    frequency = 10 #+ np.random.randint(1)
    amplitude = 60 #+ np.random.randint(1)
    positions = np.arange(sample)
    # Makes a sine wave according to the formula
    static_wave = 200 + amplitude * bonus * np.sin(2 * np.pi * frequency * positions / sample)# +\
      #  np.random.normal(0, 1, sample)
    return static_wave

# Semi-periodically regenerates the sinewave to simulate fluctuations.
def redraw_wave():
    while True:
        global static_wave
        global static_wave_2
        sleep(10)
        print("wave reversion")
        static_wave = make_a_wave(1)
        #sleep(3 + np.random.randint(1))
        #static_wave = make_a_wave(1)
        sleep(10)
        print("wave change")
        static_wave = make_a_wave(5)

# Starts up the LSL server and kicks off other functions.
def start_sender(argv):
    global sample
    name = 'Nexus-10'
    type_of_signal = 'EEG'
    # Number of channels to be moved up to 2
    n_channels = 1
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
    info = StreamInfo(name, type_of_signal, n_channels, sample, 'float32', 'myuid34234')
    global outlet
    outlet = StreamOutlet(info, chunk_size=128, max_buffered=1000)
    print("now sending data...")
    global static_wave
    global static_wave_2
    static_wave = make_a_wave()
    static_wave_2 = make_a_wave()
    thread3 = threading.Thread(target=redraw_wave)
    thread3.start()
    thread4 = threading.Thread(target=inject_na)
    thread4.start()
    send_wave(outlet)

#
def inject_na():
    global static_wave
    static_wave_old = static_wave[50]
    while True:
        sleep(10)
        #static_wave[50:67] = "NaN"
        sleep(2)
        #static_wave[50:67] = 0
        sleep(1)
        #static_wave[50:67] = 100
        sleep(1)
        #static_wave[50:67] = static_wave_old


def send_wave(outlet):
    # Sends a segment of the wave, 64 at a time
    wave_position = -1
    global static_wave
    global static_wave_2
    sent_sample = np.empty(0)
    while True:
        for i in range(64):
            if wave_position >= sample - 1:
                wave_position = 0
            else:
                wave_position += 1
            # Sent in a list format, LSL likes that
            sent_sample = np.append(sent_sample, [static_wave[wave_position], static_wave_2[wave_position]]).tolist()
        sleep(0.125)
        outlet.push_chunk(sent_sample)
        sent_sample = np.empty(0)

# Starting function
if __name__ == '__main__':
    start_sender(sys.argv[1:])

# Set more channel info!
# Incorporate back into MainApp