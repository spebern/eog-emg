import numpy as np
import threading
import queue
import time
from common import gtec
import matplotlib.pyplot as plt
from common.config import *


class Recorder:
    def __init__(
        self,
        sample_duration=SAMPLE_DURATION,
        num_channels=2,
        channel_offset=0,
        signal_type="emg",
    ):
        """
        Create a Recorder for EOG/EMG signals

        :param sample_duration: duration of sampling windows [s]
        :param num_channels: number of channels to record
        :param channel_offset: number of channels to skip
        :param signal_type: emg/eog
        """
        self._stop = False
        self._sample_duration = sample_duration
        self._num_channels = num_channels
        self._channel_offset = channel_offset
        self._labels = queue.Queue()
        self._signal_type = signal_type

        self._amp = gtec.GUSBamp()
        self._amp.set_sampling_frequency(
            FS, [True for i in range(16)], None, (48, 52, FS, 4)
        )
        self._amp.start()

    def start_offline_recording(self, live=True):
        """
        Start a thread for recording.
        """
        threading.Thread(target=self._record).start()

    def stop_offline_recording(self):
        """
        Terminate the recording thread.
        """
        self._stop = True

    def get_data(self):
        """
        Get data for the duratoin of the previously defined sample_duration.
        """
        signals, _ = self._amp.get_data()
        return signals

    def read_sample_win(self, duration=None):
        """
        Read in a sample window.

        :param duration: duration to sample [s], if left out, the duration passed to the constructor will be used
        :return: sampled signals
        """
        if duration is None:
            num_samples = int(self._sample_duration * FS)
        else:
            num_samples = int(duration * FS)
        sample_win = np.zeros((num_samples, self._num_channels))

        # start sampling
        num_collected_samples = 0
        sampling = True
        while sampling:
            signals, _ = self._amp.get_data()
            for i_sample in range(signals.shape[0]):
                for channel in range(self._num_channels):
                    sample_win[num_collected_samples, channel] = signals[
                        i_sample, channel + self._channel_offset
                    ]
                num_collected_samples += 1
                if num_collected_samples == num_samples:
                    sampling = False
                    return sample_win

    def record_label(self, label):
        """
        Queue a label to be recorded

        :param label:
        """
        self._labels.put(label)

    def _record(self):
        while not self._stop:
            label = self._labels.get()
            signals = self.read_sample_win()

            np.savez(
                "training_data/{}/{}.npz".format(self._signal_type, time.time()),
                signals=signals,
                label=label.value,
            )


def main():
    recorder = Recorder(sample_duration=6)
    raw_data = recorder.read_sample_win()

    fig = plt.figure(figsize=(12, 10))

    ax2 = fig.add_subplot(2, 1, 1)
    ax2.set_title("Signal channel 2 - channel 1")
    ax2.set_xlabel("samples")
    ax2.set_ylabel("voltage")
    ax2.plot(raw_data[2 * 1200 :, 1] - raw_data[2 * 1200 :, 0])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title("Signal channel 4 - channel 3")
    ax2.set_xlabel("samples")
    ax2.set_ylabel("voltage")
    ax2.plot(raw_data[2 * 1200 :, 3] - raw_data[2 * 1200 :, 2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
