import numpy as np
from common.config import FS


class FeatureExtractor:
    def __init__(self, duration=1, fs=FS, win_size=0.2, win_shift=0.04):
        """
        Create a FeatureExtraction for EMG signals.

        :param duration: duration of the sampling window [s]
        :param fs: sampling frequency [Hz]
        :param win_size: size of the window [s]
        :param win_shift: overlap between the windows [s]
        """
        self.fs = fs
        self.duration = duration
        self.win_size = win_size
        self.win_shift = win_shift
        self.num_windows = int(self.duration / (self.win_size - self.win_shift))
        self.num_features = self.num_windows * 2

    def _extract_features_from_channel(self, signal):
        mav_features = np.zeros(self.num_windows)
        wl_features = np.zeros(self.num_windows)

        for i in range(self.num_windows):
            start = int((i * (self.win_size - self.win_shift)) * self.fs)
            end = int(((i + 1) * self.win_size - i * self.win_shift) * self.fs)

            print("start: ", start / 1200)
            print("end: ", end / 1200)

            mav = np.mean(np.abs(signal[start:end]))
            wl = np.sum(np.abs(np.diff(signal[start:end])))

            wl_features[i] = wl
            mav_features[i] = mav

        import sys

        sys.exit()

        return mav_features, wl_features

    def extract_features(self, signals):
        """
        Extract EMG features from signals.

        Two signals at a time are subtracted from each other and then processed (bipolar).

        :param signals: signals containing the EMG data
        :return: features
        """
        num_bipolar_channels = signals.shape[1] // 2
        features = np.zeros(num_bipolar_channels * self.num_features)
        for i in range(num_bipolar_channels):
            sigs = signals[:, i * 2] - signals[:, i * 2 + 1]
            features[
                i * self.num_features : (i + 1) * self.num_features
            ] = np.concatenate(self._extract_features_from_channel(sigs[:]))

        return features
