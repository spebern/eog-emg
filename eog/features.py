import numpy as np
from common.config import FS, SAMPLE_DURATION


def calc_pav(signal):
    """
    Get the maximum value.

    :param signal: signal to extract maximum from
    :return: maximum value of signal
    """
    return np.max(signal)


def calc_vav(signal):
    """
    Get the minimum value.

    :param signal: signal to extract minimum from
    :return: minimum value of signal
    """
    return np.min(signal)


def calc_auc(signal):
    return np.sum(np.abs(signal))


def calc_tcv(signal, threshold=2.0):
    """
    Calculate the count of threshold crossings for positive and negative threshold.

    :param signal: input signal
    :param threshold: threshold used for counting
    :return: number of times the threshold was crossed
    """
    tcv = 0
    prev = 0.0
    tm = -threshold
    for x in signal:
        if (x < threshold and prev > threshold) or (
            x > threshold and prev <= threshold
        ):
            prev = x
            tcv += 1
        elif (x < tm and prev > tm) or (x > tm and prev <= tm):
            prev = x
            tcv += 1
    return float(tcv)


def calc_var(signal):
    """
    Calculate the variance of the signal.

    :param signal: input signal
    :return: variance of the signal
    """
    return np.var(signal)


class FeatureExtractor:
    def __init__(
        self, fs=FS, sample_duration=SAMPLE_DURATION, win_size=1.5, win_overlap=1.0
    ):
        """
        Create a FeatureExtractor for EOG.

        :param fs: sampling frequency [Hz]
        :param sample_duration: duration of the sampling window [s]
        :param win_size: size of the windows features should be extracted from [s]
        :param win_overlap: size of the overlap of consecutive windows used for feature extraction [s]
        """
        self._sample_duration = SAMPLE_DURATION
        self._win_size = win_size
        self._win_overlap = win_overlap
        if win_overlap == 0.0:
            self._num_wins = 1
        else:
            self._num_wins = 1 + int(
                (sample_duration - win_size) / (win_size - win_overlap)
            )
        self._fs = fs
        self.num_features = 5 * self._num_wins

    def _extract_features_from_channel(self, signal):
        features = []
        start = 0
        end = int(self._win_size * self._fs)

        for i in range(self._num_wins):
            sig_win = signal[start:end]

            features.append(calc_pav(sig_win))
            features.append(calc_vav(sig_win))
            features.append(calc_auc(sig_win))
            features.append(calc_tcv(sig_win))
            features.append(calc_var(sig_win))

            start += int((self._win_size - self._win_overlap) * self._fs)
            end += int((self._win_size - self._win_overlap) * self._fs)

        return np.array(features)

    def extract_features(self, signals):
        """
        Extract features from all channels.

        Two channels at a time will be subtracted from each other (bipolar measurement).
        Therefor the number of channels given to this function should be even.

        :param signals: filtered signals of the eog channels
        """
        num_bipolar_channels = signals.shape[1] // 2
        features = np.zeros(num_bipolar_channels * self.num_features)

        for i in range(num_bipolar_channels):
            sigs = signals[:, i * 2] - signals[:, i * 2 + 1]
            features[
                i * self.num_features : (i + 1) * self.num_features
            ] = self._extract_features_from_channel(sigs[:])

        return features
