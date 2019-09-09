import numpy as np
import pickle
import time
import os

EMG_TRAINING_DATA_DIR = "training_data/emg/"
EOG_TRAINING_DATA_DIR = "training_data/eog/"

LIVE_DATA_DIR = "live_data/"

EMG_MODEL_DIR = "models/emg/"
EOG_MODEL_DIR = "models/eog/"


def _read_training_data(training_data_dir):
    labels = []
    signals = []
    for filename in sorted(os.listdir(training_data_dir)):
        fullpath = os.path.join(training_data_dir, filename)
        data = np.load(fullpath)
        labels.append(data["label"])
        signals.append(data["signals"])
    signals = np.array(signals)
    return signals, labels


def read_emg_training_data():
    """
    Read in the current EMG training data.

    :return: EMG training data
    """
    return _read_training_data(EMG_TRAINING_DATA_DIR)


def read_eog_training_data():
    """
    Read in the current EOG training data.

    :return: EOG training data
    """
    return _read_training_data(EOG_TRAINING_DATA_DIR)


def _save_model(model, model_dir):
    pickle.dump(model, open(os.path.join(model_dir, "{}.p".format(time.time())), "wb"))


def save_emg_model(model):
    """
    Save EMG model to disk using pickle.

    :param model: EMG model to save to disk
    """
    _save_model(model, EMG_MODEL_DIR)


def save_eog_model(model):
    """
    Save EOG model to disk using pickle.

    :param model: EOG model to save to disk
    """
    _save_model(model, EOG_MODEL_DIR)


def _load_latest_model(model_dir):
    filename = sorted(os.listdir(model_dir))[-1]
    fullpath = os.path.join(model_dir, filename)
    return pickle.load(open(fullpath, "rb"))


def load_latest_emg_model():
    """
    Load latest EMG model from disk.

    :return latest EMG model
    """
    return _load_latest_model(EMG_MODEL_DIR)


def load_latest_eog_model():
    """
    Load latest EOG model from disk.

    :return latest EOG model
    """
    return _load_latest_model(EOG_MODEL_DIR)


def save_live_data(signal_type, label, signals):
    """
    Save live data with timestamp to disk into live_data/ folder

    :param signal_type: type of the signal (e.g. eog, emg)
    :param label: classification label
    :param signals: signal data
    """
    ts = time.time()

    np.savez(
        os.path.join(LIVE_DATA_DIR, "{}_{}_{}".format(str(ts), signal_type, label)),
        signals=signals,
        signal_type=signal_type,
        label=label,
    )


def play_sound():
    """
    Play a short sound.
    """
    os.system("play -nq -t alsa synth {} sine {}".format(0.1, 440))
