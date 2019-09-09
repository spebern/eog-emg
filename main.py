import fire
from common.recorder import Recorder
from common.preprocessing import BandPassFilter
from common.utils import (
    read_emg_training_data,
    read_eog_training_data,
    save_emg_model,
    save_eog_model,
    load_latest_eog_model,
    load_latest_emg_model,
    save_live_data,
    play_sound,
)
from emg.features import FeatureExtractor as EMGFeatureExtractor
from eog.features import FeatureExtractor as EOGFeatureExtractor
from eog.labels import EyeMovement
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from common.config import FS
from robot.robot import Robot
import time
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})


class App(object):
    def __init__(self):
        pass

    def record_emg(self, trials=10):
        """
        Record emg data.

        Runs the experiment with animations while recording the trials of the subject.
        The data will be saved to training_data/emg and can later be used for training.
        When running multiple sessions data should be moved to somewhere else manually.

        :param trials: number of trials to record
        """
        from emg.animation import Animation as EMGAnimation

        recorder = Recorder(signal_type="emg", num_channels=2, channel_offset=0)
        recorder.start_offline_recording()

        animation = EMGAnimation(recorder, trials)
        animation.run()

    def record_eog(self, trials=10):
        """
        Record eog data.

        Runs the experiment with animations while recording the trials of the subject.
        The data will be saved to training_data/emg and can later be used for training.
        When running multiple sessions data should be moved to somewhere else manually.

        :param trials: number of trials to record
        """
        from eog.animation import Animation as EOGAnimation

        recorder = Recorder(signal_type="eog", num_channels=4, channel_offset=4)
        recorder.start_offline_recording()

        animation = EOGAnimation(recorder, trials)
        animation.run()

    def train_emg(self):
        """
        Training an EMG model and save it to disk.

        Data inside training_data/emg is used for training.
        The model is saved with pickle to model/emg.
        """
        signals, labels = read_emg_training_data()

        # bandpass filter and extract features
        feature_extractor = EMGFeatureExtractor()
        bp = BandPassFilter(lowcut=30, highcut=500, fs=FS, order=4)
        features = []
        for i in range(len(signals)):
            for channel in range(signals[i].shape[1]):
                signals[i, :, channel] = bp(signals[i, :, channel])
            features.append(feature_extractor.extract_features(signals[i]))
        features = np.array(features)

        # select features
        clf = KNeighborsClassifier(n_neighbors=8)
        fsl = SFS(
            clf,
            verbose=2,
            floating=True,
            forward=True,
            k_features=10,
            scoring="accuracy",
            cv=12,
            n_jobs=-1,
        )
        fsl.fit(features, labels)

        # train final classifier with the extracted features and save a model to disk
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(features[:, fsl.k_feature_idx_], labels)

        model = {"clf": clf, "best_features": fsl.k_feature_idx_}
        save_emg_model(model)

    def train_eog(self):
        """
        Training an EOG model and save it to disk.

        Data inside training_data/eog is used for training.
        The model is saved with pickle to model/eog.
        """
        feature_names = [
            "pav_ch_1_win_1",
            "vav_ch_1_win_1",
            "tcv_ch_1_win_1",
            "var_ch_1_win_1",
            "pav_ch_1_win_2",
            "vav_ch_1_win_2",
            "tcv_ch_1_win_2",
            "var_ch_1_win_2",
            "pav_ch_2_win_1",
            "vav_ch_2_win_1",
            "tcv_ch_2_win_1",
            "var_ch_2_win_1",
            "pav_ch_2_win_2",
            "vav_ch_2_win_2",
            "tcv_ch_2_win_2",
            "var_ch_2_win_2",
        ]

        signals, labels = read_eog_training_data()

        feature_extractor = EOGFeatureExtractor()

        # bandpass filter and extract features
        features = []
        bp = BandPassFilter(lowcut=1, highcut=22, fs=FS, order=4)
        for i in range(len(signals)):
            for channel in range(signals[i].shape[1]):
                signals[i, :, channel] = bp(signals[i, :, channel])
            features.append(feature_extractor.extract_features(signals[i]))
        features = np.array(features)

        # extract labels for relax vs other eye movement
        relax_labels = []
        for label in labels:
            if label == "relax":
                relax_labels.append("relax")
            else:
                relax_labels.append("other")

        # extract the features for eye movement classification training
        not_relax_features = []
        not_relax_labels = []
        for i in range(len(labels)):
            if labels[i] != "relax":
                not_relax_labels.append(str(labels[i]))
                not_relax_features.append(features[i])
        not_relax_features = np.array(not_relax_features)

        # select features for eye movement classification
        clf = KNeighborsClassifier(n_neighbors=3)
        fsl = SFS(
            clf,
            verbose=0,
            floating=True,
            forward=True,
            k_features=8,
            scoring="accuracy",
            cv=10,
            n_jobs=-1,
        )
        not_relax_labels = np.array(np.copy(not_relax_labels))
        fsl.fit(
            not_relax_features, not_relax_labels, custom_feature_names=feature_names
        )

        # train eye movement classifier
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(not_relax_features[:, fsl.k_feature_idx_], not_relax_labels)

        # extract features for relax vs eye movement classifier and train
        relax_clf = KNeighborsClassifier(n_neighbors=3)
        relax_fsl = EFS(
            relax_clf,
            min_features=4,
            max_features=4,
            scoring="accuracy",
            print_progress=True,
            n_jobs=-1,
            cv=4,
        )
        relax_fsl.fit(features, relax_labels)

        # train relax vs eye movement classifier with extracted features
        relax_clf = KNeighborsClassifier(n_neighbors=3)
        relax_clf.fit(features[:, relax_fsl.best_idx_], relax_labels)

        # save both classifiers as model to disk
        model = {
            "clf": clf,
            "best_features": fsl.k_feature_idx_,
            "relax_clf": relax_clf,
            "best_relax_features": relax_fsl.best_idx_,
        }
        save_eog_model(model)

    def evaluate_emg(self, trials=10):
        """
        Evaluate EMG model performance.

        The most recent model stored inside model/emg is used to do a quick
        live validation. While the subject executes the actions the classified
        label is printed to stdout.

        :param trials: number of trials
        """
        recorder = Recorder(num_channels=2)

        model = load_latest_emg_model()
        clf = model["clf"]
        best_features = model["best_features"]
        feature_extractor = EMGFeatureExtractor()
        bp = BandPassFilter(lowcut=30, highcut=500, fs=FS, order=4)
        for _ in range(trials):
            signals = recorder.read_sample_win()
            features = []
            for channel in range(signals.shape[1]):
                signals[:, channel] = bp(signals[:, channel])
            features.append(feature_extractor.extract_features(signals))
            features = np.array(features)

            label = clf.predict([features[0, best_features]])[0]
            print(label)

    def evaluate_eog(self, trials=10):
        """
        Evaluate EOG model performance.

        The most recent model stored inside model/eog is used to do a quick
        live validation. After a short tone the sampling is started. After
        classification an arrow should show at the position the subject
        looked at.

        :param trials: number of trials
        :return:
        """
        from eog.animation import Animation as EOGAnimation

        recorder = Recorder(num_channels=4, channel_offset=4)
        animation = EOGAnimation(None)
        feature_extractor = EOGFeatureExtractor()

        model = load_latest_eog_model()

        clf = model["clf"]
        best_features = model["best_features"]

        relax_clf = model["relax_clf"]
        best_relax_features = model["best_relax_features"]

        bp = BandPassFilter(lowcut=1, highcut=22, fs=FS, order=4)
        for _ in range(trials):
            # play short tone to notify the subject that sampling is about to start
            play_sound()

            signals = recorder.read_sample_win()
            features = []
            for channel in range(signals.shape[1]):
                signals[:, channel] = bp(signals[:, channel])
            features.append(feature_extractor.extract_features(signals))
            features = np.array(features)

            label = relax_clf.predict([features[0, best_relax_features]])[0]

            # display the classified action
            if label == "relax":
                animation.display_label(EyeMovement.RELAX)
            else:
                label = clf.predict([features[0, best_features]])[0]
                if label == "right":
                    animation.display_label(EyeMovement.RIGHT)
                elif label == "left":
                    animation.display_label(EyeMovement.LEFT)
                elif label == "up":
                    animation.display_label(EyeMovement.UP)
                elif label == "down":
                    animation.display_label(EyeMovement.DOWN)

            # hold the displayed action
            time.sleep(1)

            # go back to black screen
            animation.display_label(EyeMovement.RELAX)

    def evaluate_both(self, trials=10):
        """
        Run the live experiment controlling the robot.

        For each trials first EOG is classified which determines the position
        the robot moves to. After the robot reached its final position EMG
        is classified controlling the gripper.
        Recorded data is saved into the live_data folder.

        :param trials: number of trials
        :return:
        """
        recorder = Recorder(num_channels=8)

        # load models
        emg_model = load_latest_emg_model()
        emg_clf = emg_model["clf"]
        emg_best_features = emg_model["best_features"]
        emg_feature_extractor = EMGFeatureExtractor()

        eog_model = load_latest_eog_model()
        eog_clf = eog_model["clf"]
        eog_best_features = eog_model["best_features"]
        eog_relax_clf = eog_model["relax_clf"]
        eog_best_relax_features = eog_model["best_relax_features"]
        eog_feature_extractor = EOGFeatureExtractor()
        eog_bp = BandPassFilter(lowcut=1, highcut=22, fs=FS, order=4)

        # discord the first two sampling windows, which gets rid of a spike at the beginning
        recorder.read_sample_win()
        recorder.read_sample_win()

        robot = Robot()

        for _ in range(trials):
            # notify the subject that sampling is about to start
            play_sound()

            signals = recorder.read_sample_win()

            eog_signals = signals[:, 4:8]

            # bandpass filter and extract features
            eog_features = []
            for channel in range(eog_signals.shape[1]):
                eog_signals[:, channel] = eog_bp(eog_signals[:, channel])
            eog_features.append(eog_feature_extractor.extract_features(eog_signals))
            eog_features = np.array(eog_features)
            eog_label = eog_relax_clf.predict(
                [eog_features[0, eog_best_relax_features]]
            )[0]

            save_live_data("eog", eog_label, eog_signals)

            if eog_label == "relax":
                # continue if home position
                robot.move_home()
                continue
            else:
                # classify the eye movement label and move the robot into position
                eog_label = eog_clf.predict([eog_features[0, eog_best_features]])[0]
                save_live_data("eog", eog_label, eog_signals)
                if eog_label == "left":
                    robot.move_up_left()
                elif eog_label == "up":
                    robot.move_up_right()
                elif eog_label == "right":
                    robot.move_down_right()
                elif eog_label == "down":
                    robot.move_down_left()

            # notify the subject that sampling is about to start
            play_sound()

            signals = recorder.read_sample_win()
            emg_signals = signals[:, :2]

            emg_features = []
            emg_features.append(emg_feature_extractor.extract_features(emg_signals))
            emg_features = np.array(emg_features)

            emg_label = emg_clf.predict([emg_features[0, emg_best_features]])[0]
            save_live_data("emg", emg_label, emg_signals)
            if emg_label == "fist":
                robot.close_gripper()
            else:
                robot.open_gripper()

            # hold position for a little
            time.sleep(2)

            # return to initial state
            robot.move_home()


def main():
    fire.Fire(App)


if __name__ == "__main__":
    main()
