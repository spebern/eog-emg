# EOG-EMG Experiment

Framework for controling a robot using EOG/EMG data. Recording is done with a g.usb Amplifier.

``` shell
python main.py --help
```

## Record data with animations

``` shell
python main.py record_emg --trials 100
python main.py record_eog --trials 100
```

# Train classifiers

``` shell
python main.py train_emg
python main.py train_eog
```

# Evaluate classifiers live

``` shell
python main.py evaluate_emg --trials 100
python main.py evaluate_eog --trials 100
```

# Control robot using EMG/EOG

``` shell
python main.py evaluate_both --trials 100
```
