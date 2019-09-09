import random
from emg.labels import HandGesture


def gen_labels(trials):
    """
    Generate an array of length trials containing a random sequence of label types separated by relax labels.

    :param trials: number of trials
    :return random sequence of labels separated by relaxation labels
    """
    label_types = list(HandGesture)
    labels = (trials // (len(label_types) - 1)) * label_types[1:]
    while len(labels) < trials:
        labels.append(random.choice(label_types[1:]))
    random.shuffle(labels)

    relaxtion_label = label_types[0]
    labels_with_relaxation = []
    for label in labels:
        for _ in range(5):
            labels_with_relaxation.append(relaxtion_label)
        for _ in range(5):
            labels_with_relaxation.append(label)
    return labels_with_relaxation
