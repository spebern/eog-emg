import random
from eog.labels import EyeMovement


def gen_labels(trials):
    """
    Generate an array of length trials containing a random sequence of label types.

    :param trials: number of trials
    :return sequence of labels where relax labels are at even positions and a random other labels at uneven positions
    """

    # get a randomized sequence of all labels but relaxation
    label_types = list(EyeMovement)
    labels = (trials // (len(label_types) - 1)) * label_types[1:]
    while len(labels) < trials:
        labels.append(random.choice(label_types[1:]))
    random.shuffle(labels)

    # set a relaxation label in front of every other label
    relaxtion_label = label_types[0]
    labels_with_relaxation = []
    for label in labels:
        labels_with_relaxation.append(relaxtion_label)
        labels_with_relaxation.append(label)
    return labels_with_relaxation
