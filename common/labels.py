import random


def gen_labels(trials, label_types):
    """
    Generate an array of length trials containing a random
    sequence of label types.
    """
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
