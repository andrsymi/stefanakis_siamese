# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def nway_one_shot(model, x, y, n_way, n_val):
    print("\n[INFO] Performing nway validation...")
    n_correct = 0

    for i in range(n_val):
        # select a random positive sample
        positive_index = np.random.choice(np.argwhere(y == 1)[:, 0])
        # select n_way - 1 random negative samples
        negative_indices = np.random.choice(np.argwhere(y == 0)[:, 0], n_way - 1)

        # create new x_test array
        x_test = np.concatenate((x[negative_indices], np.expand_dims(x[positive_index], 0)))

        result = model.predict([x_test[:, 0], x_test[:, 1]], verbose=0)
        result_index = np.argmin(result)
        if result_index == len(x_test) - 1:
            n_correct = n_correct + 1
    print(n_correct, "correctly classified among", n_val)
    accuracy = (n_correct * 100) / n_val
    print("Accuracy:", accuracy)
    return accuracy


def make_pairs_toy(x, y):
    """
    creates all possible pairs for every sample
    """
    num_samples = len(x)

    pairs = []
    labels = []

    for i in range(num_samples):
        for j in range(i, num_samples):
            pairs.append([x[i], x[j]])
            if i == j:
                labels.append([1])
            else:
                labels.append([0])

    return (np.array(pairs), np.array(labels))


def make_pairs_full(x, y):
    """
    creates all possible pairs for every sample
    """
    num_samples = len(x)

    pairs = []
    labels = []

    for i in range(num_samples):
        for j in range(i, num_samples):
            pairs.append([x[i], x[j]])
            if y[i] == y[j]:
                labels.append([1])
            else:
                labels.append([0])

    return (np.array(pairs), np.array(labels))


def make_pairs(x, y):
    """
    creates one positive and one negative pair for every sample
    """
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairs = []
    labels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    num_classes = len(np.unique(y))
    indices = [np.where(y == i)[0] for i in np.unique(y)]
    # loop over all images
    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        # randomly pick an image that belongs to the *same* class label

        idx2 = np.random.choice(indices[label1])
        x2 = x[idx2]

        pairs.append([x1, x2])
        labels.append([1])


        # add a non-matching example
        label2 = np.random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = np.random.randint(0, num_classes - 1)

        idx2 = np.random.choice(indices[label2])
        x2 = x[idx2]
        pairs.append([x1, x2])
        labels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairs), np.array(labels))


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    feats_a, feats_b = vectors
    # compute the sum of squared distances between the vectors
    sum_squared = K.sum(K.square(feats_a - feats_b), axis=1,
                        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def contrastive_loss(y_true, y_pred, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y_true = tf.cast(y_true, y_pred.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squared_preds = K.square(y_pred)
    squared_margin = K.square(K.maximum(margin - y_pred, 0))
    loss = K.mean((1 - y_true) * squared_preds + y_true * squared_margin)
    # return the computed contrastive loss to the calling function
    return loss


def plot_training(h, plot_path):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    # plt.plot(h.history["accuracy"], label="train_acc")
    # plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)


