from siamese_network import build_siamese_model
import config
import utils
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob


DATA_PATH = "/Users/nikos/Documents/FORTH/guitar_data_stefanakis/data_fft_1024/"


def main():
    # Load training set
    df = pd.read_csv("data.csv")

    # Shuffle and prepare training set
    df = df.sample(frac=1).reset_index(drop=True)
    x_train = df.iloc[:, 1:]
    y_train = df.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, shuffle=True, stratify=y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True, stratify=y_train)

    x_train = x_train.values
    x_val = x_val.values
    x_test = x_test.values

    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = y_train.values.astype(int)
    y_val = y_val.values.astype(int)
    y_test = y_test.values.astype(int)

    (pairs_train, labels_train) = utils.make_pairs(x_train, y_train)
    (pairs_val, labels_val) = utils.make_pairs(x_val, y_val)
    (pairs_test, labels_test) = utils.make_pairs(x_test, y_test)

    print("\nSamples' balance:")
    print("positives: ", len(np.argwhere(labels_train == 1)[:, 0]) / len(labels_train))
    print("negatives: ", len(np.argwhere(labels_train == 0)[:, 0]) / len(labels_train), "\n")

    # configure the siamese network
    sample_a = Input(shape=config.SAMPLE_SHAPE)
    sample_b = Input(shape=config.SAMPLE_SHAPE)
    feature_extractor = build_siamese_model(config.SAMPLE_SHAPE)
    feats_a = feature_extractor(sample_a)
    feats_b = feature_extractor(sample_b)

    # finally, construct the siamese network
    distance = Lambda(utils.euclidean_distance)([feats_a, feats_b])

    outputs = Dense(1, activation="sigmoid")(distance)

    # contrastive
    model = Model(inputs=[sample_a, sample_b], outputs=outputs)

    # compile the model
    model.compile(loss=utils.contrastive_loss, optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        [pairs_train[:, 0], pairs_train[:, 1]], labels_train[:],
        validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val[:]),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS)

    # Evaluate model

    results = model.evaluate([pairs_test[:, 0], pairs_test[:, 1]], labels_test[:], config.BATCH_SIZE)
    print("Siamese: test loss, test acc:", results)

    preds = model.predict([pairs_test[:, 0], pairs_test[:, 1]])

    utils.nway_one_shot(model, x=pairs_test, y=labels_test[:,0], n_way=5, n_val=1000)


if __name__ == "__main__":
    main()
