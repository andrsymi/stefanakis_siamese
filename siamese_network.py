# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import  Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def build_siamese_model(input_shape, embedding_dim=12):
    # specify the inputs for the feature extractor network
    inputs = Input(input_shape)
    # x = Conv1D(60, kernel_size=10, activation='relu')(inputs)
    l1 = Dense(int((2/3) * input_shape), activation='relu')(inputs)
    l2 = Dense(int((1/2) * input_shape), activation='relu')(l1)
    l3 = Dense(int((1/3) * input_shape), activation='relu')(l2)
    l4 = Dense(int(((1/3) * input_shape)/2), activation='relu')(l3)
    outputs = Dense(embedding_dim, activation='relu')(l4)

    model = Model(inputs, outputs)

    return model
