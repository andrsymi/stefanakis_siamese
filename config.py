# import the necessary packages
import os
# specify the shape of the inputs for our network
SAMPLE_SHAPE = 127
# specify the batch size and number of epochs
BATCH_SIZE = 128
EPOCHS = 10

# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot

# Binary Cross-entropy
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
FINAL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "final_plot.png"])

# Contrastive
# MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_siamese_model"])
# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_plot.png"])
