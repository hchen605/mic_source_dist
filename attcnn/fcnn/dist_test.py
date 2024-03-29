import os
import sys
sys.path.append("..")
import argparse
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras import backend as K

from utils import *
from funcs import *

from ts_dataloader import *
from models.small_fcnn_att import model_fcnn

from tensorflow.compat.v1 import ConfigProto, InteractiveSession, Session
import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", type=str, default=os.path.join("weight", "AttCNN"))
parser.add_argument("--test_csv", type=str, default='../../data/phase3_all_seen_test.csv')
parser.add_argument("--root", type=str, default='/home/koredata/hsinhung/speech')
parser.add_argument("--narrowband", type=int, nargs='+', default=None)

args = parser.parse_args()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sess = Session(graph=tensorflow.compat.v1.get_default_graph(), config=config)
K.set_session(sess)

x_test, y_test = load_data(args.test_csv, root=args.root, narrowband=args.narrowband)
num_freq_bin = 128
num_audio_channels = 1

print ("=== Number of test data: {}".format(len(y_test)))

# -
#length = x_train[0].shape[0]

# Model
model = model_fcnn(input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
weights_path = os.path.join(args.savedir, 'best.hdf5')
model.load_weights(weights_path)

model.compile(loss='mean_absolute_error')
model.summary()

score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score)

labels = np.array([[0.2, 1, 3, 5, 7, 9]])
y_pred = model.predict(x_test)
AE = np.abs(y_pred - labels)
y_pred = labels[0, np.argmin(AE, axis=1)]
MAE = np.abs(y_pred - y_test).mean()
print('Quantized MAE:', MAE)


