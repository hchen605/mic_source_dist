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

x_test, y_test = load_data(args.test_csv, root=args.root, narrowband=args.narrowband, label_type=np.str)
num_freq_bin = 128
num_audio_channels = 1

# labels = sorted(list(set(y_test)))
labels = ['0.02', '1', '3', '5', '7', '9']
label2idx = {label: idx for idx, label in enumerate(labels)}
idx2label = {idx: label for label, idx in label2idx.items()}
y_test_float = y_test.astype(np.float32)
y_test = np.array([label2idx[label] for label in y_test])
y_test = np.eye(len(labels))[y_test]

print ("=== Number of test data: {}".format(len(y_test)))

# -
#length = x_train[0].shape[0]

# Model
model = model_fcnn(input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0, nclass=len(labels))
weights_path = os.path.join(args.savedir, 'best.hdf5')
model.load_weights(weights_path)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score)

output = model.predict(x_test)
output = np.argmax(output, axis=1)
output = [idx2label[i] for i in output]
output = np.array(output, dtype=np.float32)
MAE = np.abs(output - y_test_float).mean()
print(f'MAE: {MAE}')
