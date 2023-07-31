import os
import sys
sys.path.append("..")
import argparse
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from keras import backend as K

from utils import *
from funcs import *

from ts_dataloader import *
from models.small_fcnn_att import model_fcnn

from tensorflow.compat.v1 import ConfigProto, InteractiveSession, Session
import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=int, default=100, help="number of epochs")
parser.add_argument("--savedir", type=str, default=os.path.join("weight", "AttCNN"))
parser.add_argument("--train_csv", type=str, default='../../data/phase3_all_seen_train.csv')
parser.add_argument("--dev_csv", type=str, default='../../data/phase3_all_seen_val.csv')
parser.add_argument("--narrowband", type=int, nargs='+', default=None)
parser.add_argument("--root", type=str, default='/home/koredata/hsinhung/speech')
args = parser.parse_args()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.python.keras import backend as K
config = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=config)
K.set_session(sess)

x_train, y_train = load_data(args.train_csv, root=args.root, narrowband=args.narrowband, label_type=np.str)
x_dev, y_dev = load_data(args.dev_csv, root=args.root, narrowband=args.narrowband, label_type=np.str)

labels = sorted(list(set(y_train)))
label2idx = {label: idx for idx, label in enumerate(labels)}
y_train = np.array([label2idx[label] for label in y_train])
y_dev = np.array([label2idx[label] for label in y_dev])
y_train = np.eye(len(labels))[y_train]
y_dev = np.eye(len(labels))[y_dev]

print ("=== Number of training data: {}".format(len(y_train)))

# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps


# Model
model = model_fcnn(input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0, nclass=len(labels))


model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))


model.summary()

# Checkpoints
os.makedirs(args.savedir, exist_ok=True)

save_path = os.path.join(args.savedir, 'best.hdf5')
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [checkpoint]

# Training
exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_dev, y_dev), callbacks=callbacks)

print("=== Best Val. Loss: ", max(exp_history.history['val_loss']), " At Epoch of ", np.argmax(exp_history.history['val_loss'])+1)
