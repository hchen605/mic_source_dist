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
parser.add_argument("--nclass", type=int, default=1, help="3class (0), 12class(1)")
parser.add_argument("--mixed", type=int, default=0, help="fixed location (0), mixed location(1)")
parser.add_argument("--limit", type=int, default=100, help="number of data")
parser.add_argument("--seed", type=int, default=0, help="data random seed")
parser.add_argument("--eps", type=int, default=30, help="number of epochs")
args = parser.parse_args()

#tensorflow.reset_default_graph()
os.environ['PYTHONHASHSEED']=str(args.seed)
tensorflow.random.set_seed(args.seed)
tensorflow.compat.v1.set_random_seed(args.seed)
random.seed(args.seed)
#tensorflow.keras.utils.set_random_seed(1)
np.random.seed(args.seed)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sess = Session(graph=tensorflow.compat.v1.get_default_graph(), config=config)
K.set_session(sess)


# +
classes_3 = ['C','D','M']
classes_dist = ['3m','4m','5m','6m','7m','8m','9m']
genders = ['full', 'female', 'male']
classes_room = ['large','medium','small']
class_m = ['mic','dimension','dist']



# -

classes_test = classes_dist

if args.mixed:
    test_csv = '../data/test_dist_multi.csv'
else:
    test_csv = '../data/test_dist.csv'


print('loading microphone data')
test = load_data(test_csv)




print ("=== Number of test data: {}".format(len(test)))


x_test, y_test_3, y_test_12 = list(zip(*test))
x_test = np.array(x_test)



if args.nclass == 0:
    y_test = y_test_3
    classes = classes_3
    experiments = '3class/'
else:
    y_test = y_test_12
    classes = classes_test
    experiments = '12class/'

cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)


# y_test = [cls2label[y] for y in y_test]
y_test = np.array([i.strip('m') for i in y_test], dtype = np.float64)

# y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps


# -
#length = x_train[0].shape[0]

# Model
model = model_fcnn(input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
weights_path = 'weight/weight_dist_limit{}/12class/best.hdf5'.format(args.limit)
model.load_weights(weights_path)


model.compile(loss='mean_absolute_error',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))


model.summary()



score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score)


os.makedirs("record", exist_ok=True)
file1 = open("record/record_dist_cls_{}_re.txt".format(args.limit), "a") 
  
# writing newline character
file1.write("\n")
file1.write(str(score))
file1.close()

'''
#confusion matrix
y_pred = model.predict(x_test)
y_test_ = np.argmax(y_test, axis=-1)
y_pred_ = np.argmax(y_pred, axis=-1)


cm = confusion_matrix(y_test_, y_pred_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Source distance prediction')
plt.show()
#plt.savefig('./music_log/cm_{}_{}.pdf'.format(mic,target))
plt.savefig('./confusion/cm_dist_multi_test_{}.pdf'.format(args.limit))
'''