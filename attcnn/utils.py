import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

def split(data, limit):
    dict_16 = dict()
    for d in data:
        wav, type_3, type_16 = d
        if type_16 not in dict_16:
            dict_16[type_16] = list()
        dict_16[type_16].append(wav)
    

    new_data = []
    for t in dict_16:
        indexes = list(range(len(dict_16[t])))
        random.shuffle(indexes)
        #random.Random(5).shuffle(indexes)
        for i in indexes[:limit]:
            new_data.append((dict_16[t][i],t[0],t))

    return new_data

def split_seed(data, limit, seed):
    dict_16 = dict()
    for d in data:
        wav, type_3, type_16 = d
        if type_16 not in dict_16:
            dict_16[type_16] = list()
        dict_16[type_16].append(wav)
    

    new_data = []
    for t in dict_16:
        indexes = list(range(len(dict_16[t])))
        random.Random(seed).shuffle(indexes)
        for i in indexes[:limit]:
            new_data.append((dict_16[t][i],t[0],t))

    return new_data

def split_rir(data, limit):
    dict_16 = dict()
    dict_rir = dict()
    for d in data:
        wav, type_3, type_16, rir = d
        if type_16 not in dict_16:
            dict_16[type_16] = list()
            dict_rir[type_16] = list()
        dict_16[type_16].append(wav)
        dict_rir[type_16].append(rir)
    
    new_data = []
    for t in dict_16:
        indexes = list(range(len(dict_16[t])))
        random.shuffle(indexes)
        #random.Random(5).shuffle(indexes)
        for i in indexes[:limit]:
            new_data.append((dict_16[t][i],t[0],t,dict_rir[t][i]))
    
    return new_data

def split_rir_seed(data, limit, seed):
    dict_16 = dict()
    dict_rir = dict()
    for d in data:
        wav, type_3, type_16, rir = d
        if type_16 not in dict_16:
            dict_16[type_16] = list()
            dict_rir[type_16] = list()
        dict_16[type_16].append(wav)
        dict_rir[type_16].append(rir)
    

    new_data = []
    for t in dict_16:
        indexes = list(range(len(dict_16[t])))
        random.Random(seed).shuffle(indexes)
        for i in indexes[:limit]:
            new_data.append((dict_16[t][i],t[0],t,dict_rir[t][i]))

    return new_data


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
    
    ax.set_title(title, fontsize='large')
    
    tick_marks = np.arange(len(classes))    
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
