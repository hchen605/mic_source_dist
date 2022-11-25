import librosa
import numpy as np
import pandas as pd
import soundfile as sound

from tensorflow import keras
from tqdm.contrib.concurrent import process_map

import glob
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Pool

from scipy import signal

import os
import sys

sr = 16000 
duration = 3

# +
def _load_data(args):
    wav, narrowband = args
    if not os.path.isfile(wav):
        print(f"Warning: audio file {wav} does not exist", file=sys.stderr)
        return None
    # stereo, fs = sound.read(wav)
    stereo, fs = librosa.load(wav, sr=None, mono=True)

    if narrowband != None:
        if narrowband[0] == 0:
            b, a = signal.butter(8, narrowband[1], 'lowpass', fs=fs)
        else:
            b, a = signal.butter(8, narrowband, 'bandpass', fs=fs)
        stereo = signal.filtfilt(b, a, stereo)
    if np.isnan(stereo).any():
        print(f"The wavform contains nan, try lower butterworth order", file=sys.stderr)
        exit(1)
    if stereo.shape[0] == 0:
        return None
    stereo = stereo / np.abs(stereo).max()
    # stereo = librosa.to_mono(stereo.T)

    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)

    if stereo.shape[0] > sr*duration:
        start = (stereo.shape[0] - sr*duration) // 2
        x = stereo[start:start+sr*duration]
    else:
        x = np.pad(stereo, (0, sr*duration-stereo.shape[0]))
    
    return x

def load_data(data_csv, root, narrowband=None):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels = data_df['distance'].to_list()
    wavpath = [os.path.join(root, i) for i in wavpath]

    mapper = [(i, narrowband) for i in wavpath]
    wav = process_map(_load_data, mapper, desc=f"Loading {data_csv}")
    wav = np.array([i for i in wav if i is not None])

    labels = np.array([i.strip('m') for i in labels], dtype = np.float64)
    return wav, labels

def _load_data_rir(data):
    wav, y_3, y_16, rir = data
    #print(wav)
    stereo, fs = sound.read(wav)
    #print(stereo.shape)
    assert stereo.shape[0] > 0
    
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)
        
    #assert stereo.shape[0] > 16000
    #noise only
    #orig_shape = stereo.shape
    #assert orig_shape[0] > 16000
    #trimmed, index = librosa.effects.trim(stereo, top_db=20)
    #stereo = np.concatenate((stereo[0:index[0]], stereo[index[1]:-1]))
    if stereo.shape[0] > sr*duration:
        start = (stereo.shape[0] - sr*duration) // 2
        x = stereo[start:start+sr*duration]
    else:
        x = np.pad(stereo, (0, sr*duration-stereo.shape[0]))
    
    return x, y_3, y_16, rir

def load_data_rir(data_csv, rir):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels_3 = data_df['3_types'].to_list()
    labels_16 = data_df['16_types'].to_list()
    #rir
    with open(rir) as f:
        feats = f.readlines()

    #rir
    rir = []
    for feat in feats:
        #count += 1
        #print(f'line {count}: {feat}') 
        x = feat.split(',')
        x = map(float,x)
        x = list(x)
        #print(x)
        rir = rir + [x]
        #print(data)    

    #rir = np.array(rir)

    #rir[:,0] = rir[:,0]/rir[:,0].max()
    #rir[:,1] = rir[:,1]/rir[:,1].max()
    #rir= list(rir)

    datas = zip(wavpath, labels_3, labels_16, rir)

    with Pool(32) as p:
        return p.map(_load_data_rir, datas)

def _load_data_gsc(data):
    wav, y = data
    #print(y)
    stereo, fs = sound.read(wav)
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)
    #assert stereo.shape[0] > 16000
    if stereo.shape[0] > sr:
        start = 0
        x = stereo[start:start+sr]
    else:
        x = np.pad(stereo, (0, sr-stereo.shape[0]))
    
    #print(x)
    return x, y

def load_data_gsc(data_csv):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    label = data_df['class'].to_list()
    datas = zip(wavpath, label)

    with Pool(32) as p:
        return p.map(_load_data_gsc, datas)
# -


