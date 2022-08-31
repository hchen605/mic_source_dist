
import numpy as np
import pandas as pd
import librosa
import soundfile as sound


#data_csv = '../../12class/data/test_room_unseen.csv'
data_csv = "/home/hsinhung/microphone_classification_extend/12class/data/test_dist_multi.csv"

data_df = pd.read_csv(data_csv, sep='\t')   
wavpath = data_df['filename'].tolist()
labels_3 = data_df['3_types'].to_list()
labels_16 = data_df['16_types'].to_list()
datas = zip(wavpath, labels_3, labels_16)
sample_rate = 44100

#print(wavpath)

for i in range(len(wavpath)):
    print(wavpath[i])
    wav, sample_rate = librosa.load(wavpath[i], sr=sample_rate, mono=True)
    if wav.shape[0] == 0:
        print(wavpath[i])