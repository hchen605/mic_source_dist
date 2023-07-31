import os
import sys
import warnings
import multiprocessing as mp

import msgpack_numpy as m

import numpy as np
import pandas as pd
import sklearn.model_selection as skms

import tqdm
import librosa

import torch.utils.data as td

from utils import transforms

from typing import Tuple
from typing import Optional
from typing import List

import soundfile as sf

from scipy import signal

from ignite_trainer import _utils

from tqdm.contrib.concurrent import process_map
import sys
import subprocess

m.patch()

class MicClassification(td.Dataset):
    def __init__(self,
                train_csv: str,
                dev_csv: str,
                label_type: str,
                train: bool = True,
                root: str = "",
                sample_rate: int = 44100,
                narrowband = None,
                label_set = None,
                transform=None,
                rir='',
                clean='',
                augment_num=0,
                target_transform=None,
                regression: bool = False):
        super(MicClassification, self).__init__()
        self.root = root
        self.sample_rate = sample_rate
        self.augment_num = augment_num if train else 0
        if self.augment_num > 0:
            self.load_rir(rir=rir, clean=clean)
        self.csv = train_csv if train else dev_csv
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.label_set = label_set

        self.label_to_index = dict()
        self.data = dict()
        self.load_data(regression=regression, narrowband=narrowband)
        self.indices = list(self.data.keys())
    def load_rir(self, rir: str, clean: str):
        self.rir = []
        rir_list = subprocess.check_output(['find', rir, '-name', '*.wav']).decode('utf-8').strip('\n').split('\n')
        items_to_load = [(j, i, self.sample_rate) for j, i in enumerate(rir_list)]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunksize = int(np.ceil(len(items_to_load) / pool._processes)) or 1
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (data={rir})')
            for idx, sample_rate, wav in pool.starmap(
                    func=self._load_worker,
                    iterable=items_to_load,
                    chunksize=chunksize
            ):
                rir_info = os.path.basename(rir_list[idx]).split('.')[0]
                dist = float(rir_info.split('_')[0].strip('m'))
                self.rir.append((wav, dist))
        
        self.clean = []
        clean_list = subprocess.check_output(['find', clean, '-name', '*.WAV']).decode('utf-8').strip('\n').split('\n')
        items_to_load = [(j, i, self.sample_rate) for j, i in enumerate(clean_list)]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunksize = int(np.ceil(len(items_to_load) / pool._processes)) or 1
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (data={clean})')
            for idx, sample_rate, wav in pool.starmap(
                    func=self._load_worker,
                    iterable=items_to_load,
                    chunksize=chunksize
            ):
                self.clean.append(wav)

    @staticmethod
    def _load_worker(idx: int, filename: str, sample_rate: Optional[int] = None, narrowband=None) -> Tuple[int, int, np.ndarray]:
        if not os.path.exists(filename):
            print("Warning: file %s does not exist, skipped..." % filename, file=sys.stderr)
            return -1,-1,np.zeros(0)
        try:
            wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)
            if narrowband is not None:
                if narrowband[0] == 0:
                    b, a = signal.butter(8, narrowband[1], 'lowpass', fs=sample_rate)
                else:
                    b, a = signal.butter(8, narrowband, 'bandpass', fs=sample_rate)
                wav = signal.filtfilt(b, a, wav)
        except ValueError:
            print("Warning: Failed to load file %s, skipping..." % filename, file=sys.stderr)
            return -1,-1,np.zeros(0)
        if wav.ndim == 1: wav = wav[:, np.newaxis]
        wav = wav.T * 32768
        return idx, sample_rate, wav.astype(np.float32)
    def load_data(self, regression=False, narrowband=None):
        if self.csv == "":
            return
        meta = pd.read_csv(self.csv, sep='\t')
        assert self.label_type in meta, \
            "Error: label_type %s isn't found. Need to be in %s" % (self.label_type, list(meta))
        label_set = self.label_set if self.label_set is not None else \
                sorted(set(meta[self.label_type].values))
        self.label_to_index = {label:idx for idx, label in enumerate(label_set)}

        label_count = {label:0 for label in label_set}
        items_to_load = []
        for idx, row in meta.iterrows():
            label = row[self.label_type]
            label_count[label] += 1
            items_to_load.append((idx, os.path.join(self.root, row['filename']),
                     self.sample_rate, narrowband))
        
        warnings.filterwarnings("ignore")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunksize = int(np.ceil(len(items_to_load) / pool._processes)) or 1
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (data={self.csv})')
            for idx, sample_rate, wav in pool.starmap(
                    func=self._load_worker,
                    iterable=items_to_load,
                    chunksize=chunksize
            ):
                if len(wav) == 0: continue
                
                row = meta.loc[idx]
                if regression:
                    target = float(str(row[self.label_type]).strip('m'))
                else:
                    target = self.label_to_index[row[self.label_type]]

                self.data[idx] = {
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': target
                }
    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        if not (0 <= index < len(self)):
            raise IndexError
        if index < len(self.data):
            audio: np.ndarray = self.data[self.indices[index]]['audio']
            target = self.data[self.indices[index]]['target']
        else:
            rir, target = self.rir[np.random.randint(0, len(self.rir))]
            audio = self.clean[np.random.randint(0, len(self.clean))]
            rir = rir[0]; audio = audio[0]
            nfft = len(rir) + len(audio) - 1
            audio = np.fft.fft(audio, n=nfft)
            rir = np.fft.fft(rir, n=nfft)
            audio = np.fft.ifft(audio * rir).real
            audio = np.expand_dims(audio, axis=0)

        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return audio, target

    def __len__(self) -> int:
        return len(self.data) + self.augment_num

class ConcatDataset(td.Dataset):
    def __init__(self, datasets, train=True, **kwargs) -> None:
        self.datasets = []
        for dataset in datasets:
            Dataset: Type = _utils.load_class(dataset['class'])
            self.datasets.append(Dataset(**{**dataset['args'], "train": train, **kwargs}))
        self.datasets_indices = np.cumsum([len(i) for i in self.datasets])
        self.train = train
    def idx_mapper(self, idx):
        dataset_idx = np.argmax(self.datasets_indices > idx)
        if dataset_idx == 0:
            item_idx = idx
        else:
            item_idx = idx - self.datasets_indices[dataset_idx - 1]
        return dataset_idx, item_idx
    def __getitem__(self, idx):
        dataset_idx, item_idx = self.idx_mapper(idx)
        return self.datasets[dataset_idx][item_idx]
    def __len__(self):
        return self.datasets_indices[-1]

class ASVspoof(td.Dataset):
    def __init__(self,
                root: str,
                dev_trl: str,
                label_set: List[str],
                sample_rate: Optional[int] = 44100,
                train=False, **kwargs) -> None:
        assert not train
        self.idx2label = label_set
        self.label2idx = {label: idx for idx, label in enumerate(self.idx2label)}

        with open(dev_trl, 'r') as f:
            lines = f.readlines()

        targets = []
        flacfiles = []
        for line in lines:
            row = line.strip('\n').split(" ")
            if row[-1] == 'bonafide':
                label = 'bonafide'
            elif row[-1] == 'spoof':
                label = row[-2]
            else:
                raise NotImplementedError(row[-1])
            flacfiles.append((os.path.join(root, f"{row[1]}.flac"), sample_rate))
            targets.append(self.label2idx[label])
        with _utils.tqdm_stdout() as orig_stdout:
            chunksize = int(np.ceil(len(flacfiles) / mp.cpu_count())) or 1
            tqdm.tqdm.write(f'data={dev_trl}')
            audios = process_map(
                    self._load_flacfile,
                    flacfiles,
                    file=orig_stdout,
                    chunksize=chunksize,
                    ncols=150,
                    ascii=True,
                    desc=f'Loading {self.__class__.__name__}')
        max_length = np.max([len(i) for i in audios])
        audios = [np.pad(i, (0, max_length - len(i)), mode='wrap') for i in audios]
        self.data = [{
            "audio": audio,
            "target": target
        } for audio, target in zip(audios, targets)]
    @staticmethod
    def _load_flacfile(mapper) -> np.ndarray:
        flacfile, sample_rate = mapper
        data, sr = sf.read(flacfile)
        if sr != sample_rate:
            data = librosa.resample(data, sr, sample_rate)
        return data
    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        if not (0 <= index < len(self)):
            raise IndexError
        audio: np.ndarray = self.data[index]['audio']
        target = self.data[index]['target']
        return audio, target
    def __len__(self) -> int:
        return len(self.data)
