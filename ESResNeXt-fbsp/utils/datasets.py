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


m.patch()

class MicClassification(td.Dataset):
    def __init__(self,
                train_csv: str,
                dev_csv: str,
                label_type: str,
                train: bool = True,
                root: str = "",
                sample_rate: int = 44100,
                transform=None,
                target_transform=None,
                regression: bool = False,
                limit=np.inf):
        super(MicClassification, self).__init__()
        self.root = root
        self.csv = train_csv if train else dev_csv
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.sample_rate = sample_rate
        self.limit = limit if train else np.inf

        self.label_to_index = dict()
        self.data = dict()
        self.load_data(regression=regression)
        self.indices = list(self.data.keys())
    @staticmethod
    def _load_worker(idx: int, filename: str, sample_rate: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
        if not os.path.exists(filename):
            print("Warning: file %s does not exist, skipped..." % filename, file=sys.stderr)
            return -1,-1,np.zeros(0)
        try:
            wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)
        except ValueError:
            print("Warning: Failed to load file %s, skipping..." % filename, file=sys.stderr)
            return -1,-1,np.zeros(0)
        if wav.ndim == 1: wav = wav[:, np.newaxis]
        wav = wav.T * 32768
        return idx, sample_rate, wav.astype(np.float32)
    def load_data(self, regression=False):
        meta = pd.read_csv(self.csv, sep='\t')
        assert self.label_type in meta, \
            "Error: label_type %s isn't found. Need to be in %s" % (self.label_type, list(meta))
        label_set = sorted(set(meta[self.label_type].values))
        self.label_to_index = {label:idx for idx, label in enumerate(label_set)}

        label_count = {label:0 for label in label_set}
        items_to_load = []
        for idx, row in meta.iterrows():
            label = row[self.label_type]
            if label_count[label] >= self.limit: continue
            label_count[label] += 1
            items_to_load.append((idx, os.path.join(self.root, row['filename']),
                     self.sample_rate))
        
        warnings.filterwarnings("ignore")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunksize = int(np.ceil(len(items_to_load) / pool._processes)) or 1
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (train={self.train})')
            for idx, sample_rate, wav in pool.starmap(
                    func=self._load_worker,
                    iterable=items_to_load,
                    chunksize=chunksize
            ):
                if len(wav) == 0: continue
                
                row = meta.loc[idx]
                if regression:
                    target = float(row[self.label_type].strip('m'))
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
        audio: np.ndarray = self.data[self.indices[index]]['audio']
        target = self.data[self.indices[index]]['target']
        
        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return audio, target
    def __len__(self) -> int:
        return len(self.data)
