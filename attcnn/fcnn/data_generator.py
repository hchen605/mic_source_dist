import tensorflow as tf
import numpy as np

class MicRIRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_train, y_train, x_rir, y_rir, x_timit, nrir=6000, batch_size=32):
        self.x_train = x_train
        self.y_train = y_train

        self.clean = x_timit
        self.x_rir = x_rir
        self.y_rir = y_rir
        self.nrir = nrir

        self.batch_size = batch_size

        self.indices = np.arange(len(self.y_train) + self.nrir)
        self.on_epoch_end()
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    def __len__(self):
        return (len(self.y_train) + self.nrir) // self.batch_size
    def __getitem__(self, idx):
        indices = np.arange(idx * self.batch_size, (idx+1) * self.batch_size)
        results = [self.getwave(idx) for idx in indices]
        audio = [i[0] for i in results]
        target = [i[1] for i in results]
        audio = np.array(audio)
        target = np.array(target)
        return audio, target
    def getwave(self, idx):
        idx = self.indices[idx]
        if idx >= len(self.y_train):
            rir_idx = np.random.randint(0, len(self.y_rir))
            rir, target = self.x_rir[rir_idx], self.y_rir[rir_idx:rir_idx+1]
            audio = self.clean[np.random.randint(0, len(self.clean))]
            nfft = len(rir) + len(audio) - 1
            orig_length = len(audio)
            audio = np.fft.fft(audio, n=nfft)
            rir = np.fft.fft(rir, n=nfft)
            audio = np.fft.ifft(audio * rir).real
            audio = np.expand_dims(audio[:orig_length], axis=0)
            return audio, target
        else:
            return self.x_train[idx:idx+1], self.y_train[idx:idx+1]