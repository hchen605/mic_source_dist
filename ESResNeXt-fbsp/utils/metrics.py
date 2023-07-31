from ignite.metrics.metric import Metric
import torch

import os
import numpy as np

class MAE(Metric):
    required_output_keys = ("y_pred", "y")
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = torch.FloatTensor(labels)
    def reset(self):
        self.SAE = 0
        self.n = 0
    def update(self, output):
        y_pred, y = output
        y = y.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()

        y = self.labels[y]
        y_pred = self.labels[y_pred]

        self.SAE += torch.abs(y - y_pred).sum()
        self.n += y.shape[0]
    def compute(self):
        MAE = self.SAE / self.n
        return MAE.detach().cpu()

class QuantizedMAE(Metric):
    required_output_keys = ("y_pred", "y")
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = torch.FloatTensor(labels).unsqueeze(0)
    def reset(self):
        self.n = 0
        self.SAE = 0
    def update(self, output):
        y_pred, y = output
        self.n += y.shape[0]
        y = y.detach().cpu()
        y_pred = y_pred.unsqueeze(1).detach().cpu()

        AE = torch.abs(y_pred - self.labels)
        y_pred = self.labels[0, AE.argmin(1)]
        
        self.SAE += torch.abs(y_pred - y).sum()
    def compute(self):
        return self.SAE / self.n

class Output_vectors(Metric):
    required_output_keys = ("y_pred", "y")
    def __init__(self, output_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    def reset(self):
        self.batch = 0
    def compute(self):
        return 0
    def update(self, output):
        y_pred, y = output
        with open(os.path.join(self.output_dir, f"batch_{self.batch}_y.npy"), "wb") as f:
            np.save(f, y.detach().cpu().numpy())
        with open(os.path.join(self.output_dir, f"batch_{self.batch}_y_pred.npy"), "wb") as f:
            np.save(f, y_pred.detach().cpu().numpy())
        self.batch += 1


