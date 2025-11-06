import torch.nn as nn
import numpy as np
import time
from FLcore.client.clientbase import Client
import sys

class clientAVG(Client):
    def __init__(self,args,id,dataset):
        super().__init__(args,id,dataset)
    
    def train(self):
        print(self.id,end=' ')
        sys.stdout.flush()

        self.model.train()

        max_local_epochs = self.local_epochs
        for step in range(max_local_epochs):
            self.optimizer.zero_grad()
            output = self.model(self.dataset.x,self.dataset.edge_index)
            loss = self.loss(output[self.dataset.train_mask],self.dataset.y[self.dataset.train_mask])
            loss.backward()
            self.optimizer.step()
            # pred = output.argmax(dim=1)  # Use the class with highest probability.
            # test_correct = pred[self.dataset.test_mask] == self.dataset.y[self.dataset.test_mask]  # Check against ground-truth labels.
            # test_acc = int(test_correct.sum()) / int(self.dataset.test_mask.sum())
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()