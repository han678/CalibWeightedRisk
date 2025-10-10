import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TemperatureScale(nn.Module):
    def __init__(self, temp=1.5, criterion=nn.CrossEntropyLoss(), maxiter=50):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1) * temp)
        self.criterion = criterion
        self.maxiter = maxiter

    def temp_scale(self, logits):
        return logits / self.temp.expand_as(logits).to(logits.device)

    def predict_prob(self, logits):
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).cuda()
        return F.softmax(self.temp_scale(logits), dim=1)

    def _optimize_temp(self, logits, labels):
        optimizer = optim.LBFGS([self.temp], lr=0.01, max_iter=self.maxiter)
        criterion = self.criterion.cuda()
        def closure():
            loss = criterion(self.temp_scale(logits), labels)
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)
        print('Searched Temperature:', round(self.temp.item(), 4))

    def set_temp_from_logits(self, logits, labels):
        if isinstance(logits, np.ndarray): logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray): labels = torch.from_numpy(labels)
        self._optimize_temp(logits.cuda(), labels.cuda())

    def set_temp_from_loader(self, loader, model):
        logits, labels = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                logits.append(x.cuda())
                labels.append(y.cuda())
        self._optimize_temp(torch.cat(logits), torch.cat(labels))

class ModelWithTemperature(nn.Module):
    def __init__(self, model, temp=1.5, maxiter=50):
        super().__init__()
        self.model = model
        self.model.eval()
        self.temp = nn.Parameter(torch.ones(1) * temp)
        self.maxiter = maxiter

    def forward(self, input):
        with torch.no_grad():
            return self.temp_scale(self.model(input))

    def temp_scale(self, logits):
        temp = self.temp.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(logits.device)
        return logits / temp

    def set_temp(self, val_loader, criterion=nn.CrossEntropyLoss()):
        logits, labels = [], []
        criterion = criterion.cuda()
        with torch.no_grad():
            for input, label in val_loader:
                input = input.cuda()
                logits.append(self.model(input))
                labels.append(label)
            logits, labels = torch.cat(logits, 0).cuda(), torch.cat(labels, 0).cuda()
        optimizer = optim.LBFGS([self.temp], lr=0.01, max_iter=self.maxiter)
        def eval():
            loss = criterion(self.temp_scale(logits), labels)
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(eval)
        print('\nSearched Temperature on Validation Data: ', round(self.temp.item(), 4))