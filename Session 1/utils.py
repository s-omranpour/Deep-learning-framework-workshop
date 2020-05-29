import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import numpy as np

def save(model, path, name):
    torch.save(self.state_dict(), path + str(name) + '.pkl')

def load(model, path, name):
    self.load_state_dict(torch.load(path + str(name) + '.pkl'))

def train(model, t_loader, v_loader, criterion, optimizer, device='cuda:0', epochs=100, verbose=10, chechpoint=200):
    for epoch in tqdm(range(epochs)):
        train_loss = []
        train_acc = []
        
        # TRAINING LOOP
        for train_batch in t_loader:
            x, y = train_batch

            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            train_loss.append(loss.item())
            train_acc = accuracy_score(y.numpy(), torch.argmax(logits,dim=1).cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        log = '\r{} train_loss:{} train_acc:{}\n'.format(epoch+1, train_loss, train_acc)
        log += evaluate(model, v_loader, criterion, device)

        if (epoch+1) % verbose == 0:
            print(log)

        if (epoch+1) % chechpoint == 0:
            save(model,'weights/', 'cls_'+str(epoch+1))

def evaluate(model, loader, criterion, device):
    # VALIDATION LOOP
    with torch.no_grad():
        val_loss = []
        val_acc = []
        for val_batch in loader:
            x, y = val_batch
            logits = model(x.to(device))
            val_loss.append(criterion(logits, y.to(device)).item())
            val_acc = accuracy_score(y.numpy(), torch.argmax(logits,dim=1).cpu().numpy())

        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)
        return 'val_loss:{} val_acc:{}\n'.format(val_loss, val_acc)
    
def get_n_params(self):
    params = []
    for param in self.parameters():
        a = 1
        for s in param.shape:
            a *= s
        params += [a]
    return sum(params)