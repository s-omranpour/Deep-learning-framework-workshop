import torch
from torch import nn
from sklearn.metrics import accuracy_score

def save(model, path, name):
    torch.save(self.state_dict(), path + str(name) + '.pkl')

def load(model, path, name):
    self.load_state_dict(torch.load(path + str(name) + '.pkl'))

def train(model, t_loader, v_loader, criterion, optimizer, epochs=100, verbose=10, chechpoint=200):
    for epoch in tqdm(range(epochs)):
        train_loss = []
        train_acc = []
        # TRAINING LOOP
        i = 0
        for train_batch in t_loader:
            x, y = train_batch

            logits = model(x.cuda())
            loss = criterion(logits, y.cuda())
            train_loss.append(loss.item())
            train_acc = accuracy_score(torch.argmax(y,dim=1).numpy(), torch.argmax(logits,dim=1).cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('\r',i,end='')
            i+=1

        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        log = '\r{} train_loss:{} train_acc:{}\n'.format(epoch+1, train_loss, train_acc)
        log += evaluate(model, v_loader, criterion)

        if (epoch+1) % verbose == 0:
            print(log)

        if (epoch+1) % chechpoint == 0:
            save(model,'weights/', 'cls_'+str(epoch+1))

def evaluate(model, loader, criterion):
    # VALIDATION LOOP
    with torch.no_grad():
        val_loss = []
        val_acc = []
        for val_batch in loader:
            x, y = val_batch
            logits = model(x.cuda())
            val_loss.append(criterion(logits, y.cuda()).item())
            val_acc = accuracy_score(torch.argmax(y,dim=1).numpy(), torch.argmax(logits,dim=1).cpu().numpy())

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