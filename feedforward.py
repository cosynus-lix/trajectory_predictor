#%%
from random import random
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from trajectory_predictor.dataset.Dataset import Dataset as ds
## author @pierrepeaucelle

#%%
##   0) Prepare data
##

##load data
dataset = ds()
dataset.load_data('../centerline/map0.csv', '../runs/run0/spline.npy', '../runs/run0/history.npy')
data = dataset.to_np()

##creating a pd dataframe

index_values = np.linspace(1,len(data),len(data)).astype(int)
column_values = ['Ds','delta','K']
df = pd.DataFrame(data=data,index = index_values, columns=column_values)
dfy = df.drop(['K'],axis=1)

print(df.head(10))
print(dfy.head(10))

##Parameters for past & future

past=20
future=1

## Creating x and y array
X = np.empty((len(data)-past,past,3))
y = np.empty((len(data)-past,future,2))

for i in range (len(data)-past-future):
    for j in range (past):
        X[i,j,:]=df.iloc[i+j]

for i in range (len(data)-past-future):
    for j in range(future):
        y[i,j,:]=dfy.iloc[i+past+j]

X = X.reshape(len(X),3*past)
y = y.reshape(len(y),2*future)
#%%
## check print

print(f'data[3] = {data[3]}, x[1] = {X[1]}, y[0]={y[0]}')

# %%
## Creating x_test, x_train, y_test, y_train, and create tensor

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)

X_train = torch.from_numpy(X_train.astype(np.float64))
X_test = torch.from_numpy(X_test.astype(np.float64))

#%%
##  Standadizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform (y_test)
print(f'X_train.shape = {X_train.shape} \n X_test.shape = {X_test.shape}')
y_train = torch.from_numpy(y_train.astype(np.float64))
y_test = torch.from_numpy(y_test.astype(np.float64))
print(f'y_train.shape = {y_train.shape} \n y_test.shape = {y_test.shape}')
# %%
##  1)  creating train_dataset
##

class Train_Dataset(Dataset):
    def __init__(self):
        self.x_data = X_train
        self.y_data = y_train
        self.past = past
        self.future = future
        self.n_samples = len(X_train)

    
    ## support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    ## we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

#%%
##  1.bis)  Creating test_dataset

class Test_Dataset(Dataset):
    def __init__(self):
        self.x_data = X_test# size [n_samples, n_features]
        self.y_data = y_test# size [n_samples, 1]
        self.past = past
        self.future = future
        self.n_samples = len(X_test)

    ## support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    ##  we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
#%%
#check print Dataset

test_dataset = Test_Dataset()
train_dataset = Train_Dataset()

firstt_data = train_dataset[0]
first_data = test_dataset[0]

print(f'first train data {firstt_data} \n first test data {first_data} \n')
#%%
##  2) Create Dataloader
##

train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False,drop_last=False)

# %%
##  3)  Model definition

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim).double()

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim).double()

        # Input dimension
        self.input_dim = input_dim

    def forward(self, x):
        x = x.reshape(-1,self.input_dim)
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out
#%%
##  4)  Model implementation
##
input_dim = 3*past
output_dim = 2*future
hidden_dim = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim).to(device)


# %%
## 5) Instantiate loss and optimizer
##

# mseLoss

loss_fn = nn.MSELoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# %%
##  6) Defining train and test loop
##

##  Train loop

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_list=[0]
    i = 0
    for batch, (X, y) in enumerate(dataloader):
        i = i+1
        # Compute prediction and loss
        pred = model(X)
        #print(f'n={i},pred[0] = {pred[0]}, y[0]={y[0]}')
        loss = loss_fn(pred, y)
        lossn = loss.detach().numpy()
        loss_list = np.concatenate((loss_list,lossn),axis=None)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_list

## Test Loop

def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss = 0
    loss_list=[0]
    deltapred = [0]
    Deltaspred = [0]

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            nppred = pred.detach().numpy()
            delta_pred = np.transpose(nppred[:,1])
            Deltas_pred = np.transpose(nppred[:,0])
            lossn = loss_fn(pred,y).detach().numpy()
            test_loss += loss_fn(pred, y).item()
            loss_list = np.concatenate((loss_list,lossn),axis = None)
            deltapred = np.concatenate((deltapred,delta_pred),axis = None)
            Deltaspred = np.concatenate((Deltaspred,Deltas_pred),axis = None)
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return loss_list,deltapred,Deltaspred


# %%
##  7)  Running Code
##

train_loss=[0]
test_loss=[0]
epochs = 10

for t in range(epochs):
    deltatest = [0]
    Deltastest = [0]
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss_list = train_loop(train_dataloader, model, loss_fn, optimizer)
    #print(loss_list)
    train_loss = np.concatenate((train_loss,train_loss_list),axis = None)
    test_loss_list, delta_test, Deltas_test = test_loop(test_dataloader, model, loss_fn)
    test_loss = np.concatenate((test_loss,test_loss_list),axis = None)
    deltatest = np.concatenate((deltatest,delta_test),axis = None)
    Deltastest = np.concatenate((Deltastest,Deltas_test),axis = None)
print("Done!")
#%%
## Post treating results

deltatest = deltatest[deltatest !=0]
Deltastest = Deltastest[Deltastest !=0]
train_loss = train_loss[train_loss !=0]
test_loss = test_loss[test_loss !=0]
#removing first 2% for the print
train_remove = int(len(train_loss)*0.02)
test_remove = int(len(train_loss)*0.02)
#loss = loss[remove:]
#%%
##  8)  Plot loss results
##

plt.figure()
plt.title('Loss function for train and test',loc='center')
plt.subplot(3,2,1)
plt.plot(train_loss)
plt.title('train loss')
plt.subplot(3,2,2)
plt.plot(test_loss)
plt.title('test loss')
plt.subplot(3,1,2)
plt.plot(y_test[:,1].numpy(),color = 'blue', label ='delta true')
plt.plot(deltatest, color = 'orange',label='delta predicted')
plt.legend(loc = 'best')
plt.title('delta : prediction vs reality')
plt.subplot(3,1,3)
plt.plot(Deltastest, color = 'orange',label='Delta s predicted')
plt.plot(y_test[:,0].numpy(),color = 'blue', label ='Delta s true')
plt.legend(loc = 'best')
plt.title('Delta s : prediction vs reality')
plt.show()
# %%

