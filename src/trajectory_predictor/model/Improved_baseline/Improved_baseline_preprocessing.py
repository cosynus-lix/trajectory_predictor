from ast import Raise
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from trajectory_predictor.dataset.Dataset import Dataset
import numpy as np




class Improved_Baselinepreprocessing():

    def __init__(self,dataset,past =30,future = 1):
        super().__init__()
        
        data_not_scaled = dataset.to_np()
        self.data_not_scaled = data_not_scaled
        ##Scale data
        self.mean_Deltas = np.mean(data_not_scaled[:,0])
        self.std_Deltas = np.std(data_not_scaled[:,0])
        self.mean_delta = np.mean(data_not_scaled[:,1])
        self.std_delta = np.std(data_not_scaled[:,1])
        self.mean_curvature = np.mean(data_not_scaled[:,2])
        self.std_curvature = np.std(data_not_scaled[:,2])

        c0 = (data_not_scaled[:,0]-self.mean_Deltas)/self.std_Deltas
        c1 = (data_not_scaled[:,1]-self.mean_delta)/self.std_delta
        c2 = (data_not_scaled[:,2]-self.mean_curvature)/self.std_curvature

        data = np.column_stack((c0,c1,c2))

        #attributes
        self.Deltas = c0
        self.delta = c1
        self.track = c2
        
        ## Creating x and y array

        X = np.empty((len(data)-past,past,3))
        y = np.empty((len(data)-past,future,2))

        for i in range (len(data)-past-future):
            for j in range (past):
                X[i,j,:]=data[i+j]

        for i in range (len(data)-past-future):
            for j in range(future):
                y[i,j,:]=data[i+past+j][0:2]

        X = X.reshape(len(X),3*past)
        #print(X.shape)
        1#zeros = np.zeros((X.shape[0],1))
        #X = np.append(X,zeros,axis = 1)
        #print(X.shape)
        #print(X[3])
        #for i in range(X.shape[0]-1):
            #on ajoute l’information de curvature du temps t+1 pour que le réseau apprenne en connaissant la piste
            #print(X[i].shape)
            #print('i plus 1')
            #print(X[i+1])
            #print(X[i+1][X.shape[1]-1])
            #X[i][X.shape[1]-1]=X[i+1][X.shape[1]-2]
        #print(X[3])
            
        #X = np.delete(X,X.shape[0]-1,0)
        y = y.reshape(len(y),2*future)
        #y= np.delete(y,y.shape[0]-1,0)
        #print(f'X vient d’être défini\n x.shape = {X.shape} \n data[3] = {data[3]}\n x[3] = {X[3]}\n y[0]={y[0]}')

        ## Creating x_test, x_train, y_test, y_train, and create tensor

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12,shuffle = False)

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        
        ##create a class for test and train dataset

        class Train_Dataset():
            def __init__(self):
                self.x_data = X_train# size [n_samples, n_features]
                self.y_data = y_train# size [n_samples, 1]
                self.n_samples = len(X_train)

            ## support indexing such that dataset[i] can be used to get i-th sample
            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]

            ##  we can call len(dataset) to return the size
            def __len__(self):
                return self.n_samples

        class Test_Dataset():
            def __init__(self):
                self.x_data = X_test# size [n_samples, n_features]
                self.y_data = y_test# size [n_samples, 1]
                self.n_samples = len(X_test)

            ## support indexing such that dataset[i] can be used to get i-th sample
            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]

            ##  we can call len(dataset) to return the size
            def __len__(self):
                return self.n_samples

        test_dataset = Test_Dataset()
        train_dataset = Train_Dataset()
        
        
        ##create dataloader
        train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)
        test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False,drop_last=False)
        #print(f'train_Dataset[0] : {train_dataset[0]}\n train dataset[1] : {train_dataset[1]}')
        #print(f'test_Dataset[0] : {test_dataset[0]}\n test_dataset[1] : {test_dataset[1]}')

    
        ##create a class for all values

        class Dataset():
            def __init__(self):
                self.x_data = X
                self.y_data = y
                self.n_samples = len(X)

            ## support indexing such that dataset[i] can be used to get i-th sample
            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]

            ##  we can call len(dataset) to return the size
            def __len__(self):
                return self.n_samples

        dataset = Dataset()

        #print(f'Dataset[0] : {dataset[0]}\n dataset[1] : {dataset[1]}')

        ##create dataloader
        dataloader = DataLoader(dataset,batch_size=64,shuffle=False,drop_last=True)


        ## Attributes

        self.past = past
        self.X = X
        self.y_scaled = y
        self.y = np.column_stack((data_not_scaled[:,0],data_not_scaled[:,1]))
        #print(f'Scalered X[3] = {X[3]}')
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.ds = dataset
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.loader = dataloader

        #print(f'X[:,0] inversé = {scaler0.inverse_transform(X[:,0].reshape(-1,1))}\n X[0] non traité = {data_not_scaled[:,0]}\n Y[:,0] inversé = {scaler0.inverse_transform(y[:,0].reshape(-1,1))}')
        

    def create_loader(self,point = 30,init = False,previous_pred = [0,0],curvature=0):
        if (point<self.past):
            print(f'point must be superior to {self.past}')
            raise ValueError

        if init:
            #on récupère les données de trajectoire
            x = self.X[point - self.past]
            #on ajoute le y ref, avec Delta progress et delta
            y = self.y[point - self.past]
            #print(f'valeur des Deltas au point 30 = {self.Deltas[30]} \n valeur des delta au  point 30 = {self.delta[30]}')
            #print(x.shape)
            #print(y)
        else:
            x = self.X[point - self.past]
            x = x[3:]
            x = np.concatenate((x,previous_pred))
            x = np.append(x,curvature[1])
            #print(f'valeur des Deltas au point 30 = {self.Deltas[30]} \n valeur des delta au  point 30 = {self.delta[30]}')
            #print(f'valeur de la piste au point 30 {self.track[30]}\n valeur de la piste au point 31 {self.track[31]}')
            #print(x.shape)
        return x