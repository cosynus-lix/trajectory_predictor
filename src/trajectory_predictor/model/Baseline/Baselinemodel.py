
import os
from tkinter import Y
from ..Model import Model
from .FeedforwardNeuralNet import FeedforwardNeuralNet
import torch
import torch.nn as nn
import numpy as np
from .Baseline_preprocessing import Baselinepreprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm


class BaselineModel(Model):
    def __init__(self, past, hidden_dim, horizon):

        super().__init__()
        self.model = FeedforwardNeuralNet(past*3,hidden_dim,horizon*2)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = 0.1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) 
        self.past = past
        self.horizon = horizon
        self.hidden_dim = hidden_dim


    #train function
    def train(self,dataset,epochs=1,plot = True):

        dataset = Baselinepreprocessing(dataset,past=self.past,future=self.horizon)
        ValidationLoader = dataset.test_loader
        DataLoader = dataset.train_loader
        size = len(DataLoader.dataset)
        average_epoch_loss = [0]
        average_validation_loss = [0]
        average_validation_DS = [0]
        average_validation_d = [0]
        for t in tqdm(range(epochs)):
            #print(f'EPOCH {t+1}/{epochs}')
            loss_list=[0]
            i = 0
            for batch, (X, y) in enumerate(DataLoader):
                i = i+1
                # Compute prediction and loss
                pred = self.model(X)
                np.set_printoptions(precision=5)
                #print(f'n={i}\n pred[0] = {pred[0].detach().numpy()}\n y[0]={y[0].detach().numpy()}')
                

            # Backpropagation
                self.optimizer.zero_grad()
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                lossn = loss.detach().numpy()
                loss_list = np.concatenate((loss_list,lossn),axis=None)

                if batch % 10 == 0:
                    loss, current = loss.item(), batch * len(X)
                    #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_list = loss_list[loss_list!=0]
            average_epoch_loss.append(np.mean(loss_list))

            
            ##Validation

            loss_validation_list = [0]
            loss_DS_list = [0]
            loss_d_list = [0]
            deltapred = [0]
            Deltaspred = [0]
            with torch.no_grad():
                test_loss=[0]
                for X, y in ValidationLoader:
                    pred = self.model(X)
                    nppred = pred.detach().numpy()
                    
                    delta_pred = np.transpose(nppred[:,1])
                    Deltas_pred = np.transpose(nppred[:,0])

                    lossn = self.loss_fn(pred,y).detach().numpy()
                    loss_Ds = self.loss_fn(pred[:,0],y[:,0])
                    loss_d = self.loss_fn(pred[:,1],y[:,1])

                    loss_validation_list = np.concatenate((loss_validation_list,lossn),axis = None)
                    loss_DS_list = np.concatenate((loss_DS_list,loss_Ds),axis = None)
                    loss_d_list = np.concatenate((loss_d_list,loss_d),axis = None)

                    deltapred = np.concatenate((deltapred,delta_pred),axis = None)
                    Deltaspred = np.concatenate((Deltaspred,Deltas_pred),axis = None)  
                
                
            deltapred = deltapred[deltapred!=0]
            Deltaspred = Deltaspred[Deltaspred!=0]
            #print(f'deltapred.shape = {deltapred.shape}\n dataset.y.shape = {dataset.y_test[:,1].shape}')
            loss_validation_list = loss_validation_list[loss_validation_list !=0]
            loss_d_list = loss_d_list[loss_d_list!=0]
            loss_DS_list = loss_DS_list[loss_DS_list!=0]

            average_validation_loss.append(np.mean(loss_validation_list))
            average_validation_d.append(np.mean(loss_d_list))
            average_validation_DS.append(np.mean(loss_DS_list))
        
        
        average_validation_loss = np.delete(average_validation_loss,0)
        average_validation_DS = np.delete(average_validation_DS,0)
        average_validation_d = np.delete(average_validation_d,0)
        average_epoch_loss = np.delete(average_epoch_loss,0)

        #print(self.model.state_dict)


        if plot:

            plt.figure()
            plt.title('Loss function for train and validation', loc='center')
            #plt.subplot(5,1,1)
            #plt.title('average train loss')
            plt.plot(average_epoch_loss)
            plt.subplot(4,1,1)
            plt.title('average validation loss')
            plt.plot(average_validation_loss)
            plt.subplot(4,2,3)
            plt.title('average delta loss')
            plt.plot(average_validation_d)
            plt.subplot(4,2,4)
            plt.title('average delta progress loss')
            plt.plot(average_validation_DS)
            plt.subplot(4,1,3)
            plt.title('Delta prediction vs reality')
            plt.plot(deltapred,color = 'red',label ='Prediction')
            plt.plot(dataset.y_test[:,1],color = 'blue',label = 'Reality')
            plt.legend()
            plt.subplot(4,1,4)
            plt.title('Delta Progress prediction vs reality')
            plt.plot(Deltaspred,color = 'red',label ='Prediction')
            plt.plot(dataset.y_test[:,0],color = 'blue',label = 'Reality')
            plt.legend()
            plt.subplots_adjust(wspace=None,hspace=None)
            #plt.savefig('./prediction100epoch.png')
            plt.show()

    def predict(self,dataset,len,begin,plot=True):
            dataset = Baselinepreprocessing(dataset,self.past,self.horizon)
            full_pred = np.zeros((self.horizon,2))
            X,y= dataset.extractserieforpred(len,begin)
            #print(f'taille des input{dataset.y.shape}')

            for k in range (len):

                nppred = self.model(X[k]).detach().numpy()
                nppred = np.reshape(nppred,[self.horizon,2])
                #print(nppred.shape)
                Deltas_pred = nppred[:,0]
                delta_pred = nppred[:,1]
            
                #deltapred = dataset.scaler1.inverse_transform(delta_pred.reshape(-1,1))
                delta_pred = delta_pred.reshape(self.horizon)
                delta_pred = delta_pred*dataset.std_delta + dataset.mean_delta
                Deltas_pred = Deltas_pred.reshape(self.horizon)
                #Deltaspred = dataset.scaler0.inverse_transform(Deltas_pred.reshape(-1,1))
                Deltas_pred = Deltas_pred * dataset.std_Deltas + dataset.mean_Deltas
                pred = np.vstack([Deltas_pred,delta_pred]).T
                #print(f'k = {k}\n pred = {pred}')
                full_pred = np.vstack([full_pred,pred])
            #print(full_pred.shape)

            if plot:
                plotD = np.zeros(1)
                plotd = np.zeros(1)
                for k in range(len):
                    plotD =np.append(plotD,full_pred[k*self.horizon][0])
                    plotd = np.append(plotd,full_pred[k*self.horizon][1])
                plotD = plotD[plotD !=0]
                plotd = plotd[plotd !=0]
                
                Deltas = y[:,0][1:]
                delta = y[:,1][1:]
                plt.figure()
                plt.subplot(2,1,1)
                plt.title('Delta progress, prediction and reality')
                plt.plot(Deltas, color = 'red',label = 'true Delta progress')
                plt.plot(plotD, color = 'green',label = 'predicted Delta progress')
                plt.legend()  
                plt.subplot(2,1,2)
                plt.plot(delta, color = 'red',label = 'true delta')
                plt.plot(plotd, color = 'green',label = 'predicted delta')
                plt.legend()
                plt.title('delta, prediction and reality')
                plt.show()
                            


            full_pred = np.delete(full_pred,slice(self.horizon),0)       
            #print(np.column_stack((full_pred[:][:,0],y[:,0],full_pred[:][:,1],y[:,1])))
            return full_pred,y





    def load(self, path):
        if not os.path.exists(path):
            print('Fichier non trouvé')
            raise FileNotFoundError
        model = FeedforwardNeuralNet(self.past*3,self.hidden_dim,self.horizon*2)
        model.load_state_dict(torch.load(path))
        #print('loaded model')
        #print(self.model.state_dict)
        #for p in model.parameters():
            #if p.requires_grad:
                #print(p.name, p.data)
        self.model = model.eval()

        

    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
            if not os.path.exists(path):
                raise FileNotFoundError
        model_name = 'Baseline_model_scenario4.pt'
        
        torch.save(self.model.state_dict(),f'{path}/{model_name}')
        #print(self.model.state_dict)
        #for p in self.model.parameters():
            #if p.requires_grad:
                #print(p.name, p.data)