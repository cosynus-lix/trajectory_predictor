
import os
from re import X
import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.model.Improved_baseline.Improved_baseline_preprocessing import Improved_Baselinepreprocessing
from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
from ..Model import Model
from trajectory_predictor.model.Baseline.FeedforwardNeuralNet import FeedforwardNeuralNet
import torch.nn as nn
import torch
from tqdm import tqdm

class Improved_BaselineModel(Model):

    def __init__(self, past, hidden_dim):
        super().__init__()
        self.model = FeedforwardNeuralNet(past*3,hidden_dim,2)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = 0.1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) 
        self.past = past
        self.hidden_dim = hidden_dim

    def train(self,dataset,epochs=100,plot=False):
        dataset = Improved_Baselinepreprocessing(dataset,past = self.past, future=1)
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
        
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            if not os.path.exists(path):
                raise FileNotFoundError
        model_name = 'Improved_baseline_3.pt'
        
        torch.save(self.model.state_dict(),f'{path}/{model_name}')

    def predict(self,datasets,init,len,optim):

        ##First prediction
        dataset = Improved_Baselinepreprocessing(datasets,past = self.past)
    
        x = dataset.create_loader(init,True)
        all_pred = self.model(torch.from_numpy(x)).detach().numpy()
    
        Deltas = all_pred[0][0]*dataset.std_Deltas+dataset.mean_Deltas

        if self.optimizer is None:
            raise Exception('SplineOptimizer is not initialized')
        curvature = np.zeros(2)
        #on stocke en 0 le delta progress total, en 1 la curvature à l’instant précédent, et en 2 la curvature future
        curvature[0] = np.sum(dataset.data_not_scaled[:,0][:init])
        curvature[1] = optim.k(Deltas)
        
        #print(f'somme initiale = {curvature[0]}')

        for k in range (len-1):
            x = dataset.create_loader(init+k,False,all_pred[k-1],curvature)
            pred = self.model(torch.from_numpy(x)).detach().numpy() 
            Deltas = pred[0][0]*dataset.std_Deltas+dataset.mean_Deltas
            delta = pred[0][1]*dataset.std_delta+dataset.mean_delta
            if(Deltas <0):
                Deltas = 0
            curvature[0] = curvature[0]+Deltas
            curvature[1] = optim.k(curvature[0])
            
            all_pred = np.vstack((all_pred,pred))

            ##print
            #if (k%100==0):
               #print(f'curvature = {curvature[1]}')

        delta = all_pred[:,1]*dataset.std_delta+dataset.mean_delta
        Deltas=all_pred[:,0]*dataset.std_Deltas+dataset.mean_Deltas
        all_pred = np.column_stack((Deltas,delta))
        return all_pred

            
    
    def load(self,path):
        if not os.path.exists(path):
            print('fichier non trouvé')
            raise FileNotFoundError
        model = FeedforwardNeuralNet(self.past*3,self.hidden_dim,2)
        model.load_state_dict(torch.load(path))
        self.model = model.eval()