import numpy as np
import torch
import torch.nn.functional as F
from Module import *
from utils import Data_Generater
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self,module_name,func_str,t_span,delta_t,comput_domain,epoch,lr,batch_size):
        self.module = choose_module(module_name)(2,40).to(device)
        self.module_optim = torch.optim.Adam(self.module.parameters(),lr) 

        self.dg = Data_Generater(func_str,batch_size)
        self.t_span = t_span
        self.delta_t = delta_t
        self.comput_domain = comput_domain
        self.epoch = epoch
        
    def train(self):
        Train_Dataset = self.dg.data_collector(self.t_span,self.delta_t,self.comput_domain,num_trajectory=100)
        Test_Dataset = self.dg.data_collector(self.t_span,self.delta_t,self.comput_domain,num_trajectory=10)
        
        for i in range(self.epoch):
            self.module.train()
            for data in Train_Dataset:
                z1,z2 = data
                predict_z2 = self.module(z1)
                loss = F.mse_loss(predict_z2,z2).mean()
                self.module_optim.zero_grad()
                loss.backward()
                self.module_optim.step()
            
            self.module.eval()
            total_loss = []
            for data in Test_Dataset:
                z1,z2 = data
                predict_z2 = self.module(z1)
                total_loss.append(F.mse_loss(predict_z2,z2).mean().cpu().detach().numpy())
            print(f"epoch: {i+1}, loss: {np.mean(total_loss)}")

    def evaulate(self,init_y):
        result = self.dg.compute_trajectory(self.t_span,init_y,self.delta_t)
        t = result.t
        trajectory = result.y.T
        y0 = torch.FloatTensor(trajectory[0,:]).view(-1,2).to(device)
        predict_trajectory = y0
        with torch.no_grad():
            for _ in range(len(trajectory)-1):
                y0 = self.module(y0)
                predict_trajectory = torch.cat([predict_trajectory,y0],0)
        predict_trajectory = predict_trajectory.cpu().numpy()

        assert predict_trajectory.shape == trajectory.shape
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.plot(t,trajectory[:,0])
        ax1.scatter(t,predict_trajectory[:,0])

        ax2.plot(t,trajectory[:,1])
        ax2.scatter(t,predict_trajectory[:,1])

        ax3.plot(trajectory[:,0],trajectory[:,1])
        ax3.scatter(predict_trajectory[:,0],predict_trajectory[:,1])
        plt.show()

    def save(self,):
        pass

    def load(self,):
        pass