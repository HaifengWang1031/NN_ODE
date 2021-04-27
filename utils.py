import numpy as np
import scipy.integrate as integrate
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_1(t,x):
    x1,x2 = x[0],x[1]
    return [x1+x2-2,x1-x2] 

def linear_2(t,x):
    x1,x2 = x[0],x[1]
    return [x1-4*x2,4*x1-7*x2]

def damped_pendulum(t,x,alpha = 8.91,beta = 0.2):
    x1,x2 = x[0],x[1]
    return [x2,-alpha*x2-beta*np.sin(x1)]

def genetic_toggle_switch(t,x,alpha1=156.25,alpha2=15.6,beta=2.5,gamma=1,eta = 2.0015,K = 2.9618e-5,IPTG = 1e-5):
    x1,x2 = x[0],x[1]
    z = x1/(1 + IPTG/K)**eta
    return [alpha1/(1+x2**beta)-x1,alpha2/(1+z**gamma)-x2]


govern_funcs = {
    "linear_1":linear_1,
    "linear_2":linear_2,
    "damped_pendulum":damped_pendulum,
    "genetic_toggle_switch":genetic_toggle_switch
    }



class Data_Generater:
    def __init__(self,func_str,batch_size = 128):
        self.govern_funcs = govern_funcs[func_str]
        self.batch_size = 128 

    def compute_trajectory(self,t_span,init_y,delta_t,method="RK45"):
        t_eval = np.arange(t_span[0],t_span[1]+delta_t,delta_t)
        print(t_eval)
        result =integrate.solve_ivp(self.govern_funcs,t_span,init_y,method,t_eval)
        return result

    def data_collector(self,t_span,delta_t,comput_domain,num_trajectory=100):
        x1_domain,x2_domain = comput_domain[0],comput_domain[1]
        x1_dis = x1_domain[1] - x1_domain[0]
        x2_dis = x2_domain[1] - x2_domain[0]
        for i in range(num_trajectory):
            init_y = [np.random.normal()*x1_dis + x1_domain[0],np.random.normal()*x2_dis + x2_domain[0]]
            result = self.compute_trajectory(t_span,init_y,delta_t)
            z1 = result.y[:,:-1]
            z2 = result.y[:,1:]
            if i == 0:
                dataset_z1,dataset_z2 = z1,z2
            else:
                dataset_z1 = np.concatenate([dataset_z1,z1],axis=1)
                dataset_z2 = np.concatenate([dataset_z2,z2],axis=1)
        dataset_z1 = torch.FloatTensor(dataset_z1.T).to(device)
        dataset_z2 = torch.FloatTensor(dataset_z2.T).to(device)
        dataset = TensorDataset(dataset_z1,dataset_z2)
        return DataLoader(dataset,batch_size=self.batch_size,shuffle=True,num_workers=0)


if __name__ == "__main__":
    dg = Data_Generater("damped_pendulum")
    result = dg.compute_trajectory([0,20],[-1.193,-3.876],0.02,"RK45")
    import matplotlib.pyplot as plt

    t = result.t
    y = result.y
    x1 = y[0]
    x2 = y[1]
    plt.plot(t,x1,"r")
    plt.plot(t,x2,"b")
    plt.show()