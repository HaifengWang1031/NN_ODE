import torch
import torch.nn as nn
import torch.nn.functional as F



class ResNet(nn.Module):
    def __init__(self,sol_dim,hidden_units=30,M = 3):
        super().__init__()
        self.layer1 = nn.Linear(sol_dim,hidden_units)
        self.layers = []
        for _  in range(M-2):
            self.layers.append(nn.Linear(hidden_units,hidden_units))
            self.layers.append(nn.Sigmoid())
        self.layer2 = nn.Sequential(*self.layers)
        self.layer3 = nn.Linear(hidden_units,sol_dim,bias=False)

    def forward(self,x):
        y = torch.sigmoid(self.layer1(x))
        y = self.layer2(y)
        y = x + self.layer3(y)
        return y


class RT_ResNet(nn.Module):
    def __init__(self,sol_dim,hidden_units=30,K=3,M=3):
        super().__init__()
        self.K = K
        self.res_block = ResNet(sol_dim,hidden_units,M=3)

    def forward(self,x):
        y = x
        for _ in range(self.K):
            y = self.res_block(y)
        
        return y

    def step_forward(self,x):
        return self.res_block(y)


class RS_ResNet(nn.Module):
    def __init__(self,sol_dim,hidden_units=30,K=3,M=3):
        super().__init__()
        self.res_blocks = []
        self.K = K
        self.ptr = 0
        for _ in range(K):
            self.res_blocks.append(ResNet(sol_dim,hidden_units,M))
        self.layers = nn.Sequential(*self.res_blocks)

    def forward(self,x):
        return self.layers(x)
    
    def reset_ptr(self):
        self.ptr = 0

    def step_forward(self,x):
        y = self.res_blocks[self.ptr](x)
        self.ptr += 1
        if self.ptr>= self.K:
            self.reset_ptr()
        return y


class RK_ResNet(nn.Module):
    def __init__(self,sol_dim,hidden_units=30,M=3):
        super().__init__()
        self.res_block_1 = ResNet(sol_dim)
        self.res_block_2 = ResNet(sol_dim)
        self.res_block_3 = ResNet(sol_dim)
        self.sum_weight = nn.parameter.Parameter(torch.rand(3))

    def forward(self,x):
        weight = F.softmax(self.sum_weight,dim=0)
        result = x
        y1 = self.res_block_1(x)
        y2 = self.res_block_2(y1)
        y3 = self.res_block_3(y2)

        result = weight[0]*y1 + weight[1]*y2 + weight[2]*y3


        return result
    

module_dict = {
    "ResNet":ResNet,
    "RT_ResNet":RT_ResNet,
    "RS_ResNet":RS_ResNet,
    "RK_ResNet":RK_ResNet
}    

def choose_module(name):
    assert isinstance(name,str)
    if name in module_dict.keys():
        return module_dict[name]
    else:
        print("Warning: illegal module name")
        print("Module list")
        for index,mn in enumerate(module_dict.keys()):
            print(index,": ",mn)
        raise RuntimeError


if __name__ == "__main__":
    rk_res = RK_ResNet(2)
    print(list(rk_res.parameters()))
