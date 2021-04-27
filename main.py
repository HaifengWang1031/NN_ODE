import argparse
import Module_Trainer
import numpy as np
import math

# parse = argparse.ArgumentParser(description="NN ODE code refactor")
# parse.add_argument("module_name",type=str,default="ResNet",help="Select from: 1. ResNet, 2. RT_ResNet,3. RS_ResNet")
# parse.add_argument("govern_func",type=str,default="linear_1",help="Select from: 1. linear_1, 2. linear_2, 3. damped_pendulum, 4. genetic_toggle_switch")
# args = parse.parse_args()

config = {
   "module_name":"RT_ResNet", #select from "ResNet" "RT_ResNet" "RS_ResNet"
   "func_str":"damped_pendulum", #select from "linear_1" "linear_2" "damped_pendulum" "genetic_toggle_switch"
   "t_span":[0.,20.],
   "delta_t":0.1,
   "comput_domain":[[-np.pi,np.pi],[-2*np.pi,2*np.pi]],
   "epoch" : 300,
   "lr": 0.001, 
   "batch_size" : 128
}
init_y = [-1.193,-3.876]


trainer = Module_Trainer.Trainer(**config)
trainer.train()
trainer.evaulate(init_y)
