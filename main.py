import argparse
import Module_Trainer
import numpy as np
import math

config_1 = {
   "module_name":"ResNet",
   "func_str":"linear_2",
   "t_span":[0.,2.],
   "delta_t":0.1,
   "comput_domain":[[0.,2.],[0,2]],
   "epoch" : 300,
   "lr": 0.001, 
   "batch_size" : 128
}


config_3 = {
   "module_name":"RT_ResNet",
   "func_str":"damped_pendulum",
   "t_span":[0.,20.],
   "delta_t":0.1,
   "comput_domain":[[-np.pi,np.pi],[-2*np.pi,2*np.pi]],
   "epoch" : 300,
   "lr": 0.001, 
   "batch_size" : 128
}



trainer = Module_Trainer.Trainer(**config_1)
trainer.train()
trainer.evaulate(init_y = [0,-1])
