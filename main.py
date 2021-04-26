import argparse
import Module_Trainer




trainer = Module_Trainer.Trainer("RT_ResNet","linear_1",[0.,2.],0.1,[0.,2.],300,lr = 0.001,batch_size = 128)
trainer.train()
trainer.evaulate(init_y=(0.5,0.2))