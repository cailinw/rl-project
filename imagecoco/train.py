import os
import random
import time
import numpy as np
import torch

from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from rewarder import Rewarder


#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
SEQ_LENGTH = 32 # sequence length
START_TOKEN = 0
BATCH_SIZE = 512
# TODO: Add hyperparameters here

#########################################################################################
#  Reward Hyper-parameters
#########################################################################################
MID_LAYER_G = [256]
MID_LAYER_R = [512]
re_dropout_keep_prob = 0.45
re_l2_reg_lambda = 1e-5
re_batch_size = BATCH_SIZE
ent_w = 1.0
R_decay = 16 # SGD learn epoch decay
R_rate = 0.01
# TODO: Add hyperparameters here


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 51
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample'+str(ent_w)+'.txt'
eval_file_prefix = 'save/evaler_file'+str(ent_w)
pretrain_file_prefix = 'save/pretrain_file'+str(ent_w)
generated_num = 20000
restore = False
off_num = 2048  # off_policy samples(use PPO2)


#########################################################################################
#  Helper Functions
#########################################################################################



#########################################################################################
#  Main Training Loop
#########################################################################################

def main():
	pass

if __name__ == '__main__':
	main()