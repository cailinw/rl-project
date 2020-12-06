import os
import random
import time
import numpy as np
import torch

from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from rewarder import Rewarder
from rollout import Rollout


#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
SEQ_LENGTH = 32 # sequence length
START_TOKEN = 0
BATCH_SIZE = 512
ROLL_NUM = 4
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

def sample_from_policy(rollout, batch_size, generated_num):
	trajectories = []
    policy_probs = []
    for _ in range(int(generated_num / batch_size)):
        samples, sample_probs = rollout.generate()
        trajectories.append(samples)
        policy_probs.append(sample_probs)

def generate_samples(generator, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        samples = trainable_model.generate()
        generated_samples.extend(samples)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


#########################################################################################
#  Main Training Loop
#########################################################################################

def main():
	# Get dataloaders
	gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    vocab_size = 4839
    dis_data_loader = Dis_dataloader(re_batch_size)

    # Load models
    # TODO: Initialize these classes with correct params
    generator = Generator()
    rewarder = Rewarder()
    rollout = Rollout(generator)

    # Create batches from training dataset
    gen_data_loader.create_batches(positive_file)

    for total_batch in range(TOTAL_BATCH):
    	# See what sequences are getting generated with the currently policy
    	# if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
    	# 	generate_samples(generator, BATCH_SIZE, generated_num, eval_file_prefix + str(total_batch))

    	# TRAIN GENERATOR
    	g_losses = []
    	# Generate trajectories (samples) from the current policy (generator)
    	trajectories, policy_probs = sample_from_policy(rollout, BATCH_SIZE, off_num)
    	# Compute the rewards for each of the trajectories at each time step
    	for it in range(off_num // BATCH_SIZE):
    		rewards = rollout.get_reward(trajectories[it], ROLL_NUM, rewarder)
    		avg_reward.append(rewards)
    	# Update the generator
    	for it in range(off_num // BATCH_SIZE):
    		_, g_loss = generator.rl_train_step(trajectories[it], avg_reward[it], policy_probs[it], ent_w)
    		g_losses.append(g_loss)
    	print('MaxentPolicy Gradient {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, np.mean(g_losses)))

    	# TRAIN REWARDER
    	r_losses = []
    	for _ in range(8):
    		generate_samples(generator, BATCH_SIZE, generated_num, negative_file)
    		dis_data_loader.load_train_data(positive_file, negative_file)
    		for _ in range(3):
    			dis_data_loader.reset_pointer()
    			for it in range(dis_data_loader.num_batch):
    				x_text = dis_data_loader.next_batch()
    				# TODO: Update rewarder
    				r_loss
    				r_losses.apend(r_loss)
        print('Reward training {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, np.mean(r_losses)))


if __name__ == '__main__':
	main()