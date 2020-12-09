# import os
# import random
import time
import numpy as np
import pickle

# import torch

from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from rewarder import Rewarder
from rollout import Rollout


#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
SEQ_LENGTH = 32  # sequence length
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
R_decay = 16  # SGD learn epoch decay
R_rate = 0.01
# TODO: Add hyperparameters here


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 51
positive_file = "save/real_data.txt"
negative_file = "save/generator_sample" + str(ent_w) + ".txt"
eval_file_prefix = "save/evaler_file" + str(ent_w)
pretrain_file_prefix = "save/pretrain_file" + str(ent_w)
generated_num = 20000
restore = False
off_num = 2048  # off_policy samples(use PPO2)


#########################################################################################
#  Helper Functions
#########################################################################################


def generate_samples(generator, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        samples = generator.generate()
        generated_samples.extend(samples)

    with open(output_file, "w") as fout:
        for poem in generated_samples:
            buffer = " ".join([str(x) for x in poem]) + "\n"
            fout.write(buffer)


#########################################################################################
#  Other Constants
#########################################################################################
vocab_size = 4838

#########################################################################################
#  Pretrain Transformer
#########################################################################################
token_map = pickle.load(open("save/token_map.pkt", "rb"))
generator = Generator(SEQ_LENGTH, vocab_size, token_map)
generator.pretrain("save/str_real_data.txt")


#########################################################################################
#  Main Training Loop
#########################################################################################

# Get dataloaders
gen_data_loader = Gen_Data_loader(BATCH_SIZE)
dis_data_loader = Dis_dataloader(re_batch_size)

# Load models
# TODO: Initialize these classes with correct params
generator = Generator()
rewarder = Rewarder()
rollout = Rollout(generator, rewarder)

# Create batches from training dataset
gen_data_loader.create_batches(positive_file)

for total_batch in range(TOTAL_BATCH):
    # See what sequences are getting generated with the currently policy
    # if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
    # 	generate_samples(generator, BATCH_SIZE, generated_num, eval_file_prefix + str(total_batch))

    # TRAIN GENERATOR
    start = time.time()
    g_losses = []
    # Generate trajectories (samples) from the current policy (generator)
    trajectories, policy_probs = rollout.sample_from_policy(BATCH_SIZE, off_num)
    # Compute the rewards for each of the trajectories at each time step
    # (batch_size, seq_length)
    rewards_to_go = rollout.compute_rewards_to_go(trajectories, ROLL_NUM)
    # Update the generator
    for it in range(rewards_to_go.shape[0]):
        g_loss = generator.rl_train_step(
            trajectories[it], rewards_to_go[it], policy_probs[it], ent_w
        )
        g_losses.append(g_loss)
    speed = time.time() - start
    print(
        "MaxentPolicy Gradient {} round, Speed:{:.3f}, Loss:{:.3f}".format(
            total_batch, speed, np.mean(g_losses)
        )
    )

    # TRAIN REWARDER
    start = time.time()
    r_losses = []
    for _ in range(8):
        generate_samples(generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_text = (
                    dis_data_loader.next_batch()
                )  # Real (positive) and generated (negative) text
                r_loss = rewarder.train_step(x_text, generator)
                r_losses.append(r_loss)
    speed = time.time() - start
    print(
        "Reward training {} round, Speed:{:.3f}, Loss:{:.3f}".format(
            total_batch, speed, np.mean(r_losses)
        )
    )
