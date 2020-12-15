import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.data import DataLoader
from IPython import display

from coco_dataset import COCOImageCaptionsDataset
from generator import Generator
from rewarder import Rewarder
from utils import save

#########################################################################################
#  Hyper-parameters
#########################################################################################
# Constants
vocab_size = 4839

# General parameters
SEQ_LENGTH = 32

# Model parameters
g_hidden_state_size = 768
r_hidden_state_size = 512
action_embed_dim = 56


# Generator training step parameters
generator_batch_size = 64
roll_num = 4

# Rewarder training step parameters
real_batch_size = 64
generated_batch_size = 64

# Training parameters
G_LEARNING_RATE = 5e-5
R_LEARNING_RATE = 0.01
NUM_ITERS = 51
G_ITERS = 1
R_ITERS = 10
restore = False


#########################################################################################
#  Initialization and Pretraining
#########################################################################################

str_map = pickle.load(open("save/str_map.pkl", "rb"))

# Load models
if restore:
    pass
    # TODO: Implement this...inside Generator and Rewarder classes
    # generator = restore_lates("checkpoints/", "g")
    # rewarder = restore_latest("checkpoints/", "r")
else:
    generator = Generator(SEQ_LENGTH, str_map)
    rewarder = Rewarder(
        SEQ_LENGTH,
        vocab_size,
        g_hidden_state_size,
        action_embed_dim,
        r_hidden_state_size,
        R_LEARNING_RATE,
    )

# Load training data
train_data = COCOImageCaptionsDataset("save/train_data.pkl")
train_dataloader = DataLoader(train_data, batch_size=real_batch_size, shuffle=True)

# Pretrain generator
# TODO: Implement training loop here
# generator.pretrain(train_data)

fig, ax = plt.subplots(1,2,figsize=(14,7))
g_losses = []
r_losses = []

#########################################################################################
#  Main Training Loop
#########################################################################################


for it in range(NUM_ITERS):

    # TRAIN GENERATOR
    start = time.time()
    loss_sum = 0
    for g_it in range(G_ITERS):
        g_loss = generator.rl_train_step(
            rewarder, generator_batch_size
        )
        loss_sum += g_loss
    speed = time.time() - start
    g_losses.append(loss_sum / G_ITERS)
    save(generator.model, "/checkpoints/model_checkpoints/generator_" + str(it) + ".pt")
    print(
        "MaxentPolicy Gradient {} iteration, Speed:{:.3f}, Loss:{:.3f}".format(
            it, speed, g_loss
        )
    )


    # TRAIN REWARDER
    start = time.time()
    loss_sum = 0
    for r_it in range(R_ITERS):
        real_trajectories = next(iter(train_dataloader))
        r_loss = rewarder.train_step(real_trajectories[0], generator, generated_batch_size)
        loss_sum += r_loss
    speed = time.time() - start
    r_losses.append(loss_sum / R_ITERS)
    save(rewarder.model, "/checkpoints/model_checkpoints/rewarder_" + str(it) + ".pt")
    print(
        "Reward training {} iteration, Speed:{:.3f}, Loss:{:.3f}".format(
            it, speed, r_loss
        )
    )


    # Logging
    if it % 5 == 0 or it == NUM_ITERS - 1:
        # Generate samples
        generated_samples = generator.generate(batch_size, 1, None, False, False, True)
        output_file = "/checkpoints/generated_samples/generator_sample_" + str(it) + ".txt"
        with open(output_file, 'w') as fout:
        for sentence in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

        # Plot loss
        display.clear_output(wait=True)
        ax[0].cla(); ax[0].plot(g_losses)
        ax[1].cla(); ax[1].plot(r_losses)
        display.display(plt.gcf())
        print(it, g_loss, r_loss)