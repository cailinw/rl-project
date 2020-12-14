import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.data import DataLoader
from IPython import display

from coco_dataset import COCOImageCaptionsDataset
from generator import Generator
from rewarder import Rewarder
#from utils import save, restore_latest

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
SEQ_LENGTH = 32  # sequence length
G_BATCH_SIZE = 512
generated_num = 20000
ROLL_NUM = 4
g_hidden_state_size = 768

#########################################################################################
#  Reward Hyper-parameters
#########################################################################################
R_BATCH_SIZE = 512
r_hidden_state_size = 512
action_embed_dim = 56

#########################################################################################
#  Basic Training Parameters and Constants
#########################################################################################
vocab_size = 4839
G_LEARNING_RATE = 5e-5
R_LEARNING_RATE = 0.01
G_ITERS = 1
R_ITERS = 10
EPOCHS = 51
restore = False

#########################################################################################
#  Initialization and Pretraining
#########################################################################################

# TODO: What should str map be? Unpickle it into dict here?
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
train_dataloader = DataLoader(train_data, batch_size=R_BATCH_SIZE, shuffle=True)

# Pretrain generator
# TODO: Implement training loop here
# generator.pretrain(train_data)

fig, ax = plt.subplots(1,2,figsize=(14,7))
g_losses = []
r_losses = []

#########################################################################################
#  Main Training Loop
#########################################################################################


for epoch in range(EPOCHS):

    # TRAIN GENERATOR
    start = time.time()
    losses = []
    for _ in range(G_ITERS):
        num_batches = generated_num // G_BATCH_SIZE
        for batch_idx in range(num_batches):
            g_loss = generator.rl_train_step(
                rewarder, G_BATCH_SIZE
            )
            losses.append(g_loss)
    speed = time.time() - start
    g_loss = np.mean(losses)
    g_losses.append(g_loss)
    # generator.save_model()  # TODO: Add this
    print(
        "MaxentPolicy Gradient {} epoch, Speed:{:.3f}, Loss:{:.3f}".format(
            epoch, speed, g_loss
        )
    )

    # TRAIN REWARDER
    start = time.time()
    losses = []
    for _ in range(R_ITERS):
        for batch_idx, trajectories_real in enumerate(train_dataloader):
            r_loss = rewarder.train_step(trajectories_real, generator, G_BATCH_SIZE)
            losses.append(r_loss)
    speed = time.time() - start
    r_loss = np.mean(losses)
    r_losses.append(r_loss)
    # rewarder.save_model()  # TODO: Add this
    print(
        "Reward training {} epoch, Speed:{:.3f}, Loss:{:.3f}".format(
            epoch, speed, r_loss
        )
    )


    # Logging
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        # Generate samples
        generated_samples = generator.generate(batch_size, 1, None, False, False, True)
        output_file = "/save/generated_samples/generator_sample_" + str(epoch) + ".txt"
        with open(output_file, 'w') as fout:
        for sentence in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

        # Plot loss
        display.clear_output(wait=True)
        ax[0].cla(); ax[0].plot(g_losses)
        ax[1].cla(); ax[1].plot(r_losses)
        display.display(plt.gcf())
        print(epoch, g_loss, r_loss)

