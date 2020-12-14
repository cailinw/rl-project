import time
import numpy as np
import pickle
from torch.utils.data import DataLoader

from coco_dataset import COCOImageCaptionsDataset
from generator import Generator
from rewarder import Rewarder

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
SEQ_LENGTH = 32  # sequence length
G_BATCH_SIZE = 512
generated_num = 20000
ROLL_NUM = 4

#########################################################################################
#  Reward Hyper-parameters
#########################################################################################
r_hidden_state_size = 512
R_BATCH_SIZE = 512
R_LEARNING_RATE = 0.01


#########################################################################################
#  Basic Training Parameters
#########################################################################################
EPOCHS = 51

generated_num = 20000
restore = False
off_num = 2048

#########################################################################################
#  Initialization and Pretraining
#########################################################################################

# TODO: What should str map be? Unpickle it into dict here?
str_map = pickle.load(open("save/str_map.pkl", "rb"))

# Load models
generator = Generator(SEQ_LENGTH, str_map)
rewarder = Rewarder(
    SEQ_LENGTH,
    BATCH_SIZE // 2,
    BATCH_SIZE // 2,
    vocab_size,
    r_hidden_state_size,
    hidden_state_size,
    embed_dim,  #
    mlp_hidden_size,
    R_LEARNING_RATE,
)

# Load training data
train_data = COCOImageCaptionsDataset("save/train_data.pkl")
train_dataloader = DataLoader(train_data, batch_size=R_BATCH_SIZE, shuffle=True)

# Pretrain generator
# TODO: Implement training loop here
#generator.pretrain(train_data)

#########################################################################################
#  Main Training Loop
#########################################################################################


for epoch in range(EPOCH):
    # See what sequences are getting generated with the currenty policy
    # TODO: Make this write generated sequences to log
    # if epoch % 5 == 0 or epoch == EPOCHS - 1:
    # 	generator.generate(batch_size, 1, None, False, False, True)


    # TRAIN GENERATOR
    start = time.time()
    g_losses = []
    # Generate trajectories (samples) from the current policy (generator)
    num_batches = generated_num // G_batch_size
    trajectories, probs = generator.generate(
        G_BATCH_SIZE,
        num_batches,
        None,
        inc_hidden_state=False,
        inc_probs=True,
        decode=False,
    )
    trajectories = trajectories.reshape(num_batches, G_BATCH_SIZE, SEQ_LENGTH),
    probs = probs.reshape(num_batches, G_BATCH_SIZE, SEQ_LENGTH, -1)
    for batch_idx in range(num_batches):
        rewards_to_go = rewarder.compute_rewards_to_go(
            trajectories[batch_idx] ,rewarder, ROLL_NUM #, reward_gamma
        )
        g_loss = generator.rl_train_step(
            trajectories[it], rewards_to_go[it], probs[it], ent_w
        )
        g_losses.append(g_loss)
    speed = time.time() - start
    print(
        "MaxentPolicy Gradient {} epoch, Speed:{:.3f}, Loss:{:.3f}".format(
            epoch, speed, np.mean(g_losses)
        )
    )

    # TRAIN REWARDER
    start = time.time()
    r_losses = []
    for _ in range(8):
        for batch_idx, trajectories_real in enumerate(train_dataloader):
            r_loss = rewarder.train_step(trajectories_real, generator)
            r_losses.append(r_loss)
    speed = time.time() - start
    print(
        "Reward training {} epoch, Speed:{:.3f}, Loss:{:.3f}".format(
            epoch, speed, np.mean(r_losses)
        )
    )