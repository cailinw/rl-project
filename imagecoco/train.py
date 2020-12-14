import time
import numpy as np
import pickle

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

# TODO: implement pretraining step here and get the right data
train_data = pickle.load(open("save/train_data.pkl", "rb"))
generator.pretrain(train_data)

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
        "MaxentPolicy Gradient {} round, Speed:{:.3f}, Loss:{:.3f}".format(
            total_batch, speed, np.mean(g_losses)
        )
    )



# for batch_idx, (data, label) in enumerate(train_loader):

    # TRAIN REWARDER
    start = time.time()
    r_losses = []


    
    for _ in range(8):
        # generate_samples(generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_real = dis_data_loader.next_batch()  # Real (positive) text
                r_loss = rewarder.train_step(x_real, generator)
                r_losses.append(r_loss)
    speed = time.time() - start
    print(
        "Reward training {} round, Speed:{:.3f}, Loss:{:.3f}".format(
            total_batch, speed, np.mean(r_losses)
        )
    )
