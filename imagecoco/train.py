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
START_TOKEN = 0
BATCH_SIZE = 512
NUM_BATCHES = 4
ROLL_NUM = 4
# TODO: Add hyperparameters here

#########################################################################################
#  Reward Hyper-parameters
#########################################################################################
MID_LAYER_G = [256]
MID_LAYER_R = 512
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
off_num = 2048


#########################################################################################
#  Initialization and Pretraining
#########################################################################################


token_map = pickle.load(open("save/token_map.pkt", "rb"))
assert len(token_map) == vocab_size

# Load models
generator = Generator(
	SEQ_LENGTH,
	token_map
)
rewarder = Rewarder(
	SEQ_LENGTH,
	BATCH_SIZE // 2,
	BATCH_SIZE // 2,
	vocab_size,
	MID_LAYER_R,
	hidden_state_size,
	embed_dim, #
	MID_LAYER_R,
	R_rate
)

train_data = pickle.load(open('save/train_data.pkl', 'rb'))
generator.pretrain(train_data)

#########################################################################################
#  Main Training Loop
#########################################################################################

# Training dataset dataloader
train_dataset = COCOImageCaptionsDataset("save/real_data.txt") # TODO...
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

for total_batch in range(TOTAL_BATCH):
    # See what sequences are getting generated with the currently policy
    # TODO: Uncomment this to save samples throughout training
    # if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
    # 	generate_samples(generator, BATCH_SIZE, generated_num, eval_file_prefix + str(total_batch))

    # TRAIN GENERATOR
    start = time.time()
    g_losses = []
    # Generate trajectories (samples) from the current policy (generator)
    trajectories, probs = generator.generate(
        batch_size,
        generated_num // batch_size,
        inc_hidden_state=False,
        inc_probs=True,
        decode=False,
    )
    trajectories = trajectories.reshape(generated_num // batch_size, batch_size, SEQ_LENGTH),
    probs = probs.reshape(generated_num // batch_size, batch_size, SEQ_LENGTH, -1)
    # Compute the rewards for each of the trajectories at each time step
    # (num_batches, batch_size, seq_length)
    rewards_to_go = rewarder.compute_rewards_to_go(trajectories, rewarder, ROLL_NUM) #, reward_gamma)
    # Update the generator
    for it in range(NUM_BATCHES):
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

    # TRAIN REWARDER
    start = time.time()
    r_losses = []
    for _ in range(8):
        # generate_samples(generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_real = (
                    dis_data_loader.next_batch()
                )  # Real (positive) text
                r_loss = rewarder.train_step(x_real, generator)
                r_losses.append(r_loss)
    speed = time.time() - start
    print(
        "Reward training {} round, Speed:{:.3f}, Loss:{:.3f}".format(
            total_batch, speed, np.mean(r_losses)
        )
    )
