import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, hidden_state_size, mlp_hidden_size, embed_size, vocab_size):
        super(RewardModel, self).__init__()
        # Size of hidden state of generator model.
        self.hidden_state_size = hidden_state_size
        self.mlp_hidden_size = mlp_hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc_i = nn.Linear(hidden_state_size + embed_size, mlp_hidden_size)
        self.fc_h = nn.Linear(mlp_hidden_size, mlp_hidden_size)
        self.fc_o = nn.Linear(mlp_hidden_size, 1)

    def forward(self, x, a):
        """
        x : (batch_size, hidden_state_size)
        a : (batch_size,)
        """

        a_embed = self.embedding(a)
        z = torch.cat((x, a_embed), dim=len(x.shape) - 1)

        # TODO: Potentially change number of layers.
        z = F.relu(self.fc_i(z))
        z = F.relu(self.fc_h(z))
        output = self.fc_o(z)

        return output


class Rewarder:
    def __init__(
        self,
        seq_length,
        real_batch_size,
        generator_batch_size,
        vocab_size,
        hidden_state_size,
        embed_dim,
        mlp_hidden_size,
        learning_rate,
    ):

        self.seq_length = seq_length
        self.real_batch_size = real_batch_size
        self.generator_batch_size = generator_batch_size
        self.vocab_size = vocab_size
        self.hidden_state_size = hidden_state_size  # hidden state of generator
        self.embed_dim = embed_dim  # action embedding
        self.mlp_hidden_size = mlp_hidden_size  # hidden layers of reward model
        self.learning_rate = learning_rate

        self.model = RewardModel(
            hidden_state_size, mlp_hidden_size, embed_dim, vocab_size
        ).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def compute_rewards_to_go(
        self, trajectories, generator, roll_num=4, reward_gamma=1.0
    ):
        """
        Compute reward for each partitial trajectory t:seq_length
        for all t 1:seq_length

        trajectories: (batch_size, seq_len) type: int 

        Returns
        rewards_to_go : (num_batches, batch_size, seq_length)
        """

        init_shape = trajectories.shape
        trajectories = trajectories.reshape((-1, self.seq_length))
        batch_size = init_shape[0]

        rewards_to_go = torch.zeros(batch_size, self.seq_length).cuda()
        print(rewards_to_go.shape)

        for t in range(self.seq_length):
            # Compute reward to go for each trajectory at s_t
            #   using MCMC sampling

            current_traj = trajectories[:, 0 : (t + 1)]
            # current_traj.shape = (batch_size, starting_seq_len))
            for k in range(roll_num):
                rollouts, rollout_hidden_states = generator.generate(
                    batch_size,
                    1,
                    current_traj,
                    inc_hidden_state=True,
                    inc_probs=False,
                    decode=False,
                )

                # rollouts_hidden_states.shape =  (batch_size, ending_seq_len, hidden_dim)
                # rollouts[0].shape = (batch_size, ending_seq_len)
                # rewards.shape = (batch_size*seq_len, 1)
                rewards = self.model(
                    rollout_hidden_states.view(-1, self.hidden_state_size),
                    rollouts[0].view(-1),
                )
                result = rewards.view(batch_size, self.seq_length).sum(axis=1)

                rewards_to_go[:, t] += result

            rewards_to_go[:, t] /= roll_num

        return rewards_to_go

    def train_step(self, real_batch, generator):
        """
        Perform one step of stochastic gradient descent for the Reward objective,
        as per equation (6) in https://arxiv.org/pdf/1804.11258.pdf.
        real_batch : (batch_size, seq_len)
        """

        # Compute reward for real sequences
        # Obtain batch of trajectories from real data. Each token is an embedding of the
        # state (context) at that index, embedded by GPT2 pre-trained layer.
        # Also store the actions taken at each timestep.
        hidden_states_real = generator.get_hidden_state(real_batch)
        # hidden_states_real.shape = self.real_batch_size, self.seq_len, self.embed_dim

        x_real = hidden_states_real.view(-1, self.embed_dim)
        a_real = real_batch.view(-1, 1)

        # Compute reward for each state, action pair in the trajectories.
        reward_real = self.model(x_real, a_real).sum() / self.real_batch_size

        actions_gen, hidden_states_gen, log_probs = generator.generate(
            self.generator_batch_size,
            1,
            None,
            inc_hidden_state=True,
            inc_probs=True,
            decode=False,  # TODO: not sure about this
        )

        reward_gen = 0
        w = np.zeros(self.generator_batch_size)
        for j in range(self.generator_batch_size):
            reward = torch.sum(self.model(hidden_states_gen[j], actions_gen[j]))

            # We cast anything in the computation of w[j] as numpy arrays so that
            # gradient does not pass through them.
            # Index the log_probs (probability for all actions given tokens in the sequence),
            # using action_gen, which pulls out the action that was actually taken.
            log_q = log_probs[j].data.numpy()[:, actions_gen[j].data.numpy()].sum()
            w[j] = math.exp(reward.data.numpy() - log_q)
            reward_gen += w[j] * reward
        reward_gen /= w.sum()

        loss = -(reward_real - reward_gen)
        self.optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
