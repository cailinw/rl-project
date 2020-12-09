import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, vocab_size):
        super(RewardModel, self).__init__()
        self.input_size = input_size  # Size of hidden state of generator model.
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc_i = nn.Linear(input_size + embed_size, hidden_size)
        self.fc_h = nn.Linear(hidden_size, hidden_size)
        self.fc_o = nn.Linear(hidden_size, 1)

    def forward(self, x, a):
        """
        x : (batch_size, input_size)
        a : (batch_size,)
        """

        a_embed = self.embedding(a)
        z = torch.cat((x, a_embed), dim=1)

        # TODO: Potentially change number of layers.
        z = F.relu(self.fc_i(z))
        z = F.relu(self.fc_h(z))
        output = self.fc_o(z)

        return torch.sum(output)


class Rewarder:
    def __init__(
        self,
        seq_length,
        real_batch_size,
        generator_batch_size,
        input_size,
        hidden_size,
        vocab_size,
        learning_rate,
    ):

        self.seq_length = seq_length
        self.real_batch_size = real_batch_size
        self.generator_batch_size = generator_batch_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.embed_dim = hidden_size

        self.model = RewardModel(input_size, hidden_size, vocab_size)
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def compute_reward(self, x, a):
        self.model.eval()
        return self.model(x, a)

    def train_step(self, x_real, generator):
        """
        Perform one step of stochastic gradient descent for the Reward objective,
        as per equation (6) in https://arxiv.org/pdf/1804.11258.pdf.
		x_real : (batch_size, seq_len)
        """

        # Obtain batch of trajectories from real data. Each token is an embedding of the
        # state (context) at that index, embedded by GPT2 pre-trained layer.
        # Also store the actions taken at each timestep.
        # TODO: Compute these. Using generator function x_real -> hidden_states.
        hidden_states_real = torch.zeroes(
            (self.real_batch_size, self.seq_len, self.embed_dim)
        )
        actions_real = torch.zeroes((self.real_batch_size, self.seq_len))

        x_real = hidden_states_real.view(-1, self.embed_dim)
        a_real = actions_real.view(-1, self.vocab_size)

        # Compute reward for each state, action pair in the trajectories.
        reward_real = self.model(x_real, a_real) / self.real_batch_size

        # TODO: Compute these, potentially by:
        # hidden_states_gen, log_probs, x_gen = generator.generate(
        #     num_gen, 1, inc_hidden_state=True, inc_probs=True
        # )
        hidden_states_gen = torch.zeroes(
            (self.generator_batch_size, self.seq_len, self.embed_dim)
        )
        actions_gen = torch.zeroes((self.generator_batch_size, self.seq_len))
        log_probs = torch.zeroes(
            (self.generator_batch_size, self.seq_len, self.vocab_size)
        )

        reward_gen = 0
        w = np.zeros(self.generator_batch_size)
        for j in range(self.generator_batch_size):
            reward = self.model(hidden_states_gen[j], actions_gen[j])

            # We cast anything in the computation of w[j] as numpy arrays so that
            # gradient does not pass through them.
            # Index the log_probs (probability for all actions given tokens in the sequence),
            # using action_gen, which pulls out the action that was actually taken.
            log_q = log_probs[j].data.numpy()[:, actions_gen[j].data.numpy()].sum()
            w[j] = math.exp(reward.data.numpy() - log_q)
            reward_gen += w[j] * reward
        reward_gen /= w.sum()

        self.model.train()
        loss = -(reward_real - reward_gen)
        self.optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
