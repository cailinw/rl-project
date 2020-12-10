import numpy as np

# import torch


class Rollout:
    def __init__(self, generator, rewarder, seq_length, reward_gamma=1.0):
        self.generator = generator
        self.rewarder = rewarder

        self.seq_length = seq_length
        self.reward_gamma = reward_gamma  # Future reward discount factor

    def sample_from_policy(self, batch_size, num_batches):

        """
        Returns
        trajectories : (num_batches, batch_size, seq_length)
        policy probs : (num_batches, batch_size, seq_length, vocab_size)
        """

        trajectories, probs = self.generator.generate(
            batch_size,
            generated_num // batch_size,
            inc_hidden_state=False,
            inc_probs=True,
            decode=False,
        )
        return trajectories.reshape(num_batches, batch_size, self.seq_length),
        	probs.reshape(num_batches, batch_size, self.seq_length, -1)

    def compute_rewards_to_go(self, trajectories, roll_num):
        """
        Compute reward for each partitial trajectory t:seq_length
        for all t 1:seq_length

        Returns
        rewards_to_go : (num_batches, batch_size, seq_length)
        """

        init_shape = trajectories.shape
        trajectories = trajectories.reshape((-1, self.seq_length))

        rewards_to_go = np.zeros(trajectories.shape[0], self.seq_length)

        for t in range(self.seq_length):
            # Compute reward to go for each trajectory at s_t
            # 	using MCMC sampling
            reward_to_go = 0
            for n in range(roll_num):
                # TODO: rollout trajectories from s_t to s_{seq_length}
                #   to get rollouts (batch_size, seq_length)
                # 	How to do this? Just pass through generator, priming with s_1:t,
                # 	or take the log_probs at s_t and sample from multinomial distribution?
                rollouts = []  # (batch_size, seq_length)

                # Compute reward at each state for each rollout
                rewards = self.rewarder.compute_rewards(
                    rollouts
                )  # (batch_size, seq_length)

                # Compute reward-to-go (batch_size,)
                reward_to_go += rewards[:, n] + (
                    self.reward_gamma * np.sum(rewards[:, t : self.seq_length], axis=1)
                )

            rewards_to_go[:, t] = reward_to_go / roll_num

        return rewards_to_go.reshape(init_shape)
