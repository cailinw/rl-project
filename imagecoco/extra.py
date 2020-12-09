# def train_step(self, x_text, generator):
# 	self.optimizer.zero_grad()

# 	num_real = self.batch_size // 2
# 	num_gen = self.batch_size // 2

# 	# Sequences : (batch_size/2, seq_length)
# 	real_x = torch.tensor(x_text[num_real:]) # real seqeunces
# 	gen_x = torch.tensor(x_text[:num_gen]) # generated sequences

# 	# Pass through generator to get hidden states: (seq_len, batch_size, input_size)
# 	real_x_hs = generator.get_hidden_states(real_x)


# # TODO: make sure this is correct and reshape to correct size
# 	gen_x_hs = generator.get_hidden_states(gen_x)

# 	# Compute r(s_t) for all states (t=1:seq_length) and actions (vocab_size) :
# (batch_size, seq_length, vocab_size)
# 	r_real = []
# 	r_gen = []
# 	for t in range(self.seq_length):
# 		r_real.append(torch.unsqueeze(self.model(real_x_hs[t]), 0))  # (1, batch_size, vocab_size)
# 		r_gen.append(torch.unsqueeze(self.model(gen_x_hs[t]), 0)) # (1, batch_size, vocab_size)
# 	r_real = torch.cat(r_real, 0).transpose(0, 1)
# 	r_gen = torch.cat(r_gen, 0).transpose(0, 1)

# 	# Compute R (reward)
# 	# TODO: Get tensor of shape (batch_size, seq_length) for the rewards of actual (s, a)
# 	# TODO: Sum across seq_length to get reward_real and reward_gen

# 	# Compute weighted reward
# 	weighted_reward_real = reward_real / num_real

# 	# TODO: Importance sampling stuff here
# 	weighted_reward_gen = reward_gen / Z

# 	# TODO: Compute reward

# 	loss = weighted_reward_real - weighted_reward_gen
# 	# ?? -> + l2_reg_lambda * (tf.add_n([tf.nn.l2_loss(var) for var in self.r_params if
#   var not in [self.r_embeddings]]))

# 	loss.backward()
# 	self.optimizer.step()
