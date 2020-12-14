import numpy as np
import torch
from torch.utils.data import DataLoader

from coco_dataset import COCOImageCaptionsDataset
from rewarder import Rewarder, RewardModel


class Test:
    def __init__(
        self, seq_length, batch_size, hidden_state_size, embed_dim, mlp_hidden_size
    ):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = 4348
        self.hidden_state_size = (hidden_state_size,)
        self.embed_dim = embed_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.learning_rate = 0.01

    def test_reward_model(self):
        hidden_state_size = 64

        model = RewardModel(
            hidden_state_size=hidden_state_size,
            mlp_hidden_size=self.mlp_hidden_size,
            embed_size=self.embed_dim,
            vocab_size=self.vocab_size,
        )

        x = torch.randn((self.batch_size, hidden_state_size))
        a = torch.randint(1, self.vocab_size, (self.batch_size,))

        result = model(x, a)
        print(result)
        print(result.shape)

    def test_rewarder_rewards_to_go(self):
        rewarder = Rewarder(
            self.seq_length,
            self.batch_size // 2,
            self.batch_size // 2,
            self.vocab_size,
            self.hidden_state_size,
            self.embed_dim,
            self.mlp_hidden_size,
            self.learning_rate,
        )
        trajectories = torch.randn((self.batch_size, self.seq_length))
        rewarder.rewards_to_go(trajectories, 4)

    # def test_rewarder_train_step(self):
    # 	generator = Generator()
    # 	model = Rewarder(self.seq_length, self.batch_size // 2, self.batch_size // 2, self.vocab_size, self.hidden_state_size, self.embed_dim, self.mlp_hidden_size, self.learning_rate)
    # 	trajectories = torch.randn((self.batch_size, self.seq_length))
    # 	rewarder.train_step(trajectories, generator)

    # def test_dataloader(self):
    #     dataloader = Dataloader(self.batch_size)
    #     dataloader.load_train_data("save/real_data.txt")
    #     print("sentences: ", dataloader.sentences.shape)
    #     print("sentences batches: ", len(dataloader.sentences_batches))
    #     print("num batches: ", dataloader.num_batches)
    #     one_sentence_batch = dataloader.next_batch()
    #     print("one sentence batch: ", one_sentence_batch.shape, one_sentence_batch)

    def test_dataloader(self):
        train_data = COCOImageCaptionsDataset("save/train_data.pkl")
        train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)
        for batch_idx, (truth, m_in, mask) in enumerate(train_dataloader):
            print("batch_idx: ", batch_idx)
            print(truth.shape, m_in.shape, mask.shape)
            break

    def runtests(self):
        self.test_reward_model()
        # self.test_rewarder_rewards_to_go()
        # self.test_rewarder_train_step()
        self.test_dataloader()


if __name__ == "__main__":
    test = Test(32, 64, 512, 100, 128)
    test.runtests()

