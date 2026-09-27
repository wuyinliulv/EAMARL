import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        # print("args",self.args)
        self.agent_id = agent_id
        # print("agent_id",self.agent_id)
        self.policy = MADDPG(args, agent_id)
        # self.policy.actor_network = self.policy.actor_network.to(device)
        # self.policy.critic_network = self.policy.critic_network.to(device)
        # self.policy.actor_target_network = self.policy.actor_target_network.to(device)
        # self.policy.critic_target_network = self.policy.critic_target_network.to(device)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(0, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            # noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            # u += noise
            u = np.clip(u, 0, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

