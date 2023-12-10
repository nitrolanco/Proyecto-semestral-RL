from memory import PPOMemory
from actor import Actor
from critic import Critic
import torch as T
import numpy as np

class PPOAgent(object):
    def __init__(self, input_dims, n_actions, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
               
        self.actor = Actor(input_dims, n_actions, alpha)
        self.critic = Critic(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def store_experience(self, state, action, log_probs, values, reward, done):
        self.memory.store_memory(state, action, log_probs, values, reward, done)

    def select_action(self, observation):
        action_pd = self.actor(observation)
        value = self.critic(observation)
        action = action_pd.sample()
        log_probs = T.squeeze(action_pd.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, log_probs, value

    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(done_arr[k])) - values[k])
                    discount = float(discount)*float(self.gamma)*float(self.gae_lambda)
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)

                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

    def save(self):
        print('... saving models ...')
        self.actor.save()
        self.critic.save()
        

    def load(self):
        print('... loading models ...')
        self.actor.load()
        self.critic.load()