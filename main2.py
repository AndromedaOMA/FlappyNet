from datetime import datetime

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import PIL.Image as Image
import flappy_bird_gymnasium

# Hyperparameters
train_mode = True
env_name = "FlappyBird-v0"
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 800
batch_size = 32
gamma = 0.99
lr = 1e-4
buffer_capacity = 10000
max_episodes = 1200
# epsilon_decay = max_episodes // 2
input_channels = 6  # Number of stacked frames
action_dim = 2


class DQN(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

def preprocess_frame(frame):
    gray = Image.fromarray(frame).convert('L')
    resized = gray.resize((84, 84))
    normalized = np.array(resized) / 255.0
    return normalized

def stack_frames(stacked_frames, frame, is_new_episode):
    processed_frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = [processed_frame] * input_channels
    else:
        stacked_frames.pop(0)
        stacked_frames.append(processed_frame)
    return np.stack(stacked_frames, axis=0), stacked_frames

def epsilon_greedy_policy(state, epsilon, policy_net):
    if random.random() < epsilon:
        # return env.action_space.sample()
        return 0 if random.random() < 0.95 else 1
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        return q_values.argmax().item()

def compute_loss(batch, policy_net, target_net):
    states, actions, rewards, next_states, dones = batch

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    return nn.MSELoss()(q_values, target_q_values)

def initialize_network():
    policy_net = DQN(input_channels, action_dim).to(torch.device("cpu"))
    target_net = DQN(input_channels, action_dim).to(torch.device("cpu"))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return policy_net, target_net

def dqn_train(policy_net, target_net):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay

    env = gym.make(env_name, render_mode="rgb_array", use_lidar=False)

    for episode in range(max_episodes):
        _, _ = env.reset()
        state = env.render()
        stacked_frames = deque(maxlen=input_channels)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0

        while True:
            action = epsilon_greedy_policy(state, epsilon, policy_net)
            _, reward, done, _, _ = env.step(action)
            next_state = env.render()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                break

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_loss(batch, policy_net, target_net)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            save_network(policy_net)

        print(f"Episode {episode}, Reward: {episode_reward}")
    env.close()

def test_agent(policy_net, num_episodes=100):
    env = gym.make(env_name, render_mode="human", use_lidar=False)
    policy_net.eval()

    for episode in range(num_episodes):
        _, _ = env.reset()
        state = env.render()
        stacked_frames = deque(maxlen=input_channels)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state_tensor).argmax().item()
            _, reward, done, _, _ = env.step(action)
            next_state = env.render()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
            episode_reward += reward

        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")

    policy_net.train()
    env.close()


def save_network(policy_net):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"NN_states\\history\\dqn_flappybird_{timestamp}"
    torch.save(policy_net.state_dict(), filename)
    torch.save(policy_net.state_dict(), "NN_states\\last_state\\dqn_flappybird")


def load_last_network():
    policy_net = DQN(input_channels, action_dim).to(torch.device("cpu"))
    policy_net.load_state_dict(torch.load("NN_states\\last_state\\dqn_flappybird", weights_only=True))
    return policy_net


if __name__ == '__main__':
    if train_mode:
        policy_net, target_net = initialize_network()
        dqn_train(policy_net, target_net)

        save_network(policy_net)
    else:
        policy_net = load_last_network()

    test_agent(policy_net)