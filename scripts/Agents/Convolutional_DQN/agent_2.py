import flappy_bird_gymnasium
import gymnasium
import torch
import itertools
import random
from datetime import datetime

from scripts import DuelingCNN, ReplayMemory, FrameStacker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Agent2:
    def __init__(self, model_to_test, continue_training=False):

        self.continue_training = continue_training

        # Hyperparameters
        self.replay_memory_size = 200000
        self.mini_batch_size = 84
        self.epsilon_init = 1
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.1
        self.network_sync_rate = 100
        self.learning_rate_a = 0.001
        self.discount_factor_g = 0.999
        self.stop_on_reward = 500

        self.max_episodes = 100003
        self.enable_double_dqn = True
        self.model_to_test = model_to_test
        self.image_stack_dimension = 4

        self.loss_fn = torch.nn.SmoothL1Loss()
        # self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

        # criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        # optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    def dqn_train(self):
        env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

        policy_dqn = DuelingCNN(input_channels=self.image_stack_dimension, input_size=64, out_layer_dim=2).to(device)
        # Duplicate the DQN for target_dqn where we are going to compute the Q values
        target_dqn = DuelingCNN(input_channels=self.image_stack_dimension, input_size=64, out_layer_dim=2).to(device)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        frame_stacker = FrameStacker(stack_size=self.image_stack_dimension, height=64, width=64)

        if not self.continue_training:
            # Copy the w and b at target_dqn from policy_dqn
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # frame_stacker = FrameStacker(stack_size=self.image_stack_dimension, height=64, width=64)
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            step_count = 0
            start_episode = 0
            reward_per_episode = []
        else:
            # Load checkpoint
            checkpoint = torch.load("../checkpoint.pth", map_location=device)

            policy_dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])
            target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # frame_stacker = checkpoint['frame_stacker']
            memory = checkpoint['memory']
            epsilon = checkpoint['epsilon']
            step_count = checkpoint['step_count']
            start_episode = checkpoint['episode'] + 1
            reward_per_episode = checkpoint['reward_per_episode']

            print(f"Resumed training from episode {start_episode}.")

        for episode in range(start_episode, self.max_episodes):
            _, _ = env.reset()
            frame = env.render()
            stacked_frames = frame_stacker.reset(frame)
            frame = torch.tensor(stacked_frames, dtype=torch.float, device=device).unsqueeze(0)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(frame).argmax().item()

                # Environment step
                _, reward, terminated, _, _ = env.step(action)
                new_frame = env.render()
                stacked_frames = frame_stacker.update(new_frame)

                new_frame = torch.tensor(stacked_frames, dtype=torch.float, device=device).unsqueeze(0)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                memory.append((frame, torch.tensor(action), new_frame, reward, terminated))
                step_count += 1

                episode_reward += reward
                frame = new_frame

                # Train the network
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

            reward_per_episode.append(episode_reward)

            # Update epsilon
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            # Sync target network
            if step_count >= self.network_sync_rate:
                # Copy the w and b at target_dqn from policy_dqn at each self.network_sync_rate
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0

            if episode % 100 == 0:
                checkpoint = {
                    'episode': episode,
                    # 'frame_stacker': frame_stacker,
                    'memory': memory,
                    'policy_dqn_state_dict': policy_dqn.state_dict(),
                    'target_dqn_state_dict': target_dqn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'reward_per_episode': reward_per_episode,
                    'epsilon': epsilon,
                    'step_count': step_count,
                }
                torch.save(checkpoint, "./best_models_CNN/DuelingCNN/checkpoint.pth")

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"./best_models_CNN/DuelingCNN/trained_q_function_{timestamp}_{episode_reward:.2f}.pth"
                torch.save(policy_dqn.state_dict(), filename)
                torch.save(policy_dqn.state_dict(), f"./best_models_CNN/DuelingCNN/last_state/trained_q_function")

                print(f"Checkpoint and model with reward of {episode_reward} saved at episode {episode}.")

            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

            # Exit
            if episode_reward >= self.stop_on_reward:
                print(f"Training stopped after achieving reward threshold of {self.stop_on_reward}.")
                break

    def test_agent(self):
        env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

        policy_dqn = DuelingCNN(input_channels=self.image_stack_dimension, input_size=64, out_layer_dim=2).to(device)
        policy_dqn.load_state_dict(torch.load(self.model_to_test, weights_only=True))
        policy_dqn.eval()

        frame_stacker = FrameStacker(stack_size=self.image_stack_dimension, height=64, width=64)

        for episode in itertools.count():
            _, _ = env.reset()
            frame = env.render()
            stacked_frames = frame_stacker.reset(frame)
            frame = torch.tensor(stacked_frames, dtype=torch.float, device=device).unsqueeze(0)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                action = policy_dqn(frame).argmax().item()

                # Environment step
                _, reward, terminated, _, _ = env.step(action)
                new_frame = env.render()
                stacked_frames = frame_stacker.update(new_frame)

                frame = torch.tensor(stacked_frames, dtype=torch.float, device=device).unsqueeze(0)
                episode_reward += reward

            print(f"Episode: {episode}, Reward: {episode_reward}")

        policy_dqn.train()
        env.close()

    def run(self, is_training=True):
        if is_training:
            self.dqn_train()
        else:
            self.test_agent()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Q-Learning Formula:
            q[state, action] = lr * (reward + discount_factor * max(q[new_state,:]) - q[state, action])
        ---------------------------------------------------------------------------------
        DQN Target Formula:
            Qt[state, action] = reward if new_state is terminal else
                               reward + discount_factor * max(Qt[new_state,:])
        ---------------------------------------------------------------------------------
        Double DQN Target Formula:
            best_action = arg(max(Qp[new_state,:]))
            Qt[state, action] = reward if new_state is terminal else
                               reward + discount_factor * max(Qt[best_action])
        """
        frames, actions, new_frames, rewards, terminations = zip(*mini_batch)

        frames = torch.cat(frames)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        new_frames = torch.cat(new_frames)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                """Double DQN Target Formula"""
                best_action_from_policy = policy_dqn(new_frames).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_frames).gather(dim=1, index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                """DQN Target Formula"""
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_frames).max(dim=1)[0]

        # Compute Q values from policy
        current_q = policy_dqn(frames).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
