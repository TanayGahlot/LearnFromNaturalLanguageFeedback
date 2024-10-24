# fmt: off
"""
Comparision of JointLoss against REINFORCE in InvertedPendulum env.

Adaptation of large language models to downstream task has shown promising
results but it remains costly and resource intensive for majority of production
applications. In this work, we focus on improving the convergence rate to reduce
the cost of RL based fine-tuning method. Currently, majority of RL based finetuning
techniques focus on numerical rewards and ignore the natural language
feedback. We combine the objective of policy gradient with imitation learning
objective to get a speedup of 1.5x.

More details about the joint loss objective function, experimental setup
and results can be found in the attached PDF report.
"""
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

plt.rcParams["figure.figsize"] = (10, 5)

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, model_filepath: str = ''):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
            model_filepath: filepath of model weights. If empty initialize from scratch.
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        # Intialize policy network.
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        if model_filepath: 
            self.net.load_state_dict(torch.load(model_filepath))
            self.net.eval()

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

    def save(self, filepath):
        """Saves the policy network"""
        torch.save(self.net.state_dict(), filepath)

class AssistedReinforce:
    """REINFORCE algorithm assisted by an expert.
    
    This is the implementation of joint-loss approach where the objective of 
    imitation learning(behaviour cloning) and policy gradient is combined.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int, model_filepath: str = ''):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
            model_filepath: filepath of model weights. If empty initialize from scratch.
        """

        # Hyperparameters
        self.alpha = .5 # weight for policy gradient objective.
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action.
        self.rewards = []  # Stores the corresponding rewards.
        self.expert_probs = []  # Stores probability values of the expert action.

        # Intialize policy network.
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        if model_filepath: 
            self.net.load_state_dict(torch.load(model_filepath))
            self.net.eval()

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray, expert_action:float) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        expert_action = torch.tensor(np.array(expert_action))
        expert_prob = distrib.log_prob(expert_action)

        action = action.numpy()

        self.probs.append(prob)
        self.expert_probs.append(expert_prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # join loss = policy gradient loss + online imitation learning(via liklihood maximization).
        for log_prob, expert_log_prob, delta in zip(self.probs, self.expert_probs,  deltas):
            policy_gradient_objective = (-1) *(log_prob.mean() * delta)
            imitation_learning_objective = (-1) * expert_log_prob
            loss += (self.alpha*policy_gradient_objective + (1-self.alpha)*imitation_learning_objective)  

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.expert_probs = []

    def save(self, filepath):
        """Saves the policy network"""
        torch.save(self.net.state_dict(), filepath)

class AssistedBC:
    """Online imitation learning algorithm assisted by an expert."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, model_filepath: str = ''):
        """Initializes an agent that learns a policy via behaviour cloning algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
            model_filepath: filepath of model weights. If empty initialize from scratch.
        """

        # Hyperparameters
        self.alpha = .5 # weight for policy gradient objective.
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.expert_probs = []  # Stores probability values of the expert action.

        # Intialize policy network.
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        if model_filepath: 
            self.net.load_state_dict(torch.load(model_filepath))
            self.net.eval()

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray, expert_action:float) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()

        expert_action = torch.tensor(np.array(expert_action))
        expert_prob = distrib.log_prob(expert_action)

        action = action.numpy()
        self.expert_probs.append(expert_prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        loss = 0
        # loss = online imitation learning(via liklihood maximization).
        for expert_log_prob in zip(self.expert_probs):
            imitation_learning_objective =  expert_log_prob[0] * (-1)
            loss += imitation_learning_objective

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables.
        self.expert_probs = []

    def save(self, filepath):
        """Saves the policy network"""
        torch.save(self.net.state_dict(), filepath)


# Create and wrap the environment
env = gym.make("InvertedPendulum-v4")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(3e2)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]

# Learn policy using a novel joint loss function.
rewards_over_seeds_assisted = []
for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = AssistedReinforce(obs_space_dims, action_space_dims)
    expert_agent = REINFORCE(obs_space_dims, action_space_dims, model_filepath="weights/expert_agent_1.pt")
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            expert_action = expert_agent.sample_action(obs)
            action = agent.sample_action(obs, float(expert_action[0]))

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds_assisted.append(reward_over_episodes)

# Learn policy using a behaviour cloning objective.
rewards_over_seeds_bc = []
for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = AssistedBC(obs_space_dims, action_space_dims)
    expert_agent = REINFORCE(obs_space_dims, action_space_dims, model_filepath="weights/expert_agent_1.pt")
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            expert_action = expert_agent.sample_action(obs)
            action = agent.sample_action(obs, float(expert_action[0]))

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds_bc.append(reward_over_episodes)


# Learn policy using policy gradient methods.
rewards_over_seeds = []
for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)


# Format data in the following format: <"episode", "type_of_agent", "reward", "seed"> 
no_of_seeds = len(rewards_over_seeds)
data = []
column_name = ["episode", "type_of_agent", "reward", "seed" ]
for seed in range(no_of_seeds):
    for episode_index, reward in enumerate(rewards_over_seeds[seed]):
        data.append([episode_index, "policy_gradient", reward[0], seed])

    for episode_index, reward in enumerate(rewards_over_seeds_assisted[seed]):
        data.append([episode_index, "joint_loss", reward[0], seed])
    
    for episode_index, reward in enumerate(rewards_over_seeds_bc[seed]):
        data.append([episode_index, "behaviour_cloning", reward[0], seed])

df = pd.DataFrame(data, columns=column_name)
df.to_csv("episodes_and_rewards.csv")

# Plot multiple line plot which compares agents.
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episode", y="reward", hue="type_of_agent", data=df).set(
    title="REINFORCE vs JointLoss vs BC"
)
plt.show()
