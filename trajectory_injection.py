# fmt: off
"""
Training using REINFORCE for Mujoco
===================================

.. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig1.gif
  :width: 400
  :alt: agent-environment-diagram

This tutorial serves 2 purposes:
 1. To understand how to implement REINFORCE [1] from scratch to solve Mujoco's InvertedPendulum-v4
 2. Implementation a deep reinforcement learning algorithm with Gymnasium's v0.26+ `step()` function

We will be using **REINFORCE**, one of the earliest policy gradient methods. Unlike going under the burden of learning a value function first and then deriving a policy out of it,
REINFORCE optimizes the policy directly. In other words, it is trained to maximize the probability of Monte-Carlo returns. More on that later.

**Inverted Pendulum** is Mujoco's cartpole but now powered by the Mujoco physics simulator -
which allows more complex experiments (such as varying the effects of gravity).
This environment involves a cart that can moved linearly, with a pole fixed on it at one end and having another end free.
The cart can be pushed left or right, and the goal is to balance the pole on the top of the cart by applying forces on the cart.
More information on the environment could be found at https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/

**Training Objectives**: To balance the pole (inverted pendulum) on top of the cart

**Actions**: The agent takes a 1D vector for actions. The action space is a continuous ``(action)`` in ``[-3, 3]``,
where action represents the numerical force applied to the cart
(with magnitude representing the amount of force and sign representing the direction)

**Approach**: We use PyTorch to code REINFORCE from scratch to train a Neural Network policy to master Inverted Pendulum.

An explanation of the Gymnasium v0.26+ `Env.step()` function

``env.step(A)`` allows us to take an action 'A' in the current environment 'env'. The environment then executes the action
and returns five variables:

-  ``next_obs``: This is the observation that the agent will receive after taking the action.
-  ``reward``: This is the reward that the agent will receive after taking the action.
-  ``terminated``: This is a boolean variable that indicates whether or not the environment has terminated.
-  ``truncated``: This is a boolean variable that also indicates whether the episode ended by early truncation, i.e., a time limit is reached.
-  ``info``: This is a dictionary that might contain additional information about the environment.
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


# %%
# Policy Network
# ~~~~~~~~~~~~~~
#
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig2.png
#
# We start by building a policy that the agent will learn using REINFORCE.
# A policy is a mapping from the current environment observation to a probability distribution of the actions to be taken.
# The policy used in the tutorial is parameterized by a neural network. It consists of 2 linear layers that are shared between both the predicted mean and standard deviation.
# Further, the single individual linear layers are used to estimate the mean and the standard deviation. ``nn.Tanh`` is used as a non-linearity between the hidden layers.
# The following function estimates a mean and standard deviation of a normal distribution from which an action is sampled. Hence it is expected for the policy to learn
# appropriate weights to output means and standard deviation based on the current observation.


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


# %%
# Building an agent
# ~~~~~~~~~~~~~~~~~
#
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig3.jpeg
#
# Now that we are done building the policy, let us develop **REINFORCE** which gives life to the policy network.
# The algorithm of REINFORCE could be found above. As mentioned before, REINFORCE aims to maximize the Monte-Carlo returns.
#
# Fun Fact: REINFROCE is an acronym for " 'RE'ward 'I'ncrement 'N'on-negative 'F'actor times 'O'ffset 'R'einforcement times 'C'haracteristic 'E'ligibility
#
# Note: The choice of hyperparameters is to train a decently performing agent. No extensive hyperparameter
# tuning was done.
#


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
    """REINFORCE algorithm assisted by an expert."""

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

        self.forked = False # decides whether to use agent's probs vs experts prob.

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

        self.probs.append(prob)
        return action.numpy()
    
    def log_expert_prob(self, state:np.ndarray, expert_action: float) ->np.ndarray:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        expert_action = torch.tensor(np.array([expert_action]))
        return distrib.log_prob(expert_action)

    def update(self, expert_trajectory:List[Dict[str,str]]):
        """Updates the policy network's weights."""
        # -----Compute gradient for the agent's trajectory.-----
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # Agent's trajectory.
        for log_prob, delta in zip(self.probs,  deltas):
            policy_gradient_objective = (-1) *(log_prob.mean() * delta)
            loss += policy_gradient_objective  

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.expert_probs = []
        self.forked = False

        # -----Compute gradient for the expert's trajectory.-----
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for step in expert_trajectory[::-1]:
            R = step["reward"]
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        # Compute the expert probs.
        expert_probs = []
        for step in expert_trajectory:
            expert_probs.append(self.log_expert_prob(state=step['obs'], expert_action=step["action"]))

        loss = 0
        # Expert's trajectory.
        for log_prob, delta in zip(expert_probs,  deltas):
            policy_gradient_objective = (-1) *(log_prob.mean() * delta)
            loss += policy_gradient_objective  

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fork(self):
        """Called to demark the point after which agent follows agent's policy."""
        self.forked = True

    def save(self, filepath):
        """Saves the policy network"""
        torch.save(self.net.state_dict(), filepath)


# %%
# Now lets train the policy using REINFORCE to master the task of Inverted Pendulum.
#
# Following is the overview of the training procedure
#
#    for seed in random seeds
#        reinitialize agent
#
#        for episode in range of max number of episodes
#            until episode is done
#                sample action based on current observation
#
#                take action and receive reward and next observation
#
#                store action take, its probability, and the observed reward
#            update the policy
#
# Note: Deep RL is fairly brittle concerning random seed in a lot of common use cases (https://spinningup.openai.com/en/latest/spinningup/spinningup.html).
# Hence it is important to test out various seeds, which we will be doing.


# Create and wrap the environment
agent_env = gym.make("InvertedPendulum-v4")
expert_env = gym.make("InvertedPendulum-v4")
wrapped_agent_env = gym.wrappers.RecordEpisodeStatistics(agent_env, 50)  # Records episode-reward
wrapped_expert_env = gym.wrappers.RecordEpisodeStatistics(expert_env, 50)

total_num_episodes = int(6e2)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = agent_env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = agent_env.action_space.shape[0]

# Probability that governs the point at which expert env stops following agent's policy.
# For ex: if its .1, we expect expert_env to follow agent policy for 10 steps(on average) i.e 1/p 
fork_prob = .4

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
        obs_agent, info_agent = wrapped_agent_env.reset(seed=seed)
        obs_expert, info_expert = wrapped_expert_env.reset(seed=seed)

        done_agent = False
        expert_trajectory = []
        # Phase-1: Expert and Agent env follow agent's policy.
        while not done_agent and np.random.random() >= fork_prob:
            # print(f'inside phase-1: seed: {seed}, episode:{episode}')
            agent_action = agent.sample_action(obs_agent)

            expert_trajectory.append({"obs": obs_expert, "action": agent_action})

            # Act using agent's policy in expert and agent's env.        
            obs_agent, reward_agent, terminated_agent, truncated_agent, info_agent = wrapped_agent_env.step(agent_action)
            obs_expert, reward_expert, terminated_expert, truncated_expert, info_expert = wrapped_expert_env.step(agent_action)
            
            expert_trajectory[-1]["reward"] = reward_expert
            
            # We need to make sure that the parameter of the expert env and agent env
            # are the same for this phase since we want to expert to follow the agent trajectory.
            assert (obs_expert==obs_agent).all()
            assert reward_expert==reward_agent
            assert terminated_expert == terminated_agent
            assert truncated_expert == truncated_agent
            assert info_expert == info_agent

            agent.rewards.append(reward_agent)
            done_agent = terminated_agent or truncated_agent
        
        # Phase-2: Expert env follows expert policy and Agent env follow agent's policy.
        while not done_agent:
            # print(f'inside phase-2(agent): seed: {seed}, episode:{episode}')
            agent_action = agent.sample_action(obs_agent)

            # Act using agent's policy.     
            obs_agent, reward_agent, terminated_agent, truncated_agent, info_agent = wrapped_agent_env.step(agent_action)

            agent.rewards.append(reward_agent)
            done_agent = terminated_agent or truncated_agent
        done_expert = False
        while not done_expert:
            # print(f'inside phase-2(expert): seed: {seed}, episode:{episode}')
            expert_action = expert_agent.sample_action(obs_expert)

            expert_trajectory.append({"obs": obs_expert, "action": expert_action})
            # Act using expert's policy in expert env.   
            obs_expert, reward_expert, terminated_expert, truncated_expert, info_expert = wrapped_expert_env.step(expert_action)
            done_expert = terminated_expert or truncated_expert

            expert_trajectory[-1]["reward"] = reward_expert

        reward_over_episodes.append(wrapped_agent_env.return_queue[-1])
        agent.update(expert_trajectory)

        if episode % 1000 == 0:
            agent_avg_reward = int(np.mean(wrapped_agent_env.return_queue))
            expert_avg_reward = int(np.mean(wrapped_expert_env.return_queue))
            print("Episode:", episode, "Average Reward(Agent):", agent_avg_reward)
            print("Episode:", episode, "Average Reward(Expert):", expert_avg_reward)

    rewards_over_seeds_assisted.append(reward_over_episodes)

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
        obs_agent, info_agent = wrapped_agent_env.reset(seed=seed)

        done_agent = False
        while not done_agent:
            agent_action = agent.sample_action(obs_agent)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs_agent, reward_agent, terminated_agent, truncated_agent, info_agent = wrapped_agent_env.step(agent_action)
            agent.rewards.append(reward_agent)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done_agent = terminated_agent or truncated_agent

        reward_over_episodes.append(wrapped_agent_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_agent_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

# %%
# Plot learning curve
# ~~~~~~~~~~~~~~~~~~~
#


# Format data in the following format: <"episode", "type_of_agent", "reward", "seed"> 
no_of_seeds = len(rewards_over_seeds)
data = []
column_name = ["episode", "type_of_agent", "reward", "seed" ]
for seed in range(no_of_seeds):
    for episode_index, reward_agent in enumerate(rewards_over_seeds[seed]):
        data.append([episode_index, "policy_gradient", reward_agent[0], seed])

    for episode_index, reward_agent in enumerate(rewards_over_seeds_assisted[seed]):
        data.append([episode_index, "trajectory_injection", reward_agent[0], seed])
df = pd.DataFrame(data, columns=column_name)
df.to_csv("episodes_and_rewards.csv")

# Plot multiple line plot which compares agents.
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episode", y="reward", hue="type_of_agent", data=df).set(
    title="REINFORCE vs TrajectoryInjection"
)
plt.show()
