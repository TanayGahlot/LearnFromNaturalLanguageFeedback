# Reinforcement Learning with Natural Language Feedback

This repository contains the implementation of a Reinforcement Learning (RL) framework that incorporates natural language feedback alongside numerical rewards to improve convergence in language model fine-tuning.

## Overview

Recent advancements in fine-tuning large language models have shown promising results. However, these methods are resource-intensive and often rely on numerical feedback alone, ignoring valuable natural language feedback. In this project, we propose two novel techniques to integrate natural language feedback into an existing policy gradient setup for more efficient training:

1. **Joint Objective**: A combination of policy gradient and imitation learning to achieve a 1.5x speedup.
2. **Trajectory Injection**: Introduces expert trajectories during training, leading to a 2x improvement in convergence rate.

### Problem Statement

Traditional RL-based methods primarily focus on numerical feedback, but in many real-world applications, human users provide free-form textual feedback. We incorporate the following types of feedback:

- **Numerical Feedback**: A score indicating satisfaction with the response.
- **Textual Feedback**: Explanation of what was incorrect.
- **Expected Response**: The desired or correct response provided by the user.

### Example

For instance, consider the following conversation:

**User**: _"What are the best food sources for fiber in an American diet?"_

**Bot**: _"Raspberries, pears, apples, green peas, and broccoli are known for having high fiber."_

**User**: _"That’s helpful, but can you provide a source for that information?"_

**Bot**: _"I don’t have a source, but many fruits and vegetables are high in fiber."_

Here, the user’s feedback would be:

- **Numerical Feedback**: 0 (Unsatisfactory)
- **Textual Feedback**: "Please provide a source, like Mayo Clinic or USDA."
- **Expected Response**: "As per the USDA Nutrient Database, fruits such as raspberries, pears, and apples, and vegetables like green peas and broccoli are high in fiber."

## Key Techniques

### 1. Language Generation as a Markov Decision Process (MDP)

We model language generation as an MDP where the RL agent (a language model) interacts with the environment (user) to generate responses and receive feedback.

### 2. Incorporating Human Feedback

We extend the traditional RL framework by incorporating human feedback:

- **Numerical Feedback**: Incorporated using the policy gradient method.
- **Natural Language Feedback**: Textual feedback is incorporated using imitation learning techniques.

### 3. Joint Loss

The RL objective is combined with an imitation learning objective (minimizing KL divergence between expert actions and policy actions) to improve training efficiency.

### 4. Trajectory Injection

Expert trajectories are injected during training, where expert actions are introduced partway through the episode, helping the model learn from demonstrations.

## Evaluation

Our proposed techniques were evaluated using the Mujoco InvertedPendulum environment. The results demonstrated that:

- **Joint Objective**: 1.5x speedup in convergence compared to REINFORCE.
- **Trajectory Injection**: 2x speedup in convergence compared to REINFORCE.
