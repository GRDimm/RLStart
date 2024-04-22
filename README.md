# RLStart

## Reinforcement Learning Projects Repository

This repository contains small reinforcement learning projects that I develop alone to learn and explore different techniques and algorithms in the field.

## Libraries Used
- `gymnasium`: For creating and manipulating reinforcement learning environments.
- `stable_baselines3`: A library offering quality implementations of various reinforcement learning algorithms.
- `box2d`: Used for certain environments requiring physics, such as Lunar Lander.

## Concepts

### Using PPO
Implementation of the Proximal Policy Optimization (PPO) algorithm on different Gymnasium environments.

### Using DQN
Implementation of Deep Q-Network (DQN) to learn optimal strategies in classic environments.

### Creating a custom Lunar Lander environment
Development and training of a model in a custom Lunar Lander environment, aiming to test and improve environment design skills.

## Projects/Files

### `Cartpole.py` 
- Custom environment replicating the CartPole-v1 environment from gym.
- Objective: Move left and right to prevent the bar from falling

### `CustomLander.py` 
- Custom environment replicating the LunarLander-v2 environment from gym.
- Objective: Land cleanly on a randomly positioned platform with a central engine and two side engines.

<p align="center">
  <img src="https://raw.githubusercontent.com/GRDimm/RLStart/main/images/CustomLander.gif" width="80%" height="80%" />
</p>

### `CustomExplorer.py` 
- Custom environment.
- Objective: Map the area, exploration rewards.
