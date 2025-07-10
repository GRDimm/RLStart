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

### 1 - CartPole (`Cartpole.py`)
- Custom environment replicating the CartPole-v1 environment from gym.
- Objective: Move left and right to prevent the bar from falling

### 2 - Custom Lander (`CustomLander.py`)
- Custom environment replicating the LunarLander-v2 environment from gym.
- Objective: Land cleanly on a randomly positioned platform with a central engine and two side engines.

<p align="center">
  <img src="https://raw.githubusercontent.com/GRDimm/RLStart/main/images/CustomLander.gif" width="80%"/>
</p>

### 3 - Custom Explorer (`CustomExplorer.py`) (Still in dev, not working)
- Custom environment.
- Objective: Map the area, exploration rewards.

### 4 - Dice strategy (`DiceStrategy.py`)
- Custom environment.
- Objective: Find the best strategy in a dice rolling game.
- Game : You have a 10-faced dice, numbered from 1 to 10. Each time you throw the dice you have the option to exercise the right to receive an amount in dollars equal to the top face value, but exercising this right costs 5 dollars. If you don't exercise, then you can continue to roll the dice as many times as the subquestion allows you. In the end, you can exercise 0 times and you will pay 0 and get 0.
- Results : 

#### Model Strategy and Expected Reward

#### Strategy
- **For 10 rounds left**, the model exercises if the dice value >= 10
- **For 9 rounds left**, the model exercises if the dice value >= 10
- **For 8 rounds left**, the model exercises if the dice value >= 9
- **For 7 rounds left**, the model exercises if the dice value >= 9
- **For 6 rounds left**, the model exercises if the dice value >= 9
- **For 5 rounds left**, the model exercises if the dice value >= 9
- **For 4 rounds left**, the model exercises if the dice value >= 9
- **For 3 rounds left**, the model exercises if the dice value >= 8
- **For 2 rounds left**, the model exercises if the dice value >= 8
- **For 1 round left**, the model exercises if the dice value >= 6
- **For 0 rounds left**, the model exercises if the dice value >= 5

#### Expected Reward
Expected reward with the strategy: **4.1495**

### 8 - RL Algos from scratch
- Trying to code from scratch the different RL algorithms explained in [Deepmind's course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm).

#### Greedy

- Select the action with the best past average reward.

- Example with 3 dimensional action space.
<p align="center">
  <img src="https://github.com/GRDimm/RLStart/blob/main/8%20-%20RL%20algos%20from%20scratch/images/greedy_reward_distributions.png" width="80%"/>
</p>

- Greedy's output (highlight's Greedy's issue, it doesn't explore).

| Action | Average Reward |
|--------|----------------|
| 0      | -1.85          |
| 1      | 5.0            |
| 2      | 0.0            |

#### Epsilon-Greedy

- Like Greedy but adds an exploration probability of epsilon. Each step, select a random action with probability epsilon else pick the greedy action.

- In the same example as above with 3 dimensional action space.

- Epsilon Greedy's output explores so discovers that action 2 is more rewarding on average.

| Action | Average Reward |
|--------|----------------|
| 0      | 0.86           |
| 1      | 3.31           |
| 2      | 10.03          |

#### Policy gradient

- Gradient ascent on the model's parameters.

- In the same example as above with 3 dimensional action space.

- Works good on that example

| Action | Average Reward |
|--------|----------------|
| 0      | -0.32          |
| 1      | 5.0            |
| 2      | 10.18          |

#### UCB

- Greedy but with exploration bounds.

- In the same example as above with 3 dimensional action space.

| Action | Average Reward |
|--------|----------------|
| 0      | 0.28           |
| 1      | 5.15           |
| 2      | 10.11          |



