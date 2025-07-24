import random
import torch
from algo import StatelessFiniteActionSpaceAlgorithm, StatefulFiniteActionAlgorithm
from environment.environment import FiniteStateFiniteActionEnvironment, StatelessFiniteActionEnvironment
from utils.example_utils import cartpole_example, frozenlake_example, normal_n_armed_bandit_example

class EpsilonGreedy(StatelessFiniteActionSpaceAlgorithm):
    def __init__(self, environment: StatelessFiniteActionEnvironment, epsilon: float = 0.1):
        super().__init__(environment)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            # Explore: select a random action
            return random.randint(0, self.environment.actions_dimension - 1)
        
        # Exploit: select the action with the highest average reward
        return max(self.action_reward_average, key=self.action_reward_average.get) if self.action_reward_average else random.randint(0, self.environment.actions_dimension - 1)


class TableQLearningEpsilonGreedy(StatefulFiniteActionAlgorithm):
    def __init__(self, environment: FiniteStateFiniteActionEnvironment, epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995, alpha: float = 0.1, gamma: float = 0.9):
        super().__init__(environment)
        self.environment = environment
        self.state_dim = environment.state_dimension
        self.action_dim = environment.actions_dimension
        self.q_table = [[0.0 for _ in range(self.action_dim)] for _ in range(self.state_dim)]
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = self.environment.state()
        return int(torch.argmax(torch.tensor(self.q_table[state])).item())

    def update_algorithm_step(self, action: int, reward: float, previous_state: int, next_state: int, done: bool) -> None:
        q_next = 0 if done else max(self.q_table[next_state])
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.q_table[previous_state][action]
        self.q_table[previous_state][action] += self.alpha * td_error

    def update_algorithm_episode(self, total_steps: int, total_reward: float) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.stats["Epsilon"] = self.epsilon

    def reset(self) -> None:
        self.training_step = 0
        self.episode = 0
        self.environment.reset()

    def run_episode(self) -> bool:
        self.environment.reset()
        done = False
        while not done:
            self.environment.render()
            action = self.select_action()
            _, _, done, _ = self.environment.step(action)
        return True
        

if __name__ == "__main__":
    # # Epsilon-Greedy agent example
    # agent = normal_n_armed_bandit_example(EpsilonGreedy)
    # agent.train(n_episodes=100000, render=True)

    # Q-learning with Epsilon-Greedy agent example
    q_agent = frozenlake_example(TableQLearningEpsilonGreedy, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.999)
    q_agent.train(n_episodes=10000, render=True, render_every=10)