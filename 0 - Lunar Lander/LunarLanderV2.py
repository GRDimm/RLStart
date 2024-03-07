import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, render_freq: int):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True
    
env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42)
model = PPO("MlpPolicy", env, verbose=2)
#model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Créer une instance de callback pour le rendu toutes les 1000 étapes
render_callback = RenderCallback(render_freq=1000)

# Entraîner le modèle avec le callback
model.learn(total_timesteps=200000, callback=render_callback)

model.save('500KLLV2')

while True:
    # Créez un nouvel environnement avec render_mode pour l'évaluation
    env = gym.make("LunarLander-v2", render_mode='human')

    observation, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation, deterministic=True)  # Assurez-vous que le modèle est déjà entraîné
        observation, reward, done, truncated, info = env.step(action)
        env.render()  # Cela devrait fonctionner sans avertissement si `render_mode` est spécifié
    env.close()