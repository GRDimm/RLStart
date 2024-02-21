from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import matplotlib.pyplot as plt
#!pip install gymnasium[classic-control]
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import math
import pygame
import math

from gymnasium import Env, spaces


def test(model, env):
    # Test de l'agent entraîné
    observation, _info = env.reset()
    done = False
    step = 0
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        env.render(mode='human')
        step += 1
    
    env.close()

class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render(mode='human')
        return True


class CustomCartPoleEnv(Env):
    """
    Description:
        Un poteau est attaché par une articulation non actionnée à un chariot, qui se déplace le long d'une piste sans frottement. L'objectif est de prévenir
        le poteau de tomber.
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Position du chariot       -2.4                    2.4
        1       Vitesse du chariot        -Inf                    Inf
        2       Angle du poteau           ~-41.8°                 ~41.8°
        3       Vitesse du sommet du poteau à l'angle -Inf        Inf
        
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Pousser le chariot vers la gauche
        1     Pousser le chariot vers la droite
        
        Note: L'ampleur de la force de mouvement est fixe
        
    Reward:
        Reward est de 1 pour chaque étape prise, y compris l'étape terminale.
    Starting State:
        Toutes les observations sont assignées une valeur uniforme aléatoire entre (-0.05, 0.05)
    Episode Termination:
        La position du chariot est plus de 2.4 (centre du chariot atteint la limite de l'écran)
        L'angle du poteau est plus de ±20.9°
        L'épisode dure plus de 200 étapes
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # en fait la moitié de la longueur du poteau
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # intervalle de temps entre les états
        self.kinematics_integrator = 'euler'

        # Angle à laquelle considérer l'échec, en radians
        self.theta_threshold_radians = 40 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Espace d'action et d'observation
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([-self.x_threshold, -np.finfo(np.float32).max, -self.theta_threshold_radians, -np.finfo(np.float32).max]).astype(np.float32),
                                            np.array([self.x_threshold, np.finfo(np.float32).max, self.theta_threshold_radians, np.finfo(np.float32).max]).astype(np.float32),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        np.random.seed(seed)
    
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100  # Top of cart
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.viewer.fill((255, 255, 255))
        cartx = self.state[0] * scale + screen_width / 2.0  # MIDDLE OF CART

        # Draw track
        pygame.draw.line(self.viewer, (0, 0, 0), (0, carty), (screen_width, carty), 1)

        # Draw cart
        cartrect = pygame.Rect(cartx - cartwidth / 2, carty - cartheight / 2, cartwidth, cartheight)
        pygame.draw.rect(self.viewer, (0, 0, 255), cartrect)

        # Calculate pole angle
        theta = self.state[2]  # Assuming this is the angle of the pole in radians

        # Correct pole drawing
        pole_end_x = cartx + polelen * math.sin(theta)
        pole_end_y = carty - cartheight / 2 - polelen * math.cos(theta)
        pygame.draw.line(self.viewer, (255, 0, 0), 
                        (int(cartx), int(carty - cartheight / 2)), 
                        (int(pole_end_x), int(pole_end_y)), 
                        int(polewidth))
        # Update the screen
        pygame.display.flip()
        self.clock.tick(165)  # Limit to 60 frames per second

    def step(self, action):
        err_msg = f"{action} invalid. Must be 0 (left) or 1 (right)."
        assert self.action_space.contains(action), err_msg
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Dynamique du système
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "Vous appelez 'step()' même si cela est censé être terminé. Vous devriez vérifier 'done' == True et appeler 'reset()' avant de continuer.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, False, {}

    def reset(self, **kwargs):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state), {}

    def close(self):
        if self.viewer is not None:
            pygame.display.quit()
            pygame.quit()
            self.viewer = None



# Création de l'environnement
env = CustomCartPoleEnv()

# Initialisation de l'agent
model = PPO('MlpPolicy', env, verbose=1)

test(model, env)

# Création et utilisation du callback
render_callback = RenderCallback() # Ajustez render_freq selon vos besoins

# Entraînement de l'agent avec le callback pour le rendu
model.learn(total_timesteps=30000, callback=render_callback)

# Sauvegarde du modèle
model.save("ppo_cartpole")

# Test du modèle après l'entraînement
test(model, env)
