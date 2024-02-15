import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import pygame
import numpy as np

class RenderCallback(BaseCallback):
    def __init__(self, render_freq: int):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True

class CustomLunarLander(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(CustomLunarLander, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        
        self.world = world(gravity=(0, -10))
        self.lander = None
        self.screen = None
        self.clock = pygame.time.Clock()
        self.start_epsilon = 0.5
        self.epsilon_decay = 0.025
        self.timesteps = 0
        self.lastaction = 0

    def reward_system(self, position, velocity, angle, angular_velocity):
        # Coefficients pour pondérer l'importance de chaque aspect
        position_weight = -450.0  # Plus proche de ([-1, 1], 0) est mieux
        velocity_weight = -110.0   # Plus faible est mieux, surtout près du point de cible
        angle_weight = -30.0      # Plus proche de 0 (verticale) est mieux
        angular_velocity_weight = -10.0  # Plus faible est mieux, indique une stabilisation

        # Calculer la distance au point de cible ([-1, 1], 0)
        target_x = 0.0  # Le centre de l'intervalle [-1, 1] pour simplifier
        distance_to_target = ((position.x - target_x) ** 2 + position.y ** 2) ** 0.5

        # Récompense pour la position
        position_reward = position_weight * distance_to_target

        # Récompense pour la vitesse
        velocity_magnitude = (velocity.x ** 2 + velocity.y ** 2) ** 0.5 
        velocity_reward = velocity_weight * velocity_magnitude

        # Récompense pour l'angle
        angle_reward = angle_weight * abs(angle)

        # Récompense pour la vitesse angulaire
        angular_velocity_reward = angular_velocity_weight * abs(angular_velocity)

        # Somme des récompenses pour le score total
        total_reward = position_reward + velocity_reward + angle_reward + angular_velocity_reward

        return total_reward

    def step(self, action):
        if not self.lander:  # Assurez-vous que le lander est initialisé
            return np.zeros(self.observation_space.shape), 0.0, True, {}

        force = 20 # Varier la force appliquée
        damping_factor = 0.5

        if action == 0:  # Ne rien faire
            pass
        elif action == 1:  # Moteur principal
            force_x = force * np.sin(self.lander.angle)
            force_y = force * np.cos(self.lander.angle)
            self.lander.ApplyForceToCenter(Box2D.b2Vec2(-force_x, force_y), True)
        elif action == 2:  # Moteur latéral gauche
            # Pour générer un couple, appliquer la force à un point décalé latéralement et non directement au centre
            point_d_application = Box2D.b2Vec2(-0.5, 0.5)  # Position relative par rapport au centre de gravité, ajustez selon la configuration de votre lander
            self.lander.ApplyForce(self.lander.GetWorldVector(localVector=Box2D.b2Vec2(force/4, 0)), self.lander.GetWorldPoint(localPoint=point_d_application), True)
        elif action == 3:  # Moteur latéral droit
            point_d_application = Box2D.b2Vec2(0.5, 0.5)  # Position relative par rapport au centre de gravité, ajustez selon la configuration de votre lander
            self.lander.ApplyForce(self.lander.GetWorldVector(localVector=Box2D.b2Vec2(-force/4, 0)), self.lander.GetWorldPoint(localPoint=point_d_application), True)

        # Application du damping pour réduire la vitesse angulaire
        if self.lander.angularVelocity != 0:
            damping_torque = -damping_factor * self.lander.angularVelocity
            self.lander.ApplyTorque(damping_torque, True)

        self.lastaction = action
        self.world.Step(1.0/30.0, 4, 1)

        position = self.lander.position
        velocity = self.lander.linearVelocity
        angle = self.lander.angle
        angular_velocity = self.lander.angularVelocity
        state = np.array([position.x, position.y, velocity.x, velocity.y, angle, angular_velocity], dtype=np.float32)

        reward = self.reward_system(position, velocity, angle, angular_velocity)
        self.timesteps += 1
        done = False
        truncated = False

        if abs(position.x) > 15 or position.y <= 0 or position.y >= 20:
            done = True
            if position.y <= 0 and abs(position.x) <= 1 and abs(velocity.y) < 4 and abs(velocity.x) < 4:
                reward += 1000000
                print("GGGGGGGGGGGGG")
            else:
                if position.y > 0:
                    reward += - 5 * (((position.x) ** 2 + position.y ** 2) ** 0.5)
                else:
                    reward += - (((position.x) ** 2 + position.y ** 2) ** 0.5)
                #print("CRASH")

        if self.timesteps >= 600:
            #print("Timeout : ", position.x, position.y, abs(velocity.x), abs(velocity.y))
            reward -= 10000
            truncated = True
            done = True
        else:
            truncated = False


        return state, reward, done, truncated, {}

    def reset(self, **kwargs):
        if self.lander:
            self.world.DestroyBody(self.lander)
            self.lander = None

        # Variabilité dans la position et l'angle initiaux
        initial_x = np.random.uniform(-8, -6) if np.random.uniform(0, 1) > 0.5 else np.random.uniform(7, 9)
        initial_y = np.random.uniform(14, 19)
        initial_angle = np.random.uniform(-np.pi/2, np.pi/2)

        self.lander = self.world.CreateDynamicBody(position=(initial_x, initial_y), angle=initial_angle)
        self.lander.CreatePolygonFixture(box=(0.5, 0.5), density=1, friction=0.3)

        self.timesteps = 0

        self.state = np.array([self.lander.position.x, self.lander.position.y, 0.0, 0.0, self.lander.angle, 0.0], dtype=np.float32)
        return self.state, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 400))
        
        self.screen.fill((255, 255, 255))
        
        for body in self.world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * 20 for v in shape.vertices]
                vertices = [(v[0] + 300, 400 - v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, (0, 0, 0), vertices)
        
        bar_width = 2  # Largeur des barres en pixels
        bar_height = 40  # Hauteur des barres en pixels
        bar_color = (0, 0, 0)  # Couleur des barres (noir)

        # Dessiner les barres verticales en -1 et en 1
        # Conversion des positions en coordonnées Pygame, en considérant le scaling et le déplacement
        left_bar_x = 300 + (-1 * 20) - (bar_width / 2)
        right_bar_x = 300 + (1 * 20) - (bar_width / 2)
        bar_top = bar_height  # Juste un exemple, ajustez selon la position verticale souhaitée
        bar_bottom = bar_top + bar_height

        pygame.draw.line(self.screen, bar_color, (left_bar_x, 400), (left_bar_x, 400 - bar_height), bar_width)
        pygame.draw.line(self.screen, bar_color, (right_bar_x, 400), (right_bar_x, 400 - bar_height), bar_width)

        line_length = 1  # Longueur désirée de la ligne

        if self.lastaction == 3:
            # Calculer le décalage initial basé sur l'angle du lander
            offset_x = 0.5 * np.cos(self.lander.angle)  # Décalage à droite
            offset_y = 0.5 * np.sin(self.lander.angle)  # Décalage vers le haut

            # Point de départ ajusté avec le décalage
            start_x = self.lander.position.x + offset_x
            start_y = self.lander.position.y + offset_y

            # Calculer le point de fin basé sur l'angle ajusté et le point de départ ajusté
            end_x = start_x + line_length * np.cos(self.lander.angle)
            end_y = start_y + line_length * np.sin(self.lander.angle)

            # Convertir la position de départ et le point de fin en coordonnées Pygame
            start_x_pygame = 300 + (start_x * 20)  # Ajustez le facteur de mise à l'échelle si nécessaire
            start_y_pygame = 400 - (start_y * 20)  # Ajustez le facteur de mise à l'échelle si nécessaire
            end_x_pygame = 300 + (end_x * 20)
            end_y_pygame = 400 - (end_y * 20)

            # Dessiner la ligne
            pygame.draw.line(self.screen, (255, 0, 0), (start_x_pygame, start_y_pygame), (end_x_pygame, end_y_pygame), bar_width)

        if self.lastaction == 2:

            offset_angle = self.lander.angle + np.pi
            offset_x = 0.5 * np.cos(offset_angle)  # Décalage à gauche perpendiculaire à l'angle du lander
            offset_y = 0.5 * np.sin(offset_angle)  # Décalage vers le haut perpendiculaire à l'angle du lander

            # Point de départ ajusté avec le décalage perpendiculaire
            start_x = self.lander.position.x + offset_x
            start_y = self.lander.position.y + offset_y

            # Calculer le point de fin basé sur l'angle ajusté (perpendiculaire) et le point de départ ajusté
            # Utiliser le même offset_angle pour que la ligne soit dessinée perpendiculairement vers la gauche
            end_x = start_x + line_length * np.cos(offset_angle)
            end_y = start_y + line_length * np.sin(offset_angle)

            # Convertir la position de départ et le point de fin en coordonnées Pygame
            start_x_pygame = 300 + (start_x * 20)  # Ajustez le facteur de mise à l'échelle si nécessaire
            start_y_pygame = 400 - (start_y * 20)  # Ajustez le facteur de mise à l'échelle si nécessaire
            end_x_pygame = 300 + (end_x * 20)
            end_y_pygame = 400 - (end_y * 20)

            # Dessiner la ligne
            pygame.draw.line(self.screen, (255, 0, 0), (start_x_pygame, start_y_pygame), (end_x_pygame, end_y_pygame), bar_width)

        if self.lastaction == 1:

            line_length = 1  # Longueur désirée de la ligne

            # Le point de départ au "bas" du lander, ici on le prend comme le centre pour simplifier
            start_x = self.lander.position.x
            start_y = self.lander.position.y

            # Ajuster l'angle pour le rendu Pygame où l'axe y est inversé
            # Si l'angle augmente dans le sens horaire par rapport à la verticale vers le haut, 
            # on l'utilise tel quel puisque Pygame dessine vers le bas pour un angle positif
            end_x = start_x + line_length * np.sin(self.lander.angle)  # Utilisation de sin car l'angle est par rapport à la verticale
            end_y = start_y - line_length * np.cos(self.lander.angle)  # Utilisation de cos et addition car l'axe y est inversé dans Pygame

            # Convertir la position de départ et le point de fin en coordonnées Pygame
            # Ajustez le facteur de mise à l'échelle et les décalages pour correspondre à votre échelle et origine
            start_x_pygame = 300 + (start_x * 20)
            start_y_pygame = 400 - (start_y * 20)
            end_x_pygame = 300 + (end_x * 20)
            end_y_pygame = 400 - (end_y * 20)

            # Dessiner la ligne
            pygame.draw.line(self.screen, (255, 0, 0), (start_x_pygame, start_y_pygame), (end_x_pygame, end_y_pygame), bar_width)

        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()

env = CustomLunarLander()
env.reset()

#model = PPO("MlpPolicy", env, verbose=2)

model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=2)
# Créer une instance de callback pour le rendu toutes les 1000 étapes
render_callback = RenderCallback(render_freq=1000)

# Entraîner le modèle avec le callback
model.learn(total_timesteps=180000, callback=render_callback)

model.save('LLCustom')

while True:
    env = CustomLunarLander()
    observation, _info = env.reset()  # Capturez l'observation et ignorez le dictionnaire info

    done = False
    while not done:
        action, _states = model.predict(observation, deterministic=True) 
        observation, reward, done, truncated, info = env.step(action)  # Appliquer l'action
        env.render()
        # Assurez-vous que l'indice d'observation que vous essayez d'imprimer existe
    print(f"Action: {action}, Reward: {reward}, Position : {observation[0], observation[1]}")
    
    env.close()

"""
# Utilisation de l'environnement personnalisé
env = CustomLunarLander()
obs = env.reset()
env.render()

timestep = 0
done = False
while not done:
    keys = pygame.key.get_pressed()
    action = 0
    if keys[pygame.K_RIGHT]:
        action = 2
    if keys[pygame.K_LEFT]:
        action = 3
    if keys[pygame.K_UP]:
        action = 1

    observation, reward, done, info, _ = env.step(action)  # Appliquer l'action
    env.render()
    print(f"Action: {action}, Reward: {reward}, Done: {done}, Fuel : {observation[1]}")  # Afficher le résultat de l'action
    timestep += 1
    pygame.event.pump()

env.close()
"""
