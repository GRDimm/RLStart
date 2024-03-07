import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import pygame
import numpy as np
from Box2D import b2World, b2Vec2, b2_staticBody, b2_dynamicBody, b2PolygonShape, b2FixtureDef
from Box2D import b2ContactListener
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common import logger

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure

# Assurez-vous de configurer le logger si nécessaire
# Par exemple, pour écrire dans le dossier de logs courant et à la console
logger = configure(folder="./logs", format_strings=["stdout", "log"])

class MinMaxScaler:
    def __init__(self, min_voulu, max_voulu, min_init = float('inf'), max_init = float('-inf')):
        self.min_voulu = min_voulu
        self.max_voulu = max_voulu
        self.min_observed = min_init
        self.max_observed = max_init

    def update(self, value):
        """Mettre à jour les valeurs minimales et maximales observées."""
        self.min_observed = min(self.min_observed, value)
        self.max_observed = max(self.max_observed, value)

    def normalize(self, value):
        """Normaliser une valeur donnée en fonction des min et max observés."""
        if self.max_observed == self.min_observed:  # Éviter la division par zéro
            return self.min_voulu
        normalized_value = (value - self.min_observed) / (self.max_observed - self.min_observed) * (self.max_voulu - self.min_voulu) + self.min_voulu
        return normalized_value

class LanderContactListener(b2ContactListener):
    def __init__(self, lander, target):
        super(LanderContactListener, self).__init__()
        self.is_in_collision = False
        self.lander = lander
        self.target = target
    
    def check_collision(self, fixture_a, fixture_b, type_a, type_b):
        if fixture_a.userData != None and fixture_b.userData != None:
            return {fixture_a.userData.get("type"), fixture_b.userData.get("type")} == {type_a, type_b}
        else:
            return False

    def BeginContact(self, contact):
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        if self.check_collision(fixture_a, fixture_b, "lander", "target"):
            self.lander.on_target = True
            if fixture_a.userData["type"] == 'lander':
                self.lander.collision_speed = tuple(fixture_a.body.linearVelocity)
            else:
                self.lander.collision_speed = tuple(fixture_b.body.linearVelocity)
        elif self.check_collision(fixture_a, fixture_b, "lander", "edge"):
            self.lander.on_edge = True

    def EndContact(self, contact):
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB
        if self.check_collision(fixture_a, fixture_b, "lander", "target"):
            self.lander.on_target = False
        elif self.check_collision(fixture_a, fixture_b, "lander", "edge"):
            self.lander.on_edge = False

class RenderCallback(BaseCallback):
    def __init__(self, render_freq: int):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True

class RayCastCallback(Box2D.b2RayCastCallback):
    def __init__(self, lander, **kwargs):
        Box2D.b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.lander = lander

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body == self.lander:
            return -1
        self.fixture = fixture
        self.point = Box2D.b2Vec2(point)
        self.normal = Box2D.b2Vec2(normal)
        self.fraction = fraction
        return fraction

class CustomLunarLander(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(CustomLunarLander, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.n_rays = 0
        self.grid_size = (50, 50)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4 + 2 * self.n_rays + 2,), dtype=np.float32)
        self.window_width = 800
        self.window_height = 600
        self.world = world(gravity=(0, -10))
        self.timeout = np.inf
        self.mouse = False
        self.step_reward = 0
        self.lander = None
        self.target_body = None
        self.screen = None
        self.clock = pygame.time.Clock()
        self.lander_height = 1
        self.lander_width = 1.5
        self.num_envs = 1
        self.reactor_width = 2
        self.render_mode = 'human'
        self.timesteps = 0
        self.last_action = 0
        self.last_rewards = {"distance":0, "step":0, "total":0, "exploration":0}
        self.rewards = [-1, -1, -1, -1]
        self.target_length = 2
        self.target_height = 0.1
        self.target_x = 0
        self.ray_length = 5
        self.relative_position = (0, 0)
        self.target_y = 0
        self.rays = [self.ray_length]*self.n_rays
        self.binary_rays = [0]*self.n_rays
        self.rays_index = 4
        self.binary_rays_index = self.rays_index + self.n_rays
        self.mouse_control = False
        self.lander_color = (51, 51, 204)
        self.stagnation_streak = 0
        self.reactor_length = 1
        self.damping_factor = 0.5

        self.cell_size = (self.window_height/self.grid_size[1], self.window_width/self.grid_size[0])
        self.grid = np.array([[0] * self.grid_size[0] for _ in range(self.grid_size[1])])

        self.vx_index = 0
        self.vy_index = 1

        self.th_index = 2
        self.vth_index = 3

        self.scale_ratio = 20

        self.norms = {"velocity":None, "angle":None, "distance":None}

        self.unexplored_color = (100, 100, 100, 128)
        self.grid_color = (200, 200, 200)

        self.create_world_edges()

    def draw_grid(self):

        # Calcul de la taille de chaque cellule de la grille
        cell_height = self.window_height / self.grid_size[1]
        cell_width = self.window_width / self.grid_size[0]

        # Boucle sur les cellules de la grille pour les dessiner
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)
                if self.grid[y][x] == 0:  # Si la cellule n'est pas explorée
                    s = pygame.Surface((cell_width, cell_height), pygame.SRCALPHA)
                    s.fill(self.unexplored_color)  # Remplir avec la couleur plus foncée
                    self.screen.blit(s, (x * cell_width, y * cell_height))

        # Dessiner les lignes de la grille après avoir rempli les cellules
        for i in range(self.grid_size[0] + 1):  # Lignes verticales
            pygame.draw.line(self.screen, self.grid_color, (i * cell_width, 0), (i * cell_width, self.window_height))
        for j in range(self.grid_size[1] + 1):  # Lignes horizontales
            pygame.draw.line(self.screen, self.grid_color, (0, j * cell_height), (self.window_width, j * cell_height))


    def print_rays(self):
        print(f"Rays : {self.get_rays()}")
    
    def print_binary_rays(self):
        print(f"BRays : {self.get_binary_rays()}")

    def draw_rays(self, draw_cross_only=True):
        if self.n_rays > 0:
            angle_increment = (2 * np.pi) / self.n_rays
            cross_size = 5  # Taille de la croix en pixels

            for i in range(self.n_rays):
                angle = i * angle_increment
                corner_offset = self.calculate_corner_offset(angle)
                # Calculez la position de départ du rayon avec le décalage pour partir du coin du lander
                ray_start_box2d = np.array([self.get_x() + corner_offset[0], self.get_y() + corner_offset[1]])
                direction = np.array([np.cos(angle), np.sin(angle)])
                # Calcule le point final du rayon dans le système de coordonnées Box2D
                ray_end_box2d = ray_start_box2d + direction * self.rays[i]
                # Convertit le point de départ et le point final du rayon en coordonnées Pygame
                ray_start_px = self.box2d_to_pygame(ray_start_box2d[0], ray_start_box2d[1])
                ray_end_px = self.box2d_to_pygame(ray_end_box2d[0], ray_end_box2d[1])

                if draw_cross_only:
                    # Dessine une croix rouge au bout du rayon
                    pygame.draw.line(self.screen, (int(255 * (i/(self.n_rays-1))), 0, 0), (ray_end_px[0] - cross_size, ray_end_px[1]), (ray_end_px[0] + cross_size, ray_end_px[1]), 3)
                    pygame.draw.line(self.screen, (int(255 * (i/(self.n_rays-1))), 0, 0), (ray_end_px[0], ray_end_px[1] - cross_size), (ray_end_px[0], ray_end_px[1] + cross_size), 3)
                else:
                    # Dessine la ligne entière du rayon en bleu
                    pygame.draw.line(self.screen, (0, 0, int(255 * (i/self.n_rays))), ray_start_px, ray_end_px, 1)

    def box2d_to_pygame(self, x, y):

        # Centrer l'axe X en ajoutant la moitié de la largeur de la fenêtre, puis appliquer le scale_ratio
        x_px = (x * self.scale_ratio) + (self.window_width / 2)
 
        # Inverser l'axe Y car Pygame a l'origine en haut à gauche, puis appliquer le scale_ratio.
        # La soustraction par window_height n'est pas nécessaire ici si on part du bas.
        y_px = self.window_height - (y * self.scale_ratio)
        
        return int(x_px), int(y_px)
    
    def box2d_to_pygame_vect(self, position):
        """
        Convertit les coordonnées Box2D (position) en coordonnées Pygame.
        """
        x_px = (position[0] * self.scale_ratio) + (self.window_width / 2)
        y_px = self.window_height - (position[1] * self.scale_ratio)
        return int(x_px), int(y_px)

    def calculate_corner_offset(self, angle):
        """
        Calcule le décalage du point de départ du rayon par rapport au centre du lander
        pour s'assurer qu'il parte du coin du lander.
        """
        dx = np.cos(angle) * self.lander_width / 2
        dy = np.sin(angle) * self.lander_height / 2
        return (dx, dy)

    def create_world_edges(self):
        half_width_m = (self.window_width / 2) / self.scale_ratio  # Moitié de la largeur en mètres
        window_height_m = self.window_height / self.scale_ratio  # Hauteur en mètres
        # Crée un corps statique dans le monde pour les bords
        ground_body = self.world.CreateStaticBody(position=(0, 0))
        # Bord inférieur (étendu horizontalement de gauche à droite depuis le centre)
        fixture = ground_body.CreateEdgeFixture(vertices=[(-half_width_m, 0), (half_width_m, 0)])
        fixture.userData = {"type":"edge"}
        fixture = ground_body.CreateEdgeFixture(vertices=[(-half_width_m, 0), (-half_width_m, window_height_m)])
        fixture.userData = {"type":"edge"}
        fixture = ground_body.CreateEdgeFixture(vertices=[(half_width_m, 0), (half_width_m, window_height_m)])
        fixture.userData = {"type":"edge"}
        fixture = ground_body.CreateEdgeFixture(vertices=[(-half_width_m, window_height_m), (half_width_m, window_height_m)])
        fixture.userData = {"type":"edge"}

    def update_rays(self):
        if self.n_rays > 0:
            angle_increment = (2 * np.pi) / self.n_rays

            for i in range(self.n_rays):
                angle = i * angle_increment
                corner_offset = self.calculate_corner_offset(angle)
                ray_start = Box2D.b2Vec2(self.get_x() + corner_offset[0], self.get_y() + corner_offset[1])
                direction = Box2D.b2Vec2(np.cos(angle), np.sin(angle))
                ray_end = ray_start + self.ray_length * direction

                callback = RayCastCallback(self.lander)
                self.lander.world.RayCast(callback, ray_start, ray_end)

                self.grid_min_cell_size = min(self.cell_size)

                steps = int(self.ray_length * self.scale_ratio / self.grid_min_cell_size) + 1

                if callback.fixture:
                    fixture_distance = (callback.point - ray_start).length * self.scale_ratio
                else:
                    fixture_distance = self.ray_length * self.scale_ratio

                for step in range(steps):
                    step_distance = step * (self.grid_min_cell_size / self.scale_ratio)
        
                    if step_distance > fixture_distance:
                        break

                    point = ray_start + direction * (step * (self.grid_min_cell_size / self.scale_ratio))

                    grid_x = int((point.x * self.scale_ratio + self.window_width / 2) / self.cell_size[1])
                    grid_y = int((self.window_height - point.y * self.scale_ratio) / self.cell_size[0])

                    if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]:
                        self.grid[grid_y][grid_x] = 1

                distance = (callback.point - ray_start).length if callback.fixture else self.ray_length
                self.rays[i] = distance
                if callback.fixture:
                    if callback.fixture.userData and callback.fixture.userData["type"] == "target":
                        self.binary_rays[i] = 1
                    else:
                        self.binary_rays[i] = -1
                else:
                    self.binary_rays[i] = 0
    
    def get_right_corner(self):
        offset_y = self.lander_height/2 * np.cos(self.lander.angle)
        offset_x = - self.lander_height/2 * np.sin(self.lander.angle)

        lander_top_x = self.get_x() + offset_x
        lander_top_y = self.get_y() + offset_y

        lander_corner_x = lander_top_x + self.lander_width/2 * np.cos(self.lander.angle)
        lander_corner_y = lander_top_y + self.lander_width/2 * np.sin(self.lander.angle)
        return lander_corner_x, lander_corner_y

    def get_left_corner(self):
        offset_y = self.lander_height/2 * np.cos(self.lander.angle)
        offset_x = - self.lander_height/2 * np.sin(self.lander.angle)

        lander_top_x = self.get_x() + offset_x
        lander_top_y = self.get_y() + offset_y

        lander_corner_x = lander_top_x - self.lander_width/2 * np.cos(self.lander.angle)
        lander_corner_y = lander_top_y - self.lander_width/2 * np.sin(self.lander.angle)
        return lander_corner_x, lander_corner_y

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        
        self.screen.fill((255, 255, 255))
        
        for body in self.world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                
                if isinstance(shape, Box2D.b2PolygonShape):
                    # Pour les formes polygonales
                    vertices = [(body.transform * v) * self.scale_ratio for v in shape.vertices]
                    # Ajustement des coordonnées pour l'affichage Pygame
                    vertices = [(v[0] + self.window_width/2, self.window_height - v[1]) for v in vertices]
                    if fixture.userData != None and fixture.userData["type"] == "lander":
                        pygame.draw.polygon(self.screen, self.lander_color, vertices)
                    else:
                        pygame.draw.polygon(self.screen, (0, 0, 0), vertices)
                elif isinstance(shape, Box2D.b2CircleShape):
                    # Pour les formes circulaires
                    center = body.transform * shape.pos * self.scale_ratio
                    center = (center[0] + self.window_width/2, self.window_height - center[1])
                    radius = shape.radius * self.scale_ratio
                    pygame.draw.circle(self.screen, (0, 0, 0), center, radius)
                elif isinstance(shape, Box2D.b2EdgeShape):
                    vertex1 = self.box2d_to_pygame_vect(body.transform * shape.vertex1)
                    vertex2 = self.box2d_to_pygame_vect(body.transform * shape.vertex2)

                    # Dessine une ligne entre les deux vertices convertis
                    pygame.draw.line(self.screen, (0, 0, 0), vertex1, vertex2, 1)
                else:
                    vertices = [(body.transform * v) * 20 for v in shape.vertices]
                    vertices = [(v[0] + self.window_width/2, self.window_height - v[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, (0, 0, 0), vertices)
        
        if self.last_action == 3:
            lander_corner_x, lander_corner_y = self.get_right_corner()
            end_x = lander_corner_x + self.reactor_length * np.cos(self.lander.angle)
            end_y = lander_corner_y + self.reactor_length * np.sin(self.lander.angle)
            pygame.draw.line(self.screen, (255, 0, 0), self.box2d_to_pygame_vect((lander_corner_x, lander_corner_y)), self.box2d_to_pygame_vect((end_x, end_y)), self.reactor_width)

        if self.last_action == 2:

            lander_corner_x, lander_corner_y = self.get_left_corner()
            end_x = lander_corner_x - self.reactor_length * np.cos(self.lander.angle)
            end_y = lander_corner_y - self.reactor_length * np.sin(self.lander.angle)
            pygame.draw.line(self.screen, (255, 0, 0), self.box2d_to_pygame_vect((lander_corner_x, lander_corner_y)), self.box2d_to_pygame_vect((end_x, end_y)), self.reactor_width)

        if self.last_action == 1:

            start_x = self.get_x()
            start_y = self.get_y()

            end_x = start_x + self.reactor_length * np.sin(self.lander.angle)  # Utilisation de sin car l'angle est par rapport à la verticale
            end_y = start_y - self.reactor_length * np.cos(self.lander.angle)  # Utilisation de cos et addition car l'axe y est inversé dans Pygame

            pygame.draw.line(self.screen, (255, 0, 0), self.box2d_to_pygame_vect((start_x, start_y)), self.box2d_to_pygame_vect((end_x, end_y)), self.reactor_width)
        
        #self.draw_grid()
        self.draw_rays()
                
        pygame.display.flip()
        self.clock.tick(60)


    # Fonction pour convertir les coordonnées Box2D en coordonnées Pygame
    def box2d_to_pygame(self, x, y):
        return int(x * self.scale_ratio + self.window_width / 2), int(self.window_height - y * self.scale_ratio)

    def create_target(self):
        # Création du corps statique pour le rectangle
        target_body_def = Box2D.b2BodyDef()
        target_body_def.position = b2Vec2(self.target_x, self.target_y)
        target_body_def.type = b2_staticBody
        self.target_body = self.world.CreateBody(target_body_def)
        
        # Définition de la forme du rectangle
        target_shape = b2PolygonShape(box=(self.target_length / 2, self.target_height))
        
        # Création de la fixture
        target_fixture_def = b2FixtureDef(shape=target_shape)
        fixture = self.target_body.CreateFixture(target_fixture_def)
        fixture.userData = {"type": "target"}        

    def reset(self, **kwargs):
        if self.lander:
            self.world.DestroyBody(self.lander)
            self.lander = None
        
        if self.target_body:
            self.world.DestroyBody(self.target_body)
            self.target_body = None

        # Variabilité dans la position et l'angle initiaux
        initial_x = np.random.uniform(-0.75, 0.75) * self.window_width/2/self.scale_ratio
        initial_y = np.random.uniform(0.75, 0.95) * self.window_height/self.scale_ratio

        self.target_x = np.random.uniform(-0.75, 0.75) * self.window_width/2/self.scale_ratio
        self.target_y = np.random.uniform(0.1, 0.2) * self.window_height/self.scale_ratio
        self.create_target()

        initial_angle = np.random.uniform(-np.pi, np.pi)
        
        self.lander = self.world.CreateDynamicBody(position=(initial_x, initial_y), angle=initial_angle)
        fixture = self.lander.CreatePolygonFixture(box=(self.lander_width/2, self.lander_height/2), density=1/(self.lander_width * self.lander_height), friction=0.3)

        fixture.userData = {"type": "lander"}   
        self.lander.on_target = False
        self.lander.on_edge = False
        self.lander.collision_speed = (10, 10)

        # Création de l'écouteur de contact
        contact_listener = LanderContactListener(self.lander, self.target_body)
        # Attacher l'écouteur au monde Box2D
        self.world.contactListener = contact_listener

        self.norms = {"velocity":MinMaxScaler(0, 1, min_init=0), "angle":MinMaxScaler(0, 1, min_init=0, max_init=np.pi), "distance":MinMaxScaler(0, 1, min_init=0)}
        self.timesteps = 0
        self.last_action = 0
        self.stagnation_streak = 0
        self.last_rewards = {"distance":0, "step":0, "total":0, "exploration":0}
        self.rays = [self.ray_length]*self.n_rays
        self.binary_rays = [0]*self.n_rays
        self.grid = np.array([[0] * self.grid_size[0] for _ in range(self.grid_size[1])])
        self.state = np.array([0.0, 0.0, self.lander.angle, 0.0] + self.rays + self.binary_rays + [-1, -1], dtype=np.float32)

        return self.state, {}

    def get_exploration(self):
        return (f"Exploration : {(100*env.last_rewards['exploration']/(env.grid_size[0]*env.grid_size[1])):.2f} %")
    
    def calculate_distances_to_walls(self):

        center_x = self.window_width / 2
        center_y = self.window_height

        x = self.get_x()
        y = self.get_y()

        x_lander_in_window = x * self.scale_ratio + center_x
        y_lander_in_window = center_y - (y * self.scale_ratio) # Inversion car l'axe y est inversé dans de nombreux systèmes graphiques

        distance_to_left_wall = x_lander_in_window / self.scale_ratio
        distance_to_right_wall = (self.window_width - x_lander_in_window) / self.scale_ratio

        distance_to_bottom_wall = (self.window_height - y_lander_in_window) / self.scale_ratio
        distance_to_top_wall = y_lander_in_window / self.scale_ratio

        return {
            "left": distance_to_left_wall,
            "right": distance_to_right_wall,
            "top": distance_to_top_wall,
            "bottom": distance_to_bottom_wall
        }
    
    def get_vx(self):
        return self.lander.linearVelocity.x

    def get_vy(self):
        return self.lander.linearVelocity.y
    
    def get_v(self):
        return self.lander.linearVelocity

    def get_speed(self):
        vx = self.get_vx()
        vy = self.get_vy()
        return (vx**2 + vy**2)**0.5
    
    def get_vx(self):
        return self.state[self.vx_index]
    
    def get_angle(self):
        return self.state[self.th_index]

    def update_norms(self):
        
        th = self.get_angle()
        d = self.distance_to_target()
        v = self.get_speed()

        self.norms["angle"].update(abs(th))
        self.norms["velocity"].update(v)
        self.norms["distance"].update(d)

    def get_x(self):
        return self.lander.position.x

    def get_y(self):
        return self.lander.position.y

    def get_x_offset(self):
        x = self.get_x()
        return x - self.target_x

    def get_y_offset(self):
        y = self.get_y()
        return (y - self.lander_height/2) - (self.target_y + self.target_height/2)

    def distance_to_target(self):
        return (((self.get_x_offset())**2) + (self.get_y_offset())**2)**0.5

    def close(self):
        if self.screen is not None:
            pygame.quit()
    
    def print_state(self):
        x = self.get_x()
        y = self.get_y()
        th = self.get_angle()
        v = self.get_speed()
        d = self.distance_to_target()
        print(f'Action: {self.last_action}, Reward: {self.last_rewards}, (X, Y) : ({x:.2f}, {y:.2f}), V : {self.norms["velocity"].normalize(v):.4f}, D : {self.norms["distance"].normalize(d):.2f}, Th : {self.norms["angle"].normalize(abs(th)):.2f}, "Exp" : {100*env.last_rewards["exploration"]/(env.grid_size[0]*env.grid_size[1]):.2f}')

    def on_platform(self):
        x = self.get_x()
        y = self.get_y()
        return self.lander.on_target and y >= self.target_y + self.target_height/2

    def landing_speed(self):
        vx = self.lander.collision_speed[0]
        vy = self.lander.collision_speed[1]
        return (vx**2 + vy**2)**0.5 <= 3

    def apply_damping(self):
        # Application du damping pour réduire la vitesse angulaire
        if self.lander.angularVelocity != 0:
            damping_torque = -self.damping_factor * self.lander.angularVelocity
            self.lander.ApplyTorque(damping_torque, True)
        
        # Damping vitesse linéaire
        if self.mouse and self.mouse_control:
            velocity = self.get_v()
            damping_force = -0.25 * velocity * velocity.length - 0.8 * velocity
            self.lander.ApplyForceToCenter(damping_force, True)

    def step(self, action):
        if not self.lander:
            return np.zeros(self.observation_space.shape), 0.0, True, {}

        force = 20
        
        if action == 0:  # Ne rien faire
            pass
        elif action == 1:  # Moteur principal
            force_x = force * np.sin(self.lander.angle)
            force_y = force * np.cos(self.lander.angle)
            self.lander.ApplyForceToCenter(Box2D.b2Vec2(-force_x, force_y), True)
        elif action == 2: # Moteur latéral droit
            point_d_application = Box2D.b2Vec2(self.get_right_corner())
            self.lander.ApplyForce(self.lander.GetWorldVector(localVector=Box2D.b2Vec2(force/4, 0)), point_d_application, True)
        elif action == 3: # Moteur latéral gauche
            point_d_application = Box2D.b2Vec2(self.get_left_corner())
            self.lander.ApplyForce(self.lander.GetWorldVector(localVector=Box2D.b2Vec2(-force/4, 0)), point_d_application, True)

        self.apply_damping()

        self.last_action = action
        self.world.Step(1.0/30.0, 4, 2)

        velocity = self.get_v()
        angle = self.lander.angle
        angular_velocity = self.lander.angularVelocity

        self.update_rays()

        walls_distance = self.calculate_distances_to_walls()

        offsets = [self.get_x_offset(), self.get_y_offset()]

        state = np.array([velocity.x, velocity.y, angle, angular_velocity] + self.rays + self.binary_rays + offsets, dtype=np.float32)
        self.state = state

        self.update_norms()

        reward = self.running_reward_system()

        done = False

        if self.lander.on_edge or self.lander.on_target:
            done = True
            reward += self.end_reward_system()

        truncated = False
        
        
        if self.timesteps >= self.timeout:
            #print("Timeout : ", position.x, position.y, abs(velocity.x), abs(velocity.y))
            print("TIMEOUT", self.get_exploration())
            reward += -5 #+ 25 * self.last_rewards["exploration"]/(self.grid_size[0] * self.grid_size[1])
            truncated = True
            done = True
        else:
            truncated = False
        
        
        action_reward = 0

        if action != 0:
            action_reward += -0.1

        self.timesteps += 1

        #reward += action_reward

        self.last_rewards["total"] = reward

        return state, reward, done, truncated, {}

    def get_rays(self):
        return self.rays
    
    def get_binary_rays(self):
        return self.binary_rays

    def running_reward_system(self):

        th = self.get_angle()
        v = self.get_speed()
        d = self.distance_to_target()

        # Coefficients pour pondérer l'importance de chaque aspect
        position_weight = -1  # Plus proche de ([-1, 1], 0) est mieux
        velocity_weight = -1   # Plus faible est mieux, surtout près du point de cible
        angle_weight = -1     # Plus proche de 0 (verticale) est mieux

        # STABILISATION
        angle_reward = angle_weight * self.norms["angle"].normalize(abs(th))
        velocity_reward = velocity_weight * self.norms["velocity"].normalize(v)

        # OBJECTIF (atterrir au bon endroit)
        position_reward = -self.norms["distance"].normalize(d)
        #print(position_reward, velocity_reward, angle_reward)

        touched_rays = self.get_rays()

        
        if self.n_rays > 0:
            binary_rays_reward = -sum(self.get_binary_rays())/self.n_rays
            rays_reward = - (self.ray_length*self.n_rays - sum(self.get_rays()))/(self.n_rays*self.ray_length)

        distance_diff_reward = (position_reward - self.last_rewards["distance"])

        grid_sum = np.sum(self.grid)
        exploration_diff_reward = (grid_sum - self.last_rewards["exploration"])/(self.grid_size[0] * self.grid_size[1])

        stagnation_reward = 0
        binary_exploration_reward = 0

        if exploration_diff_reward == 0:
            self.stagnation_streak += 1
        else:
            self.stagnation_streak = 0
            binary_exploration_reward += 1
        
        if self.stagnation_streak >= 5:
            #stagnation_reward += -0.1
            self.stagnation_streak = 0

        self.last_rewards["distance"] = position_reward
        self.last_rewards["step"] = distance_diff_reward
        self.last_rewards["exploration"] = grid_sum
        return distance_diff_reward*10

    def end_reward_system(self):

        reward = 0
        
        if self.on_platform() and self.landing_speed():
            reward += 25
            print("GGGGGGGGGGGGGGG", self.get_exploration())
        else:
            if self.lander.on_target:
                reward += -20
            else:
                reward += -20
            print("CRASH", self.get_exploration())

        return reward

    def convert_pygame_to_box2d(self, x, y):
        box2d_x = (x - self.window_width / 2) / self.scale_ratio
        box2d_y = (self.window_height - y) / self.scale_ratio
        return (box2d_x, box2d_y)

    def move_to_mouse(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.mouse = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.mouse = False
            
        if self.mouse:
            strength = 50
            mouse_x, mouse_y = pygame.mouse.get_pos()  # Position de la souris à l'écran
            # Convertir la position de la souris en coordonnées du monde Box2D ici
            mouse_world_pos = self.convert_pygame_to_box2d(mouse_x, mouse_y)
            force_direction = b2Vec2(mouse_world_pos) - self.lander.position
            force_direction.Normalize()  # Normalise le vecteur pour obtenir la direction

            # Application de la force multipliée par le facteur de force `strength`
            force = force_direction * strength
            self.lander.ApplyForce(force, self.lander.worldCenter, True)

""" 
models_dir = "./models/"
model_name = "LLCustom"

import time
use_last_trained = False

env = CustomLunarLander()

env.reset()

if use_last_trained:
    model = PPO.load(models_dir  + model_name)
else:
    model = PPO("MlpPolicy", env, verbose=2)

    #model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=2)
    # Créer une instance de callback pour le rendu toutes les 1000 étapes
    #render_callback = RenderCallback(render_freq=1000)

    # Entraîner le modèle avec le callback
    model.learn(total_timesteps=500000)

    model.save(models_dir + model_name)

while True:
    print("START")
    env = CustomLunarLander()
    observation, _info = env.reset()  # Capturez l'observation et ignorez le dictionnaire info
    timesteps = 0
    done = False
    while not done:
        #env.print_rays()
        #print([round(num, 2) for num in env.rewards], round(sum(env.rewards), 2))
        env.print_state()
        action, _states = model.predict(observation, deterministic=True) 
        observation, reward, done, truncated, info = env.step(action)  # Appliquer l'action
        env.render()
        # Assurez-vous que l'indice d'observation que vous essayez d'imprimer existe
        timesteps+=1
    time.sleep(1)
    env.print_state()
    
env.close()

"""

env = CustomLunarLander()

obs = env.reset()
env.mouse_control = True
env.render()

done = False
while not done:
    #env.print_binary_rays()
    #env.print_state()
    #print(np.array(env.grid))
    #print(env.last_rewards['exploration']/(env.grid_size[0] * env.grid_size[1]))
    env.render()
    keys = pygame.key.get_pressed()
    action = 0
    if keys[pygame.K_RIGHT]:
        action = 2
    if keys[pygame.K_LEFT]:
        action = 3
    if keys[pygame.K_UP]:
        action = 1


    env.move_to_mouse()
    

    observation, reward, done, truncated, info = env.step(action)  # Appliquer l'action
    
    pygame.event.pump()

env.close()
    



