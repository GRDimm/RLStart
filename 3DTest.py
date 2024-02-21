import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

def draw_floor():
    glBegin(GL_QUADS)
    glColor3f(0.5, 0.5, 0.5)  # Couleur du sol
    glVertex3f(-25, -1, 25)
    glVertex3f(25, -1, 25)
    glVertex3f(25, -1, -25)
    glVertex3f(-25, -1, -25)
    glEnd()

def draw_cube():
    vertices = (
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 7), (3, 6)
    )
    faces = (
        (0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4),
        (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6)
    )

    glBegin(GL_QUADS)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    glColor3f(0, 0, 0)  # Change the color of the edges if you like
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


# Variables pour la gestion de la caméra
camera_angle_x, camera_angle_y = 0, 0
camera_pos = [0, 0, 5] # Position initiale de la caméra
# Initialisation de Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

# Configuration initiale d'OpenGL
# Configuration initiale d'OpenGL pour reculer la caméra
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5) # Peut-être augmenter la valeur en z pour reculer davantage

# Lors de la mise à jour de la caméra, ajuste la position pour s'assurer qu'elle ne démarre pas à l'intérieur d'un objet
glLoadIdentity()
# Position de la caméra ajustée pour s'assurer qu'elle regarde depuis un point hors des objets
gluLookAt(0,0,0, 0,0,-1, 0,1,0) # Cette ligne pourrait ne pas être nécessaire si tu ajustes uniquement avec glRotatef et glTranslatef
glRotatef(-camera_angle_y, 1, 0, 0)
glRotatef(-camera_angle_x, 0, 1, 0)
glTranslatef(0.0, 0.0, -5) # Ajuste cette valeur si nécessaire pour reculer la caméra


def main():
    global camera_angle_x, camera_angle_y, camera_pos
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        camera_angle_x += mouse_dx * 0.1
        camera_angle_y += mouse_dy * 0.1
        camera_angle_y = max(min(camera_angle_y, 90), -90)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        # Ajuste la position et l'orientation de la caméra
        glLoadIdentity()
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2], 
                  camera_pos[0] + math.sin(math.radians(camera_angle_x)), 
                  camera_pos[1] - math.tan(math.radians(camera_angle_y)), 
                  camera_pos[2] - math.cos(math.radians(camera_angle_x)), 
                  0, 1, 0)
        
        draw_cube()
        draw_floor()

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()