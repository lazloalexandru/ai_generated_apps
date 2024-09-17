import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random

# Initialize Pygame and OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(75, (display[0] / display[1]), 0.1, 1000.0)
glTranslatef(0.0, 0.0, -15)

# Set the background color to white
glClearColor(1.0, 1.0, 1.0, 1.0)

# Enable lighting
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [10, 10, 10, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1])

# Enable depth testing
glEnable(GL_DEPTH_TEST)

# Gravity constant (downward force)
gravity = np.array([0, -0.01, 0])

# Function to create a box
def create_box():
    vertices = [
        [5, 5, -5], [5, -5, -5], [-5, -5, -5], [-5, 5, -5],
        [5, 5, 5], [5, -5, 5], [-5, -5, 5], [-5, 5, 5]
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    return vertices, edges

# Function to draw the box
def draw_box(vertices, edges):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# Function to create a ball
def create_ball():
    ball = gluNewQuadric()
    return ball

# Function to draw the ball with a specific color
def draw_ball(ball, position, radius, color):
    glPushMatrix()
    glTranslatef(*position)
    # Set material properties for the ball
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [color[0], color[1], color[2], 1.0])  # Ball color
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])  # White specular highlights
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)  # Shininess
    gluSphere(ball, radius, 32, 32)
    glPopMatrix()

# Function to initialize balls with no intersection and random colors
def initialize_balls(num_balls):
    balls = []
    ball_velocities = []
    ball_radii = []
    ball_masses = []
    ball_colors = []

    for _ in range(num_balls):
        while True:
            # Generate a new ball position and radius
            new_position = np.array([(random.random() - 0.5) * 8, (random.random() - 0.5) * 8, (random.random() - 0.5) * 8])
            new_radius = random.uniform(0.2, 0.5)
            no_intersection = True

            # Check for intersection with existing balls
            for i in range(len(balls)):
                distance = np.linalg.norm(new_position - balls[i])
                if distance < (new_radius + ball_radii[i]):
                    no_intersection = False
                    break

            # If no intersection, add the ball
            if no_intersection:
                balls.append(new_position)
                ball_radii.append(new_radius)
                ball_masses.append(new_radius ** 3)  # Mass proportional to volume
                # ball_velocities.append(np.array([(random.random() - 0.5) * 0.1, (random.random() - 0.5) * 0.1, (random.random() - 0.5) * 0.1]))
                ball_velocities.append(np.array([0.0, 0.0, 0.0]))
                # Generate random color for the ball
                ball_colors.append([random.random(), random.random(), random.random()])
                break

    return balls, ball_velocities, ball_radii, ball_masses, ball_colors

# Initialize box and balls
box_vertices, box_edges = create_box()
num_balls = 100  # User can specify the number of balls here
balls, ball_velocities, ball_radii, ball_masses, ball_colors = initialize_balls(num_balls)
ball = create_ball()

# Track the cube's rotation
cube_rotation = [0, 0]

# Global rotation matrix
cube_rotation_matrix = np.identity(4)

def update_rotation_matrix():
    global cube_rotation_matrix
    angle_x, angle_y = cube_rotation
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)

    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, cos_x, -sin_x, 0],
        [0, sin_x, cos_x, 0],
        [0, 0, 0, 1]
    ])

    rotation_y = np.array([
        [cos_y, 0, sin_y, 0],
        [0, 1, 0, 0],
        [-sin_y, 0, cos_y, 0],
        [0, 0, 0, 1]
    ])

    cube_rotation_matrix = np.dot(rotation_y, rotation_x)

# Function to handle ball collisions with realistic bounce
def handle_ball_collisions():
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            distance = np.linalg.norm(balls[i] - balls[j])
            if distance < (ball_radii[i] + ball_radii[j]):
                # Calculate mass of both balls
                m1, m2 = ball_masses[i], ball_masses[j]
                normal = (balls[i] - balls[j]) / distance
                relative_velocity = ball_velocities[i] - ball_velocities[j]
                speed = np.dot(relative_velocity, normal)

                # Skip if balls are moving apart
                if speed >= 0:
                    continue

                # Apply impulse based on mass and speed
                impulse = (2 * speed) / (m1 + m2)
                ball_velocities[i] -= impulse * m2 * normal
                ball_velocities[j] += impulse * m1 * normal

                # Adjust positions to prevent overlap
                overlap = (ball_radii[i] + ball_radii[j]) - distance
                balls[i] += normal * (overlap / 2)
                balls[j] -= normal * (overlap / 2)

# Function to apply gravity to the balls
def apply_gravity():
    for i in range(len(balls)):
        ball_velocities[i] += gravity  # Gravity acts on the y-axis (downward)

# Function to check and correct ball positions
def correct_ball_positions():
    global cube_rotation_matrix
    inverse_rotation_matrix = np.linalg.inv(cube_rotation_matrix[:3, :3])

    for i in range(len(balls)):
        # Transform ball position to box's local coordinate system
        local_position = np.dot(inverse_rotation_matrix, balls[i])

        # Check for collisions with the box walls in local space (including the floor for gravity)
        for j in range(3):
            if local_position[j] - ball_radii[i] < -5:
                ball_velocities[i][j] = -ball_velocities[i][j] * 0.8  # Reverse velocity (bounce), reduce by 20% for energy loss
                local_position[j] = -5 + ball_radii[i]
            elif local_position[j] + ball_radii[i] > 5:
                ball_velocities[i][j] = -ball_velocities[i][j] * 0.8  # Bounce off ceiling/walls
                local_position[j] = 5 - ball_radii[i]

        # Transform the local position back to world coordinates
        balls[i] = np.dot(cube_rotation_matrix[:3, :3], local_position)

# Function to ensure no ball overlaps after movement
def resolve_overlaps():
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            distance = np.linalg.norm(balls[i] - balls[j])
            if distance < (ball_radii[i] + ball_radii[j]):
                overlap = (ball_radii[i] + ball_radii[j]) - distance
                normal = (balls[i] - balls[j]) / distance
                balls[i] += normal * (overlap / 2)
                balls[j] -= normal * (overlap / 2)

# Function to draw the cube and apply the rotation matrix
def draw_rotated_cube():
    glPushMatrix()
    glMultMatrixf(cube_rotation_matrix.T.flatten())
    draw_box(box_vertices, box_edges)
    glPopMatrix()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Update rotation angles and matrix
    cube_rotation[0] += 0.01
    cube_rotation[1] += 0.01
    update_rotation_matrix()

    # Draw the cube with updated rotation
    draw_rotated_cube()

    # Apply gravity to the balls
    apply_gravity()

    # Update ball positions
    for i in range(len(balls)):
        balls[i] += ball_velocities[i]

    # Correct ball positions to ensure they stay within the cube
    correct_ball_positions()

    # Handle ball collisions with realistic mass-based bounce
    handle_ball_collisions()

    # Resolve overlaps after movement
    resolve_overlaps()

    # Draw balls
    for i, ball_position in enumerate(balls):
        draw_ball(ball, ball_position, ball_radii[i], ball_colors[i])

    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
