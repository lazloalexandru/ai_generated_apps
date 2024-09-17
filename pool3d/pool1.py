from vpython import *
import numpy as np

# Define custom colors
brown = vector(0.59, 0.29, 0.0)  # RGB values for brown

# Create the scene
scene = canvas(title='3D Pool Game', width=800, height=600)
scene.lights = []
distant_light(direction=vector(0, -1, 0), color=color.white)
distant_light(direction=vector(0, 0, -1), color=color.gray(0.5))

# Adjust initial camera view
scene.forward = vector(0, -0.5, -1).norm()
scene.up = vector(0, 1, 0)

# Table dimensions
table_length = 10
table_width = 5
table_height = 0.1  # Reduced table height for a thinner table surface
border_thickness = 0.2
border_height = 0.3  # Adjusted border height for realism

# Set the table surface to be at y = 0
table_y = 0

# Create the table surface
table_surface = box(pos=vector(0, table_y, 0),
                    size=vector(table_length, table_height, table_width),
                    color=color.green)

# Create table borders
border_y = table_surface.pos.y + table_height/2 + border_height/2
borders = [
    # Top border (along X-axis)
    box(pos=vector(0, border_y, -table_width/2 - border_thickness/2),
        size=vector(table_length + 2*border_thickness, border_height, border_thickness), color=brown),
    # Bottom border (along X-axis)
    box(pos=vector(0, border_y, table_width/2 + border_thickness/2),
        size=vector(table_length + 2*border_thickness, border_height, border_thickness), color=brown),
    # Left border (along Z-axis)
    box(pos=vector(-table_length/2 - border_thickness/2, border_y, 0),
        size=vector(border_thickness, border_height, table_width + 2*border_thickness), color=brown),
    # Right border (along Z-axis)
    box(pos=vector(table_length/2 + border_thickness/2, border_y, 0),
        size=vector(border_thickness, border_height, table_width + 2*border_thickness), color=brown),
]

# Combine the table surface and borders into a single compound object
pool_table = compound([table_surface] + borders, pos=vector(0, 0, 0))
pool_table.up = vector(0, 1, 0)  # Set the up vector
pool_table.axis = vector(1, 0, 0)  # Set the axis vector

# Ball properties
ball_radius = 0.2
ball_colors = [color.red, color.blue, color.yellow, color.orange, color.cyan, color.magenta]

# Create balls and shadows
balls = []
shadows = []
for i in range(6):
    x = -table_length/4 + i * ball_radius * 2.5
    y = table_y + table_height/2 + ball_radius
    z = 0
    ball = sphere(pos=vector(x, y, z), radius=ball_radius, color=ball_colors[i], make_trail=False)
    ball.velocity = vector(0, 0, 0)
    balls.append(ball)
    
    # Shadows will be updated dynamically
    shadow = cylinder(radius=ball_radius, color=color.black, opacity=0.2)
    shadows.append(shadow)

# Gravity magnitude
g_magnitude = 9.81  # Gravity magnitude (m/s^2)

# Friction coefficient
mu = 0.05  # Adjust as needed

# Time variables
dt = 0.01

# Variables for table rotation
tilt_angle_x = 0  # Rotation around X-axis
tilt_angle_y = 0  # Rotation around Y-axis

# Function to handle key presses
def keydown(evt):
    global tilt_angle_x, tilt_angle_y
    s = evt.key
    angle_increment = 0.005  # Further reduced rotation sensitivity for subtle tilt
    if s == 'up':
        tilt_angle_x += angle_increment
    elif s == 'down':
        tilt_angle_x -= angle_increment
    elif s == 'left':
        tilt_angle_y += angle_increment
    elif s == 'right':
        tilt_angle_y -= angle_increment

# Bind the keydown event
scene.bind('keydown', keydown)

# Create gravity arrow (fixed position and orientation)
arrow_length = 2  # Adjust length as needed
gravity_arrow = arrow(pos=vector(0, 5, 0), axis=vector(0, -arrow_length, 0),
                      color=color.red, shaftwidth=0.2)

# Function to calculate distance from point to plane
def point_plane_distance(point, plane_point, plane_normal):
    return dot(point - plane_point, plane_normal)

# Simulation loop
while True:
    rate(100)
    # Gravity remains constant
    gravity_direction = vector(0, -g_magnitude, 0)  # Always pointing downward in world coordinates

    # Gravity arrow remains fixed (no need to update position or axis)

    # Apply rotations to the pool table based on tilt angles
    # Reset pool_table's up vector before applying rotations
    pool_table.up = vector(0, 1, 0)
    pool_table.axis = vector(1, 0, 0)
    pool_table.rotate(angle=tilt_angle_y, axis=vector(0, 1, 0), origin=pool_table.pos)
    pool_table.rotate(angle=tilt_angle_x, axis=vector(1, 0, 0), origin=pool_table.pos)

    # Pool table's upward normal vector (perpendicular to its surface)
    table_normal = pool_table.up.norm()
    table_pos = pool_table.pos + pool_table.up * (table_height/2)  # A point on the table surface

    for i, ball in enumerate(balls):
        # Update velocities (gravity affects only the balls)
        ball.velocity += gravity_direction * dt
        
        # Apply friction
        if mag(ball.velocity) > 0:
            friction = -mu * ball.velocity.norm()
            ball.velocity += friction * dt
        
        # Update positions
        ball.pos += ball.velocity * dt

        # Update shadow position
        # Project ball position onto table surface
        distance_to_table = point_plane_distance(ball.pos, table_pos, table_normal)
        shadow_pos = ball.pos - distance_to_table * table_normal
        shadows[i].pos = shadow_pos + table_normal * 0.01  # Slightly above the table surface
        shadows[i].axis = table_normal * 0.001
        shadows[i].radius = ball_radius

        # Collision with table surface
        if distance_to_table - ball_radius <= 0:
            # Reflect velocity
            ball.velocity = ball.velocity - 2 * dot(ball.velocity, table_normal) * table_normal * 0.9
            # Adjust position to prevent penetration
            ball.pos = ball.pos - table_normal * (distance_to_table - ball_radius)

        # Collision with borders (part of the compound object)
        # We need to check for collisions with the sides of the table

        # Define the table boundaries in local coordinates
        local_pos = ball.pos - pool_table.pos
        half_length = table_length / 2
        half_width = table_width / 2

        # Transform the ball position into the table's local coordinate system
        # Since the table is rotated, we need to account for its orientation
        # We'll use the pool_table's frame of reference
        # Calculate the local coordinate axes
        x_axis = pool_table.axis.norm()
        y_axis = pool_table.up.norm()
        z_axis = cross(x_axis, y_axis).norm()

        # Decompose local_pos into components along the table's axes
        x_comp = dot(local_pos, x_axis)
        y_comp = dot(local_pos, y_axis)
        z_comp = dot(local_pos, z_axis)

        # Check for collision with the table borders
        if abs(x_comp) + ball_radius >= half_length + border_thickness / 2:
            # Reflect velocity along the x-axis
            normal = x_axis * sign(x_comp)
            ball.velocity = ball.velocity - 2 * dot(ball.velocity, normal) * normal * 0.9
            # Adjust position to prevent penetration
            overlap = abs(x_comp) + ball_radius - (half_length + border_thickness / 2)
            ball.pos -= normal * overlap

        if abs(z_comp) + ball_radius >= half_width + border_thickness / 2:
            # Reflect velocity along the z-axis
            normal = z_axis * sign(z_comp)
            ball.velocity = ball.velocity - 2 * dot(ball.velocity, normal) * normal * 0.9
            # Adjust position to prevent penetration
            overlap = abs(z_comp) + ball_radius - (half_width + border_thickness / 2)
            ball.pos -= normal * overlap

        # If the ball falls below a certain y-value, reset its position (optional)
        if ball.pos.y < -10:
            ball.pos = vector(0, table_y + table_height/2 + ball_radius, 0)
            ball.velocity = vector(0, 0, 0)

        # Collision with other balls
        for other_ball in balls:
            if other_ball != ball:
                diff = ball.pos - other_ball.pos
                dist = mag(diff)
                if dist < 2 * ball_radius:
                    # Simple elastic collision
                    normal = diff.norm()
                    relative_velocity = ball.velocity - other_ball.velocity
                    vel_along_normal = dot(relative_velocity, normal)
                    if vel_along_normal > 0:
                        continue
                    # Compute impulse scalar
                    impulse = -2 * vel_along_normal / (1/ball.radius + 1/other_ball.radius)
                    impulse_vector = impulse * normal
                    # Update velocities
                    ball.velocity += impulse_vector / ball.radius
                    other_ball.velocity -= impulse_vector / other_ball.radius
                    # Adjust positions to prevent overlap
                    overlap = 2 * ball_radius - dist
                    ball.pos += normal * (overlap / 2)
                    other_ball.pos -= normal * (overlap / 2)
