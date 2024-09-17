import random
import math
import pygame  # Add this line to import pygame
import pymunk
import pymunk.pygame_util
from PIL import Image  # For creating GIF
import io  # For managing in-memory byte objects
import numpy as np
from scipy.interpolate import interp1d
from pygame import gfxdraw
from pygame import surfarray

# Initialize pygame and pymunk
pygame.init()
pymunk.pygame_util.positive_y_is_up = False  # Pygame's Y-axis goes down
space = pymunk.Space()
space.gravity = (0, 900)  # Simulate gravity

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cannon vs Bricks with Visual Effects")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
LIGHT_BLUE = (135, 206, 235)  # Sky color

# Ground settings
ground_height = 500
flat_regions = [(300, 400), (500, 600)]  # Define flat regions for bricks to stand on

# Replace the ground_bumps definition with this new terrain generation function
def generate_terrain(length, num_control_points, min_height=450, max_height=550, smoothness='cubic'):
    control_x = np.linspace(0, length - 1, num_control_points)
    control_y = np.random.uniform(min_height, max_height, num_control_points)
    
    # Ensure flat regions for bricks
    flat_regions = [(300, 400), (500, 600)]
    for start, end in flat_regions:
        mask = (control_x >= start) & (control_x <= end)
        control_y[mask] = min_height

    f = interp1d(control_x, control_y, kind=smoothness)
    x = np.arange(length)
    y = f(x)
    return list(zip(x, y.astype(int)))

def create_terrain_surface(ground_bumps, texture_path):
    terrain_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    
    # Load the grass texture
    grass_texture = pygame.image.load(texture_path).convert_alpha()
    
    # Apply a brownish tone using an RGB filter
    brown_tone = (150, 100, 50, 255)  # Brownish color filter (R, G, B, A)
    grass_texture.fill(brown_tone, special_flags=pygame.BLEND_RGBA_MULT)
    
    texture_width, texture_height = grass_texture.get_size()
    
    # Tile the grass texture across the width of the screen
    for x in range(0, WIDTH, texture_width):
        for y in range(0, HEIGHT, texture_height):
            terrain_surface.blit(grass_texture, (x, y))
    
    # Create a mask based on the terrain shape
    mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(mask, (255, 255, 255, 255), ground_bumps + [(WIDTH, HEIGHT), (0, HEIGHT)])
    
    # Apply the mask to the terrain surface
    terrain_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    return terrain_surface


# Modify the draw_ground function
def draw_ground(terrain_surface):
    screen.blit(terrain_surface, (0, 0))

# Function to create the ground using static pymunk shapes
def create_ground():
    ground_segments = []
    for i in range(len(ground_bumps) - 1):
        start = ground_bumps[i]
        end = ground_bumps[i + 1]
        segment = pymunk.Segment(space.static_body, start, end, 3)
        segment.friction = 1.0
        space.add(segment)
        ground_segments.append(segment)
    return ground_segments

# Add a new global variable for the background image
background_image = None

def draw_sky():
    screen.blit(background_image, (0, 0))

# Particle class for visual effects
class Particle:
    def __init__(self, x, y, color, size, life=30):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.vel_x = random.uniform(-1, 1) * 2
        self.vel_y = random.uniform(-2, 0) * 2

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.life -= 1

    def draw(self):
        if self.life > 0:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

# Function to add particle effects when the cannon fires
def generate_cannon_particles(x, y):
    for _ in range(10):  # Create 10 particles at once
        particles.append(Particle(x, y, (255, 140, 0), random.randint(3, 6)))

# Cannon class
class Cannon:
    base_width = 60
    base_height = 20
    turret_width = 40
    turret_height = 20  # Increased turret height for better visuals
    barrel_length = 50  # Lengthened barrel
    barrel_width = 8     # Thickened barrel for a more solid look

    def __init__(self, ground_y):
        self.x = 50
        self.y = ground_y - 10  # Position the cannon on the ground
        self.angle = 45  # Start at 45 degrees above horizontal
        self.power = 10  # Initialize the power to 10

    def draw(self):
        # Draw the base with more depth and shading (simulate 3D effect)
        pygame.draw.rect(screen, (100, 50, 0), (self.x - self.base_width // 2, self.y - self.base_height // 2 + 4, self.base_width, self.base_height))  # Shadow
        pygame.draw.rect(screen, BROWN, (self.x - self.base_width // 2, self.y - self.base_height // 2, self.base_width, self.base_height))  # Base
        
        # Draw tank wheels with spokes
        wheel_radius = 10  # Make the wheels slightly bigger
        wheel_positions = [(-20, 10), (-10, 10), (0, 10), (10, 10), (20, 10)]
        for pos in wheel_positions:
            pygame.draw.circle(screen, BLACK, (self.x + pos[0], self.y + pos[1]), wheel_radius)  # Outer wheel
            pygame.draw.circle(screen, (80, 80, 80), (self.x + pos[0], self.y + pos[1]), wheel_radius - 3)  # Inner part of the wheel
            # Add wheel spokes
            for i in range(5):
                angle = (2 * math.pi / 5) * i
                spoke_x = int(self.x + pos[0] + math.cos(angle) * (wheel_radius - 2))
                spoke_y = int(self.y + pos[1] + math.sin(angle) * (wheel_radius - 2))
                pygame.draw.line(screen, BLACK, (self.x + pos[0], self.y + pos[1]), (spoke_x, spoke_y), 2)

        # Draw a rounded turret for a cooler look
        pygame.draw.circle(screen, BROWN, (self.x, self.y - self.base_height // 2 - self.turret_height // 2), self.turret_width // 2)  # Turret

        # Draw the barrel with shading to simulate a metallic look
        barrel_start = (self.x, self.y - self.base_height // 2 - self.turret_height // 2)
        end_x = self.x + math.cos(math.radians(self.angle)) * self.barrel_length
        end_y = self.y - self.base_height // 2 - self.turret_height // 2 - math.sin(math.radians(self.angle)) * self.barrel_length
        pygame.draw.line(screen, (80, 80, 80), barrel_start, (end_x, end_y), self.barrel_width + 2)  # Barrel shadow
        pygame.draw.line(screen, GREEN, barrel_start, (end_x, end_y), self.barrel_width)  # Barrel

        # Add a muzzle at the end of the barrel to make it more realistic
        muzzle_radius = self.barrel_width // 2 + 2
        pygame.draw.circle(screen, BLACK, (int(end_x), int(end_y)), muzzle_radius)

    def get_barrel_end(self):
        end_x = self.x + math.cos(math.radians(self.angle)) * self.barrel_length
        end_y = self.y - self.base_height // 2 - self.turret_height // 2 - math.sin(math.radians(self.angle)) * self.barrel_length
        return end_x, end_y

    def increase_angle(self):
        if self.angle < 180:
            self.angle += 1

    def decrease_angle(self):
        if self.angle > 0:
            self.angle -= 1

    def increase_power(self):
        self.power += 1

    def decrease_power(self):
        if self.power > 1:
            self.power -= 1



# Cannonball class with smoke trails and screen shake effect
class Cannonball:
    def __init__(self, x, y, angle, power):
        self.mass = 10  # Assign proper weight to the cannonball
        self.radius = 10
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius)  # Calculate moment of inertia for the ball
        self.body = pymunk.Body(self.mass, inertia)  # Create a dynamic body
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.6
        self.shape.friction = 0.5
        space.add(self.body, self.shape)

        # Apply the initial impulse based on angle and power
        impulse = power * 500
        self.body.apply_impulse_at_local_point((math.cos(math.radians(angle)) * impulse, -math.sin(math.radians(angle)) * impulse))
        self.smoke_active = True  # Start smoke trail as active
        self.spin_rate = random.uniform(-5, 5)  # Add some spin for realism
        self.air_resistance = 0.99  # Simulate air resistance
        
        # Visual effect parameters
        self.color = (150, 150, 150)  # Metallic cannonball color
        self.glow = (255, 100, 0)  # Initial glow color when fired

    def apply_physics(self):
        # Apply air resistance to slow down the cannonball
        self.body.velocity = self.body.velocity * self.air_resistance
        # Apply a small spin (rotating around its own axis)
        self.body.angle += self.spin_rate

    def draw(self):
        # Check if the cannonball has come to rest (very low velocity)
        if self.body.velocity.length < 5:
            self.smoke_active = False
        
        # Draw a glowing effect around the cannonball when it is first fired
        if self.smoke_active:
            pygame.draw.circle(screen, self.glow, (int(self.body.position.x), int(self.body.position.y)), self.radius + 4)

        # Draw the cannonball
        pygame.draw.circle(screen, self.color, (int(self.body.position.x), int(self.body.position.y)), self.radius)

        # Add smoke trails behind the cannonball if still moving
        if self.smoke_active:
            smoke_trails.append(Particle(self.body.position.x, self.body.position.y, (105, 105, 105), 5, life=30))

    def on_impact(self):
        # Create a visual impact effect with particles
        for _ in range(20):
            particles.append(Particle(self.body.position.x, self.body.position.y, (150, 75, 0), random.randint(3, 6)))
        apply_screen_shake()  # Apply screen shake on impact

# Brick class with realistic physics
class Brick:
    def __init__(self, x, y, mass=1, size=30):  # Reduced size to 30x30 for smaller crates
        self.size = size
        self.body = pymunk.Body(mass, pymunk.moment_for_box(mass, (self.size, self.size)))  # Create dynamic body
        self.body.position = x, y
        self.shape = pymunk.Poly.create_box(self.body, (self.size, self.size))  # Square shape for the brick
        self.shape.elasticity = 0.4
        self.shape.friction = 0.8
        space.add(self.body, self.shape)

        # Load the crate texture
        self.crate_texture = pygame.image.load("crate.png").convert_alpha()

    def draw(self):
        # Get the position and rotation of the brick
        position = self.body.position
        angle = math.degrees(self.body.angle)  # Convert angle to degrees for pygame
        
        # Get the texture surface and rotate it
        texture = pygame.transform.scale(self.crate_texture, (self.size, self.size))  # Scale the texture to fit the crate
        texture = pygame.transform.rotate(texture, angle)
        rect = texture.get_rect(center=(int(position.x), int(position.y)))

        # Draw the rotated texture onto the screen
        screen.blit(texture, rect.topleft)

# Function to apply screen shake effect when the cannonball hits a brick
def apply_screen_shake():
    offset_x = random.randint(-5, 5)
    offset_y = random.randint(-5, 5)
    screen.scroll(offset_x, offset_y)

def main():
    global ground_bumps, background_image  # Use the global variable
    font = pygame.font.Font(None, 36)  # Font for text display
    
    # Initialize particles and smoke_trails lists
    global particles, smoke_trails
    particles = []
    smoke_trails = []
    
    # Load the background image
    background_image = pygame.image.load("background.png").convert()

    # Generate terrain using spline interpolation
    ground_bumps = generate_terrain(WIDTH, 20)
    
    # Create the terrain surface with grass texture
    terrain_surface = create_terrain_surface(ground_bumps, "grass.png")
    
    # Initialize the cannon with the ground height at x=50
    cannon = Cannon(ground_bumps[50][1])
    
    cannonballs = []
    bricks = []
    create_ground()  # Initialize the ground with the new terrain

    # Create multiple stacks of bricks at flat regions only
    # Create multiple stacks of crate-like bricks at flat regions only
    for stack_x in [350, 550]:
        for i in range(6):  # Stack 6 crates high
            y_position = ground_bumps[stack_x][1] - (i * 32)  # Adjust spacing between crates (slightly more than the crate size)
            bricks.append(Brick(stack_x, y_position, mass=random.uniform(1, 3), size=50))


    # GIF frame capture setup
    frames = []
    frame_duration = 1000 // 30  # 30 frames per second
    capture_duration = 20 * 1000  # Capture 20 seconds
    total_frames = capture_duration // frame_duration

    running = True
    frame_count = 0

    while running:
        # Draw the sky (which is now the background image)
        draw_sky()

        # Draw the ground
        draw_ground(terrain_surface)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    cannon.increase_angle()
                if event.key == pygame.K_DOWN:
                    cannon.decrease_angle()
                if event.key == pygame.K_RIGHT:
                    cannon.increase_power()
                if event.key == pygame.K_LEFT:
                    cannon.decrease_power()
                if event.key == pygame.K_SPACE:
                    # Fire a cannonball from the end of the barrel
                    barrel_end_x, barrel_end_y = cannon.get_barrel_end()
                    cannonballs.append(Cannonball(barrel_end_x, barrel_end_y, cannon.angle, cannon.power))
                    generate_cannon_particles(barrel_end_x, barrel_end_y)

        # Update and draw cannonballs
        for cannonball in cannonballs:
            cannonball.draw()

        # Update and draw bricks
        for brick in bricks:
            brick.draw()

        # Draw the cannon
        cannon.draw()

        # Draw and update particles
        for particle in particles[:]:
            particle.update()
            particle.draw()
            if particle.life <= 0:
                particles.remove(particle)

        # Draw and update smoke trails
        for smoke in smoke_trails[:]:
            smoke.update()
            smoke.draw()
            if smoke.life <= 0:
                smoke_trails.remove(smoke)

        # Display cannon power and angle
        power_text = font.render(f"Power: {cannon.power}", True, BLACK)
        angle_text = font.render(f"Angle: {cannon.angle}", True, BLACK)
        screen.blit(power_text, (10, 10))
        screen.blit(angle_text, (10, 50))

        # Step the physics simulation
        space.step(1/60.0)

        # Capture frame for GIF if within capture duration
        if frame_count < total_frames:
            # Capture the current screen
            frame_data = pygame.image.tostring(screen, "RGB")
            image = Image.frombytes("RGB", (WIDTH, HEIGHT), frame_data)
            frames.append(image)

        pygame.display.flip()  # Refresh the screen
        clock.tick(60)  # 60 FPS

        frame_count += 1

    pygame.quit()

    # Save frames as GIF using PIL
    frames[0].save("gameplay.gif", save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)


if __name__ == "__main__":
    main()