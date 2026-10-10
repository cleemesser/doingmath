import pygame
import math
import sys

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mathematical Visualization: Sine Wave")
clock = pygame.time.Clock()

# Colors
BACKGROUND = (22, 22, 29)
AXIS_COLOR = (100, 100, 110)
GRID_COLOR = (40, 40, 50)
PLOT_COLOR = (0, 255, 200)

# Math Grid Config
SCALE = 50  # Pixels per 1 mathematical unit
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2

def to_screen(x, y):
    """Converts Cartesian math coordinates to Pygame screen space."""
    return (CENTER_X + int(x * SCALE), CENTER_Y - int(y * SCALE))

# Main Loop
time_offset = 0.0
running = True

while running:
    clock.tick(60)  # Caps framework at 60 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill(BACKGROUND)

    # 1. Draw Grid Lines
    for x_val in range(-int(CENTER_X/SCALE)-1, int(CENTER_X/SCALE)+1):
        pt1 = to_screen(x_val, -HEIGHT)
        pt2 = to_screen(x_val, HEIGHT)
        pygame.draw.line(screen, GRID_COLOR, (pt1[0], 0), (pt1[0], HEIGHT), 1)

    for y_val in range(-int(CENTER_Y/SCALE)-1, int(CENTER_Y/SCALE)+1):
        pt1 = to_screen(-WIDTH, y_val)
        pt2 = to_screen(WIDTH, y_val)
        pygame.draw.line(screen, GRID_COLOR, (0, pt1[1]), (WIDTH, pt1[1]), 1)

    # 2. Draw Main Cartesian Axes
    pygame.draw.line(screen, AXIS_COLOR, (0, CENTER_Y), (WIDTH, CENTER_Y), 2)  # X-axis
    pygame.draw.line(screen, AXIS_COLOR, (CENTER_X, 0), (CENTER_X, HEIGHT), 2) # Y-axis

    # 3. Calculate and Plot the Function: y = sin(x - t)
    points = []
    # Loop across the pixel width of the screen to sample math points
    for pixel_x in range(WIDTH):
        # Convert screen X back to math x
        math_x = (pixel_x - CENTER_X) / SCALE

        # Calculate Math Y
        math_y = math.sin(math_x - time_offset) * 2.0  # Amplitude of 2 units

        # Convert Math Y to Screen Y
        _, pixel_y = to_screen(math_x, math_y)
        points.append((pixel_x, pixel_y))

    # Draw continuous curve lines between calculated points
    if len(points) > 1:
        pygame.draw.lines(screen, PLOT_COLOR, False, points, 3)

    # Animate time forward
    time_offset += 0.05

    pygame.display.flip()

pygame.quit()
sys.exit()
