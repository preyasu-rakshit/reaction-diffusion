import sys
from math import floor

import numpy as np
import pygame
from numba import njit

# Pygame variables
width = 400
height = 400
screen_size = (width, height)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Reaction - Diffusion")
clock = pygame.time.Clock()

# setup seed size
seed_size = 20

# Diffusion Rates
dA = 1.0
dB = 0.5

f = 0.0367  # Feed Rate 
k = 0.0649  # Kill Rate

dt = 1  # time step


# Initialize grids that store the value of cocentration of chemicals in each pixel
current_grid = np.zeros((width, height, 2), dtype=np.float32)
next_grid = np.zeros((width, height, 2), dtype=np.float32)

current_grid[0:width, 0:height] = [1, 0]

a = floor(width/2 - seed_size/2)
b = floor(width/2 + seed_size/2)
c = floor(height/2 - seed_size/2)
d = floor(height/2 + seed_size/2)
current_grid[a:b, c:d] = [0, 1]

# functions


@njit
def laplace2D(x, y, grid, component):
    result = 0
    result += grid[x, y][component] * -1
    result += grid[x + 1, y][component] * 0.2
    result += grid[x - 1, y][component] * 0.2
    result += grid[x, y + 1][component] * 0.2
    result += grid[x, y - 1][component] * 0.2
    result += grid[x + 1, y + 1][component] * 0.05
    result += grid[x + 1, y - 1][component] * 0.05
    result += grid[x - 1, y + 1][component] * 0.05
    result += grid[x - 1, y - 1][component] * 0.05
    return result


@njit
def constrain(value, min_limit, max_limit):
    return min(max_limit, max(min_limit, value))


@njit
def update(c_grid):
    '''Updates the surface'''

    # set color of each pixel according to concentration
    new_arr = np.zeros((width, height, 2), dtype=np.float32)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            A = c_grid[x, y][0]
            B = c_grid[x, y][1]
            nA = A + (dA * laplace2D(x, y, c_grid, 0) - A*B*B + f*(1 - A)) * dt
            nB = B + (dB * laplace2D(x, y, c_grid, 1) + A*B*B - (k + f) * B) * dt

            new_A = constrain(nA, 0, 1)
            new_B = constrain(nB, 0, 1)
            new_arr[x, y] = [new_A, new_B]

    return new_arr


@njit
def get_color(grid):
    c_arr = np.zeros((width, height, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            A = grid[x, y][0]
            B = grid[x, y][1]
            c = floor((A - B) * 255)
            c_arr[x, y][0] = constrain(255 - c, 0, 255)
            c_arr[x, y][1] = constrain(55 - c, 0, 255)
            c_arr[x, y][2] = constrain(155 - c, 0, 255)

    return c_arr



# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    next_grid = update(current_grid)
    canvas = get_color(next_grid)
    display_surf = pygame.surfarray.make_surface(canvas)
    current_grid = next_grid

    screen.blit(display_surf, (0, 0))
    pygame.display.flip()
    # print(clock.get_fps())
    clock.tick()
