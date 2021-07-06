from math import floor

import numpy as np
from numba import njit
from PIL import Image

# setup canvas
width = 600
height = 600
seed_size = 40

# Diffusion Rates
dA = 1.0
dB = 0.5

f = 0.037  # Feed Rate
k = 0.06  # Kill Rate

dt = 1  # time step

# Initialize grids that store the value of cocentration of chemicals in each pixel
current_grid = np.zeros((width, height, 2), dtype=np.float32)
next_grid = np.zeros((width, height, 2), dtype=np.float32)

for x in range(width):
    for y in range(height):
        current_grid[x, y] = [1, 0]

a = floor(width/2 - seed_size/2)
b = floor(width/2 + seed_size/2)
c = floor(height/2 - seed_size/2)
d = floor(height/2 + seed_size/2)
current_grid[a:b, c:d] = [0, 1]


# functions
@njit
def laplace2DA(x, y, grid):
    result = 0
    result += grid[x, y][0] * -1
    result += grid[x + 1, y][0] * 0.2
    result += grid[x - 1, y][0] * 0.2
    result += grid[x, y + 1][0] * 0.2
    result += grid[x, y - 1][0] * 0.2
    result += grid[x + 1, y + 1][0] * 0.05
    result += grid[x + 1, y - 1][0] * 0.05
    result += grid[x - 1, y + 1][0] * 0.05
    result += grid[x - 1, y - 1][0] * 0.05
    return result


@njit
def laplace2DB(x, y, grid):
    result = 0
    result += grid[x, y][1] * -1
    result += grid[x + 1, y][1] * 0.2
    result += grid[x - 1, y][1] * 0.2
    result += grid[x, y + 1][1] * 0.2
    result += grid[x, y - 1][1] * 0.2
    result += grid[x + 1, y + 1][1] * 0.05
    result += grid[x + 1, y - 1][1] * 0.05
    result += grid[x - 1, y + 1][1] * 0.05
    result += grid[x - 1, y - 1][1] * 0.05
    return result


@njit
def constrain(value, min_limit, max_limit):
    return min(max_limit, max(min_limit, value))


@njit
def update(c_grid):
    '''Updates the surface'''

    new_arr = np.zeros((width, height, 2), dtype=np.float32)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            A = c_grid[x, y][0]
            B = c_grid[x, y][1]
            nA = A + (dA * laplace2DA(x, y, c_grid) - A*B*B + f*(1 - A)) * dt
            nB = B + (dB * laplace2DB(x, y, c_grid) + A*B*B - (k + f) * B) * dt

            new_A = constrain(nA, 0, 1)
            new_B = constrain(nB, 0, 1)
            new_arr[x, y] = [new_A, new_B]

    return new_arr


@njit
def get_color(grid):
    '''set color of each pixel according to concentration'''

    c_arr = np.zeros((width, height, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            A = grid[x, y][0]
            B = grid[x, y][1]
            color = constrain(A - B, 0, 1)
            c_arr[x, y][0] = floor(255 * color)
            c_arr[x, y][1] = floor(255 * color)
            c_arr[x, y][2] = floor(255 * color)

    return c_arr


# main loop
frames = 90000  # Number of frames desired by user
num = 0
for frame in range(frames):
    next_grid = update(current_grid)
    canvas = get_color(next_grid)
    current_grid = next_grid

    # generates image every fourth frame, change according to your needs
    if frame % 4 == 0:
        img = Image.fromarray(np.rot90(canvas), 'RGB')
        str_num = "0000" + str(num)
        img.save(f'video\\pic{str_num[-5:]}.png')
        num += 1

    print(frame)
