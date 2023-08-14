from math import floor
import time

import numpy as np
from PIL import Image

# setup canvas
width = 400
height = 400
seed_size = 40

# Diffusion Rates
dA = 1.0
dB = 0.5

f = 0.018  # Feed Rate
k = 0.051  # Kill Rate

dt = 1  # time step

# Initialize grids that store the value of cocentration of chemicals in each pixel
current_grid = np.zeros((2, width, height), dtype=np.float32)
next_grid = np.zeros((2, width, height), dtype=np.float32)

current_grid[0,:,:] = 1
current_grid[1,:,:] = 0

a = floor(width/2 - seed_size/2)
b = floor(width/2 + seed_size/2)
c = floor(height/2 - seed_size/2)
d = floor(height/2 + seed_size/2)
current_grid[:,a:b,c:d] = 1 - current_grid[:,a:b,c:d]


# functions
def laplace2D(plane):
    plane020 = plane * 0.2
    plane005 = plane * 0.05

    new_plane = 0 - plane
    new_plane += np.roll(plane020, 1, axis=0)
    new_plane += np.roll(plane020, 1, axis=1)
    new_plane += np.roll(plane020, -1, axis=0)
    new_plane += np.roll(plane020, -1, axis=1)
    new_plane += np.roll(plane005, (1, 1), axis=(0, 1))
    new_plane += np.roll(plane005, (1, -1), axis=(0, 1))
    new_plane += np.roll(plane005, (-1, 1), axis=(0, 1))
    new_plane += np.roll(plane005, (-1, -1), axis=(0, 1))

    return new_plane


def update(grid):
    '''Updates the surface'''
    A, B = grid[0,:,:], grid[1,:,:]
    ABB = A*B*B

    newA = A + (dA * laplace2D(A) - ABB + f * (1 - A))
    newB = B + (dB * laplace2D(B) + ABB - (k + f) * B)
    new_grid = np.array((newA, newB), dtype=np.float32).clip(0, 1)

    return new_grid


def get_color(grid):
    '''set color of each pixel according to concentration'''
    rgb = np.empty((grid.shape[1], grid.shape[2], 3), dtype=np.uint8)

    A, B = grid[0,:,:], grid[1,:,:]
    chan = ((A - B) * 255).astype(int)
    rgb[:,:,0] = (255 - chan).clip(0, 255)
    rgb[:,:,1] = (55 - chan).clip(0, 255)
    rgb[:,:,2] = (155 - chan).clip(0, 255)

    return rgb


start_time = time.time()

# main loop
frames = 90000  # Number of frames desired by user
num = 0
for frame in range(frames):
    next_grid = update(current_grid)
    current_grid = next_grid

    # generates image every fourth frame, change according to your needs
    if frame % 4 == 0 and frame >= 1000:
        canvas = get_color(next_grid)
        img = Image.fromarray(np.rot90(canvas), 'RGB')
        str_num = "0000" + str(num)
        img.save(f'output\\pic{str_num[-5:]}.png')
        num += 1

    print(frame)

elapsed = time.time() - start_time
print(f'{width} * {height} * {frames} = {width * height * frames / elapsed / 1e6 :.1f} megapixels per second')
