# Gray Scott's Reaction Diffusion, Implemented using Python.

This project implements gray scott's reaction diffusion using python. Individual frames are calculated and generated as .png photos by using Pillow. Numpy arrays are used to store value of concentration of each chemical in each cell. To speed up the calculations, numba is also used. [This](http://karlsims.com/rd.html) article by Karl Sims served as a tutorial / inspiration / reference for this project.


### Table of Contents

- Overview
- Features
- Requirements
- Installation
- Usage
- Output
- License


### Overview

The Gray-Scott model is a reaction-diffusion system that simulates the interactions between two chemical substances. By adjusting various parameters, you can observe the formation of intricate patterns, including spots, stripes, and other complex structures.


### Features

- Real-time simulation of the Gray-Scott reaction-diffusion model.
- Adjustable parameters to control the appearance of patterns.
- Visualization of the simulation using pygame.


### Requirements

- Python (>= 3.6)
- llvmlite (0.40.1)
- numba (0.57.1)
- numpy (1.24.4)
- pygame (2.5.0)


### Installation

1. Clone this repository to your local machine or download the ZIP file.
```bash
git clone https://github.com/preyasu-rakshit/reaction-diffusion.git
```

2. Navigate to the project directory.
```bash
cd reaction-diffusion
```

3. Install the required dependencies using pip.
```bash
pip install -r requirements.txt
```

### Usage

1. This simulation has two demos: one runs the simulation in real time using pygame. For this demo, run:
```bash
python gray-scott-pygame.py
```

2. For the second demo which saves each frame as an image, run:
```bash
python main.py
```

3. To change patterns, you need to change two important parameters: feed-rate and kill-rate. Below are some well known patterns and the corresponding values of kill and feed rate:

Pattern | Feed Rate | Kill Rate
Default | 0.037 | 0.060
Solitons | 0.03 | 0.062
Moving Spots | 0.014 | 0.054
Waves | 0.014 | 0.045


### Output

1. A sample with f = 0.018, k = 0.051:
<img src="./SampleImages/output.gif">



### License

This project is licensed under the MIT License.
