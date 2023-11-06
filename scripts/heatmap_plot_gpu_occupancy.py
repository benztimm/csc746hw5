#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: wes
Created: Thu Sep 30 05:51:28 PDT 2021

Description: this code generates a 2D "heatmap" style plot using sample data that
is hard-coded into the code.

Inputs: none, all problem parameters are hard-coded.

Outputs: a plot showing the heatmap, displayed to the screen

Dependencies: matplotlib, numpy

Assumptions: Developed and Tested with Python 3.8.8 on MacOS 11.6
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

threads_per_block = ['32', '64', '128', '256', '512', '1024'] # y axis, 6 of them
thread_blocks = ["1", "4", "16", "64", "256", "1024", "4096"] # x axis, 7 of them

runtime = np.array([[1.56, 1.56, 1.56, 1.56, 3.70, 14.75, 32.97],
                    [3.12, 3.12, 3.12, 3.12, 7.40, 28.95, 71.60],
                    [6.25, 6.25, 6.25, 6.25, 14.77, 54.17, 85.52],
                    [12.5, 12.5, 12.5, 12.49, 29.14, 72.55, 91.56],
                    [24.99, 24.98, 24.76, 24.73, 55.37, 85.55, 92.78],
                    [49.93, 48.97, 48.83, 48.60, 83.94, 91.19, 91.43]])


fig, ax = plt.subplots(figsize=(9,6))
im = ax.imshow(runtime, cmap="coolwarm_r")

# We want to show all ticks...
ax.set_xticks(np.arange(len(thread_blocks)))
ax.set_yticks(np.arange(len(threads_per_block)))
# ... and label them with the respective list entries
ax.set_xticklabels(thread_blocks)
ax.set_yticklabels(threads_per_block)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(threads_per_block)): # y axis
    for j in range(len(thread_blocks)): # x axis
        text = ax.text(j, i, runtime[i, j],
                       ha="center", va="center", color="k")

ax.set_title("Achieved Occupancy (%) on GPU-CUDA at Varying Block Size and Number of Blocks")
ax.set_ylabel('Threads per block')
ax.set_xlabel('Block Sizes')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("heatmap_gpu_occupancy.png")
plt.show()


# EOF
