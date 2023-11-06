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

runtime = np.array([[1330, 399.9, 104.6, 26.24, 7.12, 2.19, 1.24],
                    [673.6, 205.2, 55.6, 13.5, 4.1, 1.3, 0.77],
                    [360.6, 114.8, 27.9, 7.3, 2.2, 0.83, 0.67],
                    [232.5, 60.9, 15.5, 4.2, 1.3, 0.78, 0.64],
                    [139.4, 34.4, 9.2, 2.4, 0.93, 0.70, 0.64],
                    [79.3, 21.3, 5.5, 1.5, 0.88, 0.69, 0.66]])


fig, ax = plt.subplots(figsize=(9,6))
im = ax.imshow(runtime, cmap="coolwarm")

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

ax.set_title("gpu__time_duration.avg (Millisecond) on GPU-CUDA at Varying Block Size and Number of Blocks")
ax.set_ylabel('Threads per block')
ax.set_xlabel('Block Sizes')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("heatmap_gpu_runtime.png")
plt.show()


# EOF
