# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import matplotlib.pyplot as plt


def plot_spikes(spikes, figsize, legend, colors, title):
    offsets = list(range(1, len(spikes) + 1))
    
    plt.figure(figsize=figsize)
    
    spikes_plot = plt.eventplot(positions=spikes, 
                                lineoffsets=offsets,
                                linelength=0.9,
                                colors=colors)
    
    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")
    plt.yticks(ticks=offsets, labels=legend)
    
    plt.show()

def plot_time_series(time, time_series, ylabel, title, figsize, color):
    plt.figure(figsize=figsize)
    
    plt.step(time, time_series, color=color)
   
    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    
    plt.show()

def plot_time_series_subplots(time, time_series_y1, time_series_y2, ylabel, title, figsize, color, legend):    
    plt.figure(figsize=figsize)
    
    plt.step(time, time_series_y1, label=legend[0], color=color[0])
    plt.step(time, time_series_y2, label=legend[1], color=color[1])
        
    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    plt.legend(loc="upper left")
    
    plt.show()