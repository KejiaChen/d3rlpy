import numpy as np
import os
from d3rlpy.dataset import MDPDataset, EpisodeDataset
from fixing_detection_no_smoothing import FFAnalyzerRTSlide 
import json

# def success_threshold(criterion, threshold):

from torch.utils.data import Dataset, DataLoader
import torch

def utils_integral(y, x):
    delta_x = np.diff(x, prepend=x[0])[:-1]  # Slice to match the length of y_avg
    y_avg = (y[:-1] + y[1:]) / 2
    return np.sum(y_avg * delta_x)

def max_based_return_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_stretch_energy=0.02, ref_push_force=7.0, ref_push_effort=7.0, ref_push_energy=0.1, ref_deformation=0.01, ref_duration=1000, dt=0.005):
    # === 1. Extract Metrics ===
    stretch_force = np.max(np.abs(obs[:, 0])) # np.mean
    push_force = np.max(np.abs(obs[:, 3]))
    deformation = np.max(np.abs(obs[:, 4])) 
    duration = obs.shape[0]  # number of steps

    # === 2. Define Ideal Reference Values (based on expert or dataset stats) ===
    # REF_STRETCH_FORCE = 7.0
    # REF_PUSH_FORCE = 7.0
    # REF_DEFORMATION = 0.01
    # REF_DURATION = 1000  

    # === 3. Compute Positive Reward by Deviation ===
    stretch_score = 1.0 - (stretch_force - ref_stretch_force) / ref_stretch_force
    push_score = 1.0 - (push_force - ref_push_force) / ref_push_force
    deform_score = 1.0 - (deformation - ref_deformation) / ref_deformation
    time_score = 1.0 - (duration - ref_duration) / ref_duration

    # Clip scores to [0, 1]
    # stretch_score = np.clip(stretch_score, 0.0, 1.0)
    # push_score = np.clip(push_score, 0.0, 1.0)
    # deform_score = np.clip(deform_score, 0.0, 1.0)
    # time_score = np.clip(time_score, 0.0, 1.0)
    
    if include_deformation:
        ep_return = 10*stretch_score + 10*push_score + 10*deform_score + 10*time_score
    else:
        ep_return = 15*stretch_score + 15*push_score + 10*time_score

    return ep_return

def effort_based_return_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_stretch_energy=0.02, ref_push_force=7.0, ref_push_effort=7.0, ref_push_energy=0.1, ref_deformation=0.01, ref_duration=1000, dt=0.005):
    # 1. Total force (integrated effort)
    stretch_effort = np.sum(np.abs(obs[:, 0])) * dt
    push_effort = np.sum(np.abs(obs[:, 3])) * dt

    # 2. Max deformation and duration
    deformation = np.max(np.abs(obs[:, 4]))
    duration = obs.shape[0]

    # 3. Scores
    stretch_score = 1.0 - (stretch_effort - ref_stretch_effort) / ref_stretch_effort
    push_score    = 1.0 - (push_effort - ref_push_effort) / ref_push_effort
    deform_score  = 1.0 - (deformation - ref_deformation) / ref_deformation
    # time_score    = 1.0 - (duration - ref_duration) / ref_duration

    # 4. Clip to [0, 1]
    # stretch_score = np.clip(stretch_score, 0.0, 1.0)
    # push_score = np.clip(push_score, 0.0, 1.0)
    # deform_score = np.clip(deform_score, 0.0, 1.0)
    # time_score = np.clip(time_score, 0.0, 1.0)

    # 5. Final return
    if include_deformation:
        ep_return = 15 * stretch_score + 15 * push_score + 10 * deform_score
    else:
        ep_return = 20 * stretch_score + 20 * push_score
    return ep_return


def effort_and_max_based_return_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_stretch_energy=0.02, ref_push_force=7.0, ref_push_effort=7.0, ref_push_energy=0.1, ref_deformation=0.01, ref_duration=1000, dt=0.005):
    # 1. Total force (integrated effort)
    stretch_effort = np.sum(np.abs(obs[:, 0])) * dt
    push_effort = np.sum(np.abs(obs[:, 3])) * dt
    max_stretch_force = np.max(np.abs(obs[:, 0]))
    max_push_force = np.max(np.abs(obs[:, 3]))

    # 2. Max deformation and duration
    deformation = np.max(np.abs(obs[:, 4]))
    duration = obs.shape[0]

    # 3. Scores
    stretch_score = 1.0 - (stretch_effort - ref_stretch_effort) / ref_stretch_effort
    push_score = 1.0 - (push_effort - ref_push_effort) / ref_push_effort
    deform_score = 1.0 - (deformation - ref_deformation) / ref_deformation
    peak_stretch_score = 1.0 - (max_stretch_force - ref_stretch_force) / ref_stretch_force
    peak_push_score = 1.0 - (max_push_force - ref_push_force) / ref_push_force

    # Clip scores to [0, 1]
    # stretch_score = np.clip(stretch_score, 0.0, 1.0)
    # push_score = np.clip(push_score, 0.0, 1.0)
    # deform_score = np.clip(deform_score, 0.0, 1.0)
    # peak_stretch_score = np.clip(peak_stretch_score, 0.0, 1.0)

    if include_deformation:
        ep_return = 12 * stretch_score + 12 * push_score + 4 * deform_score + 6 * peak_stretch_score + 6 * peak_push_score
    else:
        ep_return = 17 * stretch_score + 17 * push_score + 6 * peak_stretch_score

    return ep_return


def effort_and_energy_based_return_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_stretch_energy=0.02, ref_push_force=7.0, ref_push_effort=7.0, ref_push_energy=0.1, ref_deformation=0.01, ref_duration=1000, dt=0.005):
    # 1. Total force (integrated effort)
    stretch_effort = np.sum(np.abs(obs[:, 0])) * dt
    push_effort = np.sum(np.abs(obs[:, 3])) * dt
    stretch_energy = utils_integral(obs[:, 0], obs[:, 1])
    push_energy = utils_integral(obs[:, 3], obs[:, 4])
    total_energy = stretch_energy + push_energy

    # 2. Max deformation and duration
    deformation = np.max(np.abs(obs[:, 4]))
    duration = obs.shape[0]

    # 3. Scores
    ref_toatal_energy = ref_stretch_energy + ref_push_energy

    stretch_effort_score = 1.0 - (stretch_effort - ref_stretch_effort) / ref_stretch_effort
    stretch_energy_score = 1.0 - (stretch_energy - ref_stretch_energy) / ref_stretch_energy
    push_effort_score    = 1.0 - (push_effort - ref_push_effort) / ref_push_effort
    push_energy_score    = 1.0 - (push_energy - ref_push_energy) / ref_push_energy
    total_energy_score = 1.0 - (total_energy - ref_toatal_energy) / ref_toatal_energy
    # deform_score  = 1.0 - (deformation - ref_deformation) / ref_deformation
    # time_score    = 1.0 - (duration - ref_duration) / ref_duration

    # 4. Clip to [0, 1]
    # stretch_score = np.clip(stretch_score, 0.0, 1.0)
    # push_score = np.clip(push_score, 0.0, 1.0)
    # deform_score = np.clip(deform_score, 0.0, 1.0)
    # time_score = np.clip(time_score, 0.0, 1.0)

    # 5. Final return
    if include_deformation:
        ep_return = 15 * stretch_effort_score + 15 * push_effort_score + 10*total_energy_score
    else:
        ep_return = 20 * stretch_effort_score + 20 * push_effort_score
    return ep_return
