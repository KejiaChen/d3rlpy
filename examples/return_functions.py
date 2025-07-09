import numpy as np
import os
from d3rlpy.dataset import MDPDataset, EpisodeDataset
from fixing_detection_no_smoothing import FFAnalyzerRTSlide 
import json

# def success_threshold(criterion, threshold):

from torch.utils.data import Dataset, DataLoader
import torch

def max_based_return_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_push_force=7.0, ref_push_effort=7.0, ref_deformation=0.01, ref_duration=1000, dt=0.005):
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

def effort_based_return_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_push_force=7.0, ref_push_effort=7.0, ref_deformation=0.01, ref_duration=1000, dt=0.005):
    # 1. Total force (integrated effort)
    total_stretch = np.sum(np.abs(obs[:, 0])) * dt
    total_push = np.sum(np.abs(obs[:, 3])) * dt

    # 2. Max deformation and duration
    deformation = np.max(np.abs(obs[:, 4]))
    duration = obs.shape[0]

    # 3. Scores
    stretch_score = 1.0 - (total_stretch - ref_stretch_effort) / ref_stretch_effort
    push_score    = 1.0 - (total_push - ref_push_effort) / ref_push_effort
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
