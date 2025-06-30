from typing import Optional

from ..constants import ActionSpace
from ..types import Float32NDArray, NDArray, ObservationSequence
from .buffers import InfiniteBuffer
from .episode_generator import EpisodeGenerator
from .replay_buffer import ReplayBuffer, EpisodeBase, Signature
from .trajectory_slicers import TrajectorySlicerProtocol, BasicTrajectorySlicer
from .transition_pickers import TransitionPickerProtocol, BasicTransitionPicker
from torch.utils.data import Dataset
import torch
from typing import List, Optional, Callable
import numpy as np

__all__ = ["MDPDataset", "EpisodeDataset", "EpisodeWindowDataset"]


class MDPDataset(ReplayBuffer):
    r"""Backward-compability class of MDPDataset.

    This is a wrapper class that has a backward-compatible constructor
    interface.

    Args:
        observations (ObservationSequence): Observations.
        actions (np.ndarray): Actions.
        rewards (np.ndarray): Rewards.
        terminals (np.ndarray): Environmental terminal flags.
        timeouts (np.ndarray): Timeouts.
        transition_picker (Optional[TransitionPickerProtocol]):
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer (Optional[TrajectorySlicerProtocol]):
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        action_space (Optional[d3rlpy.constants.ActionSpace]):
            Action-space type.
        action_size (Optional[int]): Size of action-space. For continuous
            action-space, this represents dimension of action vectors. For
            discrete action-space, this represents the number of discrete
            actions.
    """

    def __init__(
        self,
        observations: ObservationSequence,
        actions: NDArray,
        rewards: Float32NDArray,
        terminals: Float32NDArray,
        timeouts: Optional[Float32NDArray] = None,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        action_space: Optional[ActionSpace] = None,
        action_size: Optional[int] = None,
    ):
        episode_generator = EpisodeGenerator(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
        )
        buffer = InfiniteBuffer()
        super().__init__(
            buffer,
            episodes=episode_generator(),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            action_space=action_space,
            action_size=action_size,
        )

class ReturnEpisode(EpisodeBase):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
        episode_return: float = 0.0,
    ):
        self._observations = observations
        self._actions = actions
        self._terminals = terminals
        self._timeouts = timeouts if timeouts is not None else np.zeros_like(terminals)
        self._episode_return = episode_return

    @property
    def observations(self) -> np.ndarray:
        return self._observations

    @property
    def actions(self) -> np.ndarray:
        return self._actions

    @property
    def rewards(self) -> np.ndarray:
        # placeholder: repeat total return across all steps
        return np.full(len(self._actions), self._episode_return, dtype=np.float32)

    @property
    def terminals(self) -> np.ndarray:
        return self._terminals

    @property
    def timeouts(self) -> np.ndarray:
        return self._timeouts

    @property
    def episode_return(self) -> float:
        return self._episode_return

    def __len__(self) -> int:
        return len(self._actions)

    @property
    def observation_signature(self) -> Signature:
        return Signature(dtype=[self._observations.dtype], shape=[self._observations.shape[1:]])

    @property
    def action_signature(self) -> Signature:
        return Signature(dtype=[self._actions.dtype], shape=[self._actions.shape[1:]])

    @property
    def reward_signature(self) -> Signature:
        return Signature(dtype=[np.float32], shape=[[1]])


from torch.utils.data import Dataset

class EpisodeDataset(Dataset):  # <- Changed from ReplayBuffer to Dataset
    def __init__(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        terminals: List[np.ndarray],
        returns: List[float],
        timeouts: Optional[List[np.ndarray]] = None,
    ):
        self.episodes: List[ReturnEpisode] = []
        for i in range(len(observations)):
            self.episodes.append(
                ReturnEpisode(
                    observations=observations[i],
                    actions=actions[i],
                    terminals=terminals[i],
                    timeouts=(timeouts[i] if timeouts else None),
                    episode_return=returns[i],
                )
            )

        # Normalize returns to [0, 1] and store as weights
        self.weights = np.array(returns, dtype=np.float32)
        self.weights = (self.weights - self.weights.min()) / max(1e-8, np.ptp(self.weights))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return (
            torch.tensor(ep.observations, dtype=torch.float32),  # shape: (T, obs_dim)
            torch.tensor(ep.actions, dtype=torch.float32),       # shape: (T, act_dim)
            torch.tensor(self.weights[idx], dtype=torch.float32) # scalar weight
        )

class EpisodeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dataset: EpisodeDataset, seq_len: int = 5):
        self.seq_len = seq_len
        self.sequences = []

        for ep_obs, ep_act, weight in episode_dataset:
            T = ep_obs.shape[0]
            if T < seq_len:
                continue  # skip episodes shorter than required window
            for start in range(T - seq_len + 1):
                obs_seq = ep_obs[start:start + seq_len]  # (5, obs_dim)
                act_seq = ep_act[start:start + seq_len]  # (5, act_dim)
                self.sequences.append((obs_seq, act_seq, weight))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]  # returns: (obs_seq, act_seq, weight)

