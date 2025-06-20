from typing import Any, Optional
import os

from .logger import (
    LOG,
    AlgProtocol,
    LoggerAdapter,
    LoggerAdapterFactory,
    SaveProtocol,
)

__all__ = ["WanDBAdapter", "WanDBAdapterFactory"]


class WanDBAdapter(LoggerAdapter):
    r"""WandB Logger Adapter class.

    This class logs data to Weights & Biases (WandB) for experiment tracking.

    Args:
        algo: Algorithm.
        experiment_name (str): Name of the experiment.
        n_steps_per_epoch: Number of steps per epoch.
        project: Project name.
    """

    def __init__(
        self,
        algo: AlgProtocol,
        experiment_name: str,
        n_steps_per_epoch: int,
        project: Optional[str] = None,
        local_logdir: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Please install wandb") from e
        assert algo.impl
        self.run = wandb.init(project=project, name=experiment_name)
        self._experiment_name = experiment_name
        self.run.watch(
            tuple(algo.impl.modules.get_torch_modules().values()),
            log="gradients",
            log_freq=n_steps_per_epoch,
        )
        self._is_model_watched = False

        self._local_logdir = local_logdir
        if not os.path.exists(self._local_logdir):
            os.makedirs(self._local_logdir)
            print(f"[WanDBAdapter] Created local log directory at {self._local_logdir}")

    def write_params(self, params: dict[str, Any]) -> None:
        """Writes hyperparameters to WandB config."""
        self.run.config.update(params)

    def before_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed before writing metric."""

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        """Writes metric to WandB."""
        self.run.log({name: value, "epoch": epoch}, step=step)

    def after_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed after writing metric."""

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        """Saves models to Weights & Biases.

        """
        # Implement saving model to wandb if needed
        file_name = f"model_{epoch}.pt"
        model_path = os.path.join(self._local_logdir, file_name)
        algo.save_model(model_path)
        print(f"[WandBAdapter] Saving model locally at epoch {epoch}...")

        # import wandb
        # artifact = wandb.Artifact(self._experiment_name + f"-model-{epoch}", type="model")
        # artifact.add_file(model_path)
        # self.run.log_artifact(artifact)

    def close(self) -> None:
        """Closes the logger and finishes the WandB run."""
        self.run.finish()

    def watch_model(
        self,
        epoch: int,
        step: int,
    ) -> None:
        pass

    def load_model(self, artifact_or_name) -> str:
        """Loads model from Weights & Biases.
        This method retrieves a model artifact by its name or ID and downloads it
        """
        artifact = self.run.use_artifact(artifact_or_name) # this creates a reference within Weights & Biases that this artifact was used by this run.
        path = artifact.download() # this downloads the artifact from Weights & Biases to your local system where the code is executing.
        print(f"Data directory located at {path}")
        return path

class WanDBAdapterFactory(LoggerAdapterFactory):
    r"""WandB Logger Adapter Factory class.

    This class creates instances of the WandB Logger Adapter for experiment
    tracking.

    Args:
        project (Optional[str], optional): The name of the WandB project.
            Defaults to None.
    """

    _project: Optional[str]

    def __init__(self, project: Optional[str] = None, root_dir: str = "./d3rlpy_logs") -> None:
        self._project = project
        self._root_dir = root_dir

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> LoggerAdapter:
        logdir = os.path.join(self._root_dir, experiment_name)
        print(f"[WanDBAdapterFactory] Creating log directory at {logdir}")
        return WanDBAdapter(
            algo=algo,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            project=self._project,
            local_logdir=logdir,
        )
