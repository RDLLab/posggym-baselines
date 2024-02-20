"""Classes and functions for handling logging."""
import abc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


class Logger(abc.ABC):
    """Abstract class for logging training stats and other outputs."""

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""

    @abc.abstractmethod
    def log_image(self, tag: str, image: str, step: int):
        """Log an image."""

    @abc.abstractmethod
    def log_figure(self, tag: str, figure, step: int):
        """Log a figure."""

    @abc.abstractmethod
    def log_matrix(self, tag: str, matrix: np.ndarray, step: int):
        """Log a matrix."""

    @abc.abstractmethod
    def upload_videos(self, step: int):
        """Upload any new video files in config.video_dir."""

    def close(self):
        """Close the logger."""
        pass


class TensorBoardLogger(Logger):
    """Class for logging training stats and other outputs to tensorboard.

    Optionally logs to wandb as well.
    """

    def __init__(self, config):
        super().__init__(config)
        self.log_dir = config.log_dir
        self.run_name = config.run_name
        self.track_wandb = config.track_wandb

        if self.track_wandb:
            import wandb

            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=config.wandb_group,
                sync_tensorboard=True,
                config=vars(config),
                name=config.run_name,
                save_code=True,
            )

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )

        self.uploaded_video_files = set()

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag: str, image: str, step: int):
        self.writer.add_image(tag, image, step)

    def log_figure(self, tag: str, figure, step: int):
        self.writer.add_figure(tag, figure, step)

    def log_matrix(self, tag: str, matrix: np.ndarray, step: int):
        assert len(matrix.shape) == 2
        fig, ax = plt.subplots(figsize=(matrix.shape[1], matrix.shape[0]))
        sns.heatmap(matrix, ax=ax, annot=True, cmap="YlGnBu")
        self.writer.add_figure(tag, fig, step)

    def upload_videos(self, step: int):
        if self.config.capture_video and self.config.track_wandb:
            import wandb

            video_filenames = list(self.config.video_dir.glob("*.mp4"))
            video_filenames.sort(key=lambda x: x.name)
            for filename in video_filenames:
                if filename not in self.uploaded_video_files:
                    print(f"{self.__class__.__name__} Uploading video {filename}")
                    wandb.log(  # type:ignore
                        {
                            "video": wandb.Video(  # type:ignore
                                str(filename.absolute())
                            )
                        },
                    )
                    self.uploaded_video_files.add(filename)

    def close(self):
        self.writer.close()
        if self.config.track_wandb:
            import wandb

            wandb.finish()


class NullLogger(Logger):
    """Placeholder logging class that does nothing.

    Useful for disabling logging without changing code.
    """

    def log_scalar(self, tag: str, value: float, step: int):
        pass

    def log_image(self, tag: str, image: str, step: int):
        pass

    def log_figure(self, tag: str, figure, step: int):
        pass

    def log_matrix(self, tag: str, matrix: np.ndarray, step: int):
        pass

    def upload_videos(self, step: int):
        pass
