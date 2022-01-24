import wandb
from gym.wrappers import RecordEpisodeStatistics, RecordVideo
from pandas.io.json._normalize import nested_to_record  # For WandB logging
from .utils import convert_images_to_video
import cv2
import tempfile
from pathlib import Path
import inspect


def filter_dict(dict_kwargs, func):
    """Prevents issue when passing unused kwargs to function. Modified from https://stackoverflow.com/a/44052550."""
    sig = inspect.signature(func)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD and param.default != param.empty]
    filtered_dict = {filter_key:dict_kwargs[filter_key] for filter_key in filter_keys if filter_key in dict_kwargs}
    return filtered_dict


class WandBMonitor(RecordVideo, RecordEpisodeStatistics):
    def __init__(self, env, video_folder=None, **kwargs) -> None:
        if video_folder is None:
            self.video_tmp_folder = tempfile.TemporaryDirectory(prefix=f"wandb_video_")
            video_folder = self.video_tmp_folder.name
        else:
            self.video_tmp_folder = None
        filtered_record_eps = filter_dict(kwargs, RecordEpisodeStatistics.__init__)
        filtered_record_vids = filter_dict(kwargs, RecordVideo.__init__)
        filtered_wandb = filter_dict(kwargs, wandb.init)
        RecordEpisodeStatistics.__init__(self, env, **filtered_record_eps)
        RecordVideo.__init__(self, env, video_folder, **filtered_record_vids)
        wandb.init(monitor_gym=True, **filtered_wandb)
        self.global_step = 0

    def step(self, action):
        obs, rewards, dones, infos = super().step(action)
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        wandb.log({"global_step": self.global_step})
        for i in range(len(dones)):
            if dones[i]:
                wandb.log(nested_to_record({"episode": infos[i]["episode"]}, sep="/"))
        self.global_step += 1
        return (
            obs,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )
    
    def close(self):
        super().close()
        self.close_video_recorder()
        if self.video_tmp_folder:
            self.video_tmp_folder.cleanup()
        wandb.finish()


class CustomWandBMonitor(RecordEpisodeStatistics):
    def __init__(self, env, **kwargs) -> None:
        super().__init__(env)
        wandb.init(**kwargs)
        self.global_step = 0
        self.img_dir = self.create_img_dir()
        self.RECORD = False

    def create_img_dir(self):
        return tempfile.TemporaryDirectory(prefix=f"episode_{self.episode_count}_")

    def step(self, action):
        episode_count = self.episode_count  # Store before incremented
        observations, rewards, dones, infos = super().step(
            action
        )
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        wandb.log({"global_step": self.global_step})
        if self.RECORD:
            im = self.render(mode="rgb_array")
            file_path = Path(self.img_dir.name, f'global_frame_{self.global_step}.jpg')
            cv2.imwrite(str(file_path), im)
        for i in range(len(dones)):
            if dones[i]:
                if self.RECORD:
                    video_path = convert_images_to_video(self.img_dir.name, fps=20)
                    wandb.log({f"episode-{episode_count}": wandb.Video(video_path, fps=20, format="webm")})
                wandb.log(nested_to_record({"episode": infos[i]["episode"]}, sep="/"))
                self.img_dir.cleanup()
                self.img_dir = self.create_img_dir()
        self.global_step += 1
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )
    
    def close(self):
        super().close()
        wandb.finish()
