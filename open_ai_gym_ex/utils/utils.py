import cv2
import os


def convert_images_to_video(images_path:str, fps:int=4) -> str:
    """Converts folder of images to a webm video.

    Args:
        images_path (str): Path to images folder.

    Returns:
        str: Path to video file.
    """
    ###### Extensions and their encodings #########
    # webm, vp80 --- mp4, mp4v --- mkv, avc1
    images_path = os.path.normpath(images_path)
    images = [img for img in os.listdir(images_path) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(images_path, images[0]))
    # wandb.log({"img": [wandb.Image(os.path.join(images_path, images[0]), caption="episode")]})
    height, width, layers = frame.shape
    video_path = os.path.join(os.path.dirname(images_path),f"{os.path.basename(images_path)}.webm")
    # NOTE: WandB does not support mp4v encoding https://github.com/wandb/client/issues/2143
    fourcc = cv2.VideoWriter_fourcc(*'vp80')  # encoding
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for img in images:
        video.write(cv2.imread(os.path.join(images_path, img)))
    cv2.destroyAllWindows()
    video.release()
    return video_path