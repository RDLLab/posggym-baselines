import uuid
import os

import posggym
from posggym.wrappers import RecordVideo


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1';
    false values are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


class NoOverwriteRecordVideo(RecordVideo):
    """Record video without overwriting existing videos."""

    def __init__(self, env: posggym.Env, video_folder: str, **kwargs):
        if os.path.exists(video_folder):
            video_folder = os.path.join(video_folder, str(uuid.uuid4()))
        super().__init__(env, video_folder, **kwargs)
