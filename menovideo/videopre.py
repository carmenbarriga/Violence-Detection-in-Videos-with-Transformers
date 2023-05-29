"""
Part of the code adapted from:
Almamon Rasool Abdali, DeVTr,
https://github.com/mamonraab/Data-efficient-video-transformer/

MIT License
Copyright (c) 2021
"""

import cv2
import numpy as np
import torch

from skimage.transform import resize
from torch.utils.data import Dataset


def capture(filename, timesep, rgb, h, w):
    # Create an array to store the video frames after being processed
    frames = np.zeros((timesep, rgb, h, w), dtype=np.float)

    # VideoCapture object to open and read the video
    video_capture = cv2.VideoCapture(filename)

    # To check if the VideoCapture object was able to open the video
    if video_capture.isOpened():
        # To keep track of how many frames have been stored in the frames array
        frames_counter = 0
        while frames_counter < timesep:
            # Read the next frame
            rval, frame = video_capture.read()

            # Check if there are no more frames available
            if not rval:
                break

            # Resize the original frame to the specified dimensions (h, w, rgb) keeping its original aspect ratio
            frame = resize(frame, (h, w, rgb))
            # To add an extra dimension (1, h, w, rgb)
            frame = np.expand_dims(frame, axis=0)
            # Moves axis -1 (last axis) to index 1 (1, rgb, h, w)
            frame = np.moveaxis(frame, -1, 1)

            # Normalization of the pixel values of the frame (if necessary)
            if np.max(frame) > 1:
                frame = frame / 255.0

            # Store the processed frame in the corresponding position within the frames array
            frames[frames_counter][:] = frame

            frames_counter += 1

        del frame
        del rval

    return frames


class TaskDataset(Dataset):
    def __init__(self, data, time_steps=10, color_channels=3, height=90, width=90):
        """
        Args:
            data: pandas dataframe that contains the paths to the video files with their labels
            time_steps: number of frames
            color_channels: number of color channels
            height: height of frames
            width: width of frames
        """
        self.data_locations = data
        self.time_steps, self.color_channels, self.height, self.width = time_steps, color_channels, height, width

    def __len__(self):
        return len(self.data_locations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # To process the video and get its frames
        video = capture(self.data_locations.iloc[idx, 0], self.time_steps, self.color_channels, self.height, self.width)

        # Dictionary containing the processed video and its corresponding label
        sample = {
            'video': torch.from_numpy(video),
            'label': torch.from_numpy(np.asarray(self.data_locations.iloc[idx, 1]))
        }

        return sample
