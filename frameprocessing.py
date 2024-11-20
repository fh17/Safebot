import numpy as np
import glob
import cv2
import imageio


class reader:
    def __init__(self, video_dir, H, W, downscale=1, shorter_side=None, zfar=np.inf):
        self.video_dir = video_dir
        self.downscale = downscale
        self.zfar = zfar
        self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3, 3)

        if shorter_side is not None:
            self.downscale = shorter_side / min(H, W)

        self.H, self.W = H, W
        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

    def get_frame(self):
        color = imageio.imread(f'{self.video_dir}/1.png')
        depth = imageio.imread(f'{self.video_dir}/2.png')

        color = color[..., :3]  # Ensure color has 3 channels
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        depth = depth / 1e3  # Convert depth to meters
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0

        # Return the processed color and depth images
        return color, depth

