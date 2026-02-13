import time
from typing import List, Optional, Tuple
import numpy as np
from .camera import CameraDriver
import cv2
import pyrealsense2 as rs

def get_device_ids() -> List[str]:

    ctx = rs.context()
    devices = ctx.query_devices()
    # print(devices)
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        import pyrealsense2 as rs
        print("init", device_id)
        self._device_id = device_id
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device_id)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 90)
        self._pipeline.start(config)
        self._flip = flip
        # print(device_id)
        for _ in range(50):
            self.read()

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        return image, depth

def _debug_read(camera, save_datastream=False):

    cv2.namedWindow("image")
    # cv2.namedWindow("depth")
    counter = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while True:
        # time.sleep(0.1)
        tic = time.time()
        image, depth = camera.read()

        _, image_ = cv2.imencode('.jpg', image, encode_param)

        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        # cv2.imshow("depth", depth)
        toc = time.time()
        print(image_.shape, toc - tic)
        counter += 1


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
