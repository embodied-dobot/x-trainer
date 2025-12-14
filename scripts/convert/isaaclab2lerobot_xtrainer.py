import os
import h5py
import numpy as np

from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
"""
NOTE: Please use the environment of lerobot.

Because lerobot is rapidly developing, we don't guarantee the compatibility for the latest version of lerobot.
Currently, the commit we used is https://github.com/huggingface/lerobot/tree/v0.3.3
"""

# Feature definition for xtrainer_follower
XTRAINER_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (16,),
        "names": [
            # --- Arm 1 (Left) ---
            "J1_1.pos", "J1_2.pos", "J1_3.pos",
            "J1_4.pos", "J1_5.pos", "J1_6.pos",
            "J1_7.pos", "J1_8.pos",

            # --- Arm 2 (Right) ---
            "J2_1.pos", "J2_2.pos", "J2_3.pos",
            "J2_4.pos", "J2_5.pos", "J2_6.pos",
            "J2_7.pos", "J2_8.pos",
        ]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (16,),
        "names": [
            # --- Arm 1 (Left) ---
            "J1_1.pos", "J1_2.pos", "J1_3.pos",
            "J1_4.pos", "J1_5.pos", "J1_6.pos",
            "J1_7.pos", "J1_8.pos",

            # --- Arm 2 (Right) ---
            "J2_1.pos", "J2_2.pos", "J2_3.pos",
            "J2_4.pos", "J2_5.pos", "J2_6.pos",
            "J2_7.pos", "J2_8.pos",
        ]

    },
    "observation.images.top": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    }
}

def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    return joint_pos.astype(np.float32)

def process_xtrainer_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    try:
        actions = np.array(demo_group['actions'])
        left_joint_pos = np.array(demo_group['obs/left_joint_pos_rel'])
        right_joint_pos = np.array(demo_group['obs/right_joint_pos_rel'])
        
        left_images = np.array(demo_group['obs/left_wrist'])
        right_images = np.array(demo_group['obs/right_wrist'])
        top_images = np.array(demo_group['obs/top'])
    except KeyError:
        print(f'Demo {demo_name} is not valid, skip it')
        return False

    if actions.shape[0] < 10:
        print(f'Demo {demo_name} has less than 10 frames, skip it')
        return False

    # preprocess actions and joint pos
    actions = preprocess_joint_pos(actions)
    left_joint_pos = preprocess_joint_pos(left_joint_pos)
    right_joint_pos = preprocess_joint_pos(right_joint_pos)
    joint_pos = np.concatenate([left_joint_pos, right_joint_pos], axis=1)

    assert actions.shape[0] == joint_pos.shape[0] == top_images.shape[0] == left_images.shape[0] == right_images.shape[0]
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc='Processing each frame'):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            "observation.images.top": top_images[frame_index],
            "observation.images.left_wrist": left_images[frame_index],
            "observation.images.right_wrist": right_images[frame_index],
            "task": task,
        }
        dataset.add_frame(frame=frame)

    return True

def convert_isaaclab_to_lerobot(file_name):
    """NOTE: Modify the following parameters to fit your own dataset"""
    repo_id = 'dstx123/xtrainer_lift_cube'
    robot_type = 'xtrainer_follower'
    fps = 30
    hdf5_root = './datasets'
    hdf5_files = [os.path.join(hdf5_root, file_name)]
    task = 'Grab cube and place into plate'
    push_to_hub = False

    """convert to LeRobotDataset"""
    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=XTRAINER_FEATURES
    )

    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f'[{hdf5_id+1}/{len(hdf5_files)}] Processing hdf5 file: {hdf5_file}')
        with h5py.File(hdf5_file, 'r') as f:
            demo_names = list(f['data'].keys())
            print(f'Found {len(demo_names)} demos: {demo_names}')

            for demo_name in tqdm(demo_names, desc='Processing each demo'):
                demo_group = f['data'][demo_name]
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f'Demo {demo_name} is not successful, skip it')
                    continue

                valid = process_xtrainer_data(dataset, task, demo_group, demo_name)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f'Saving episode {now_episode_index} successfully')

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == '__main__':
    file_name="lift_cube.hdf5"
    convert_isaaclab_to_lerobot(file_name)
