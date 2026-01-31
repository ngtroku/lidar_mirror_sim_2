
import numpy as np
import open3d as o3d
import json
from pathlib import Path
import os
from datetime import datetime

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

import coord_trans

def binary_to_xyz(binary):
    """Livox custom binary to XYZ numpy array"""
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def rosbag_writer(): # シミュレーションした点群をrosbagに変換
    # --- Load External Configs ---

    with open('config.json', 'r') as f:
        config = json.load(f)

    bag_path = Path(config['main']['bag_path'])
    output_directory = Path(config['main']['output_directory'])

    # output directoryを作成
    output_directory.mkdir(parents=True, exist_ok=True)
    
    if not bag_path.exists():
        print(f"Error: {bag_path} does not exist.")
        return
    
    lidar_topic_in = config['lidar']['lidar_topic'] 
    topic_length = config['lidar']['topic_length'] 

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC)
    cnt = 0 

    # make directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with bag name
    bag_filename = bag_path.stem  # Get filename without extension

    with AnyReader([bag_path], default_typestore=typestore) as reader:

        connections = [x for x in reader.connections if x.topic == lidar_topic_in] # LiDAR only rosbag
        for connection, timestamp_msg, rawdata in reader.messages(connections=connections):

            if connection.topic == lidar_topic_in:

                msg = reader.deserialize(rawdata, connection.msgtype)
                iteration = int(msg.data.shape[0]/topic_length)
                bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                lx, ly, lz = binary_to_xyz(bin_points)
                points = np.vstack((lx, ly, lz)).T

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # Use output_directory and auto-generated filename
                output_path = output_directory / f"{cnt:04d}.pcd"
                o3d.io.write_point_cloud(str(output_path), pcd)

                cnt += 1

            else:
                pass