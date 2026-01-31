
import numpy as np
import open3d as o3d

def local_to_world(local_points, sensor_pos, sensor_quat):

    R = o3d.geometry.get_rotation_matrix_from_quaternion(sensor_quat)

    world_points = (R @ local_points.T).T + np.array(sensor_pos)

    world_x, world_y, world_z = world_points[:,0], world_points[:,1], world_points[:,2]

    return world_x, world_y, world_z

def world_to_local(world_points, sensor_pos, sensor_quat):

    R = o3d.geometry.get_rotation_matrix_from_quaternion(sensor_quat)
    R_inv = R.T  # 逆行列は転置行列で計算可能

    local_points = (R_inv @ (world_points - np.array(sensor_pos)).T).T

    #local_x, local_y, local_z = local_points[:,0], local_points[:,1], local_points[:,2]

    return local_points