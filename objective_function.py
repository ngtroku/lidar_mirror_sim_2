import numpy as np
import open3d as o3d
import time, json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from scipy.spatial.transform import Rotation as R

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import coord_trans
import eigen_decomposition

def binary_to_xyz(binary):
    """Livox custom binary to XYZ numpy array"""
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def filter_by_fov(points, sensor_pos, sensor_quat, fov_h=120, fov_v=25):
    if len(points) == 0:
        return points

    # World -> Sensor Local
    R_sensor = o3d.geometry.get_rotation_matrix_from_quaternion(sensor_quat)
    diff = points - np.array(sensor_pos)
    points_local = diff @ R_sensor 

    x = points_local[:, 0]
    y = points_local[:, 1]
    z = points_local[:, 2]

    azimuth = np.arctan2(y, x)
    hypot_xy = np.hypot(x, y)
    elevation = np.arctan2(z, hypot_xy)

    fov_h_rad = np.deg2rad(fov_h)
    fov_v_rad = np.deg2rad(fov_v)

    mask = (x > 0) & \
           (np.abs(azimuth) <= fov_h_rad / 2.0) & \
           (np.abs(elevation) <= fov_v_rad / 2.0)

    return points[mask]

def decide_mirror_yaw_triangular(base_yaw, swing_range, rotation_speed, current_time):
    if swing_range == 0 or rotation_speed == 0:
        return base_yaw

    cycle_distance = 4 * swing_range

    total_distance = rotation_speed * current_time
    cycle_pos = total_distance % cycle_distance
    
    offset = 0.0
    
    if cycle_pos < swing_range:
        offset = cycle_pos

    elif cycle_pos < 3 * swing_range:
        offset = 2 * swing_range - cycle_pos

    else:
        offset = cycle_pos - 4 * swing_range
        
    return base_yaw + offset

def get_mirror_yaw_towards_sensor(mirror_center, sensor_pos):
    """
    鏡がセンサに向くように mirror_yaw を計算
    
    Args:
        mirror_center: [x, y, z] 鏡の設置位置
        sensor_pos: [x, y, z] センサの位置
    
    Returns:
        mirror_yaw: 鏡の向く方向 (度)
    """
    mirror_center = np.array(mirror_center)
    sensor_pos = np.array(sensor_pos)
    
    # センサから鏡への向きベクトル
    direction = sensor_pos[:2] - mirror_center[:2]
    
    # atan2で角度を計算（ラジアン）
    yaw_rad = np.arctan2(direction[1], direction[0])
    
    # 度に変換
    yaw_deg = np.degrees(yaw_rad)
    
    return yaw_deg

def faster_check_intersection(point_cloud_data: np.ndarray, center: list, width: float, height: float, yaw_angle: float, sensor_pos: list) -> np.ndarray:

    # 1. 定数の準備
    O = np.array(sensor_pos)
    C = np.array(center)
    half_width = width / 2.0
    half_height = height / 2.0

    yaw_rad = np.deg2rad(yaw_angle)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    
    # ローカルから世界座標への回転行列 Rz
    Rz = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0, 0, 1]
    ])
    
    # 鏡の法線ベクトル（世界座標系）
    # このベクトルが向いている方が「表面」
    normal_world = Rz @ np.array([0, 1, 0])
    
    # 2. 光線の計算
    ray_directions = point_cloud_data - O  # (N, 3)
    
    # 3. 交点パラメータ t の一括計算
    # 分母 (N,) : 光線ベクトルと法線ベクトルの内積
    denominators = ray_directions @ normal_world
    
    # 分子 (スカラー)
    numerator = np.dot(C - O, normal_world)
    
    # ゼロ除算を避けるためのマスク
    valid_mask = np.abs(denominators) > 1e-6
    
    # t を計算 (N,)
    t = np.zeros_like(denominators)
    t[valid_mask] = numerator / denominators[valid_mask]
    
    # 4. 範囲チェック (t の条件) + 【追加】表面判定
    # 0 < t <= 1.0 : センサーと点 P の間に鏡がある
    # denominators < 0 : ビームが法線と逆方向（＝鏡の表側）から入射している
    t_mask = valid_mask & (t > 0.0) & (t <= 1.0) & (denominators < 0)
    
    # 5. 交点の世界座標を計算
    indices = np.where(t_mask)[0]
    if len(indices) == 0:
        return np.zeros(point_cloud_data.shape[0], dtype=bool)
    
    I_world = O + t[indices, np.newaxis] * ray_directions[indices]
    
    # 6. 鏡のローカル座標系への変換と境界チェック
    I_local_shifted = I_world - C
    I_local = I_local_shifted @ Rz
    
    x_local = I_local[:, 0]
    z_local = I_local[:, 2]
    
    # 境界チェック
    inside_mask = (np.abs(x_local) <= half_width) & (np.abs(z_local) <= half_height)
    
    # 最終的な結果配列の作成
    is_reflected = np.zeros(point_cloud_data.shape[0], dtype=bool)
    is_reflected[indices[inside_mask]] = True
    
    return is_reflected

# --- 光線チェック関数 (KDTree使用) ---
def check_line_of_sight(pcd_tree, start_pos, end_pos, step=0.2, radius=0.15):

    vec = np.array(end_pos) - np.array(start_pos)
    dist = np.linalg.norm(vec)
    if dist < 1e-3: return True
    
    direction = vec / dist

    # 自分のすぐ近く(0.5m)と、鏡の直前(0.2m)はチェックしない
    current_dist = 0.5 
    target_dist = dist - 0.2

    while current_dist < target_dist:
        check_point = np.array(start_pos) + direction * current_dist
        
        # 半径 radius 内に障害物点群があるか検索
        k, _, _ = pcd_tree.search_radius_vector_3d(check_point, radius)
        
        if k > 0:
            return False # 障害物あり

        current_dist += step

    return True # 障害物なし

def down_sampling_sim(points, horizontal_res, vertical_res):

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    dists = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / dists)

    res_h_rad = np.radians(horizontal_res)
    res_v_rad = np.radians(vertical_res)

    azi_indices = np.round(azimuth / res_h_rad)
    ele_indices = np.round(elevation / res_v_rad)

    combined_idx = np.stack((azi_indices, ele_indices), axis=1)
    _, unique_indices = np.unique(combined_idx, axis=0, return_index=True)

    return points[unique_indices]

def reflection_sim(points, sensor_pos, mirror_center, mirror_width, mirror_height, R):
    normal_world = R @ np.array([0, 1, 0])
    C = np.array(mirror_center)
    S = np.array(sensor_pos)

    # 1. 【追加】 そもそもセンサが鏡の裏にいたら反射はゼロ
    dist_s = np.dot(S - C, normal_world)
    if dist_s <= 0:
        return np.empty((0, 3)), np.empty((0, 3))

    # 2. 仮想センサ位置 (S_v) の計算
    S_v = S - 2 * dist_s * normal_world

    # 3. 反射源の抽出
    ray_directions = points - S_v
    denominator = ray_directions @ normal_world # (N,)
    numerator = np.dot(C - S_v, normal_world) # スカラー
    
    valid_denom = np.abs(denominator) > 1e-6
    t = np.zeros(len(points))
    t[valid_denom] = numerator / denominator[valid_denom]
    
    # 4. 【追加・修正】 条件の厳格化
    # t: 0 < t < 1 (センサと点の間に鏡がある)
    # denominator > 0: 仮想センサ(裏)から点(表)へ向かう光線が法線と同じ向き
    # (＝鏡を裏から表へ突き抜けている)
    mask = (t > 0) & (t < 1.0) & (denominator > 0)
    
    I_world = S_v + t[:, np.newaxis] * ray_directions
    I_local = (I_world - C) @ R # 行列の形に注意
    
    half_w, half_h = mirror_width / 2.0, mirror_height / 2.0
    in_boundary = (np.abs(I_local[:, 0]) <= half_w) & (np.abs(I_local[:, 2]) <= half_h)
    
    final_mask = mask & in_boundary
    P_source = points[final_mask]

    # 5. 鏡像生成
    if len(P_source) > 0:
        # P_source は必ず鏡の「表」にある点のみになる
        dist_p = np.sum((P_source - C) * normal_world, axis=1)
        P_virtual = P_source - 2 * dist_p[:, np.newaxis] * normal_world
    else:
        P_virtual = np.empty((0, 3))

    return P_virtual, P_source

def get_sensor_height(gt_x, gt_y, gt_z, mirror_pos_x, mirror_pos_y):
    gt_points = np.vstack((gt_x, gt_y, gt_z)).T

    # generate kdtree
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)

    # search nearest point
    [k, idx, _] = gt_tree.search_knn_vector_3d(np.array([mirror_pos_x, mirror_pos_y, 0.0]), 1)
    if k > 0:
        nearest_index = idx[0]
        return gt_z[nearest_index]
    
    else:
        return 0
    
def filter_by_horizontal_fov(points_world, sensor_pos, sensor_quat, fov_deg):
    """
    ワールド座標系の点群を、センサの正面を中心とした水平視野角でフィルタリングする
    """
    if fov_deg >= 360.0:
        return points_world
        
    # ワールド座標からローカル座標へ変換
    local_points = coord_trans.world_to_local(points_world, sensor_pos, sensor_quat)
    
    # 水平方位角 (Azimuth) を計算: atan2(y, x) -> [-pi, pi]
    # フロントがx軸正の方向の場合
    angles_rad = np.arctan2(local_points[:, 1], local_points[:, 0])
    angles_deg = np.degrees(angles_rad)
    
    # 判定範囲 [-fov/2, +fov/2]
    half_fov = fov_deg / 2.0
    mask = (angles_deg >= -half_fov) & (angles_deg <= half_fov)
    
    return points_world[mask]
    
def debug_visualize_2d(sensor_pos, lidar_points, mirror_center, mirror_yaw, p_virtual):

    plt.clf() # 画面をクリア
    
    # 1. 周辺点群（グレー）
    plt.scatter(lidar_points[:, 0], lidar_points[:, 1], s=1, c='gray', alpha=0.3, label='Environment')
    
    # 2. センサー位置（青）
    plt.scatter(sensor_pos[0], sensor_pos[1], s=50, c='blue', marker='o', label='Sensor')
    
    # 3. 鏡の位置（赤の×）と向き
    plt.scatter(mirror_center[0], mirror_center[1], s=100, c='red', marker='x', label='Mirror')
    
    # 鏡の向きを線で表示（鏡の面）
    # yawに対して垂直なのが鏡の面方向
    mirror_rad = np.radians(mirror_yaw)
    # 鏡の幅を3mと仮定して表示
    m_half_w = 1.5
    dx = m_half_w * np.cos(mirror_rad + np.pi/2)
    dy = m_half_w * np.sin(mirror_rad + np.pi/2)
    plt.plot([mirror_center[0] - dx, mirror_center[0] + dx], 
             [mirror_center[1] - dy, mirror_center[1] + dy], 'r-', linewidth=2)
    
    # 鏡の法線（反射の向きを確認するため）
    nx = 1.0 * np.cos(mirror_rad)
    ny = 1.0 * np.sin(mirror_rad)
    plt.arrow(mirror_center[0], mirror_center[1], nx, ny, head_width=0.3, fc='r', ec='r')

    # 4. 反射点群（赤）
    if p_virtual.size > 0:
        plt.scatter(p_virtual[:, 0], p_virtual[:, 1], s=2, c='red', label='Virtual Points')

    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.pause(0.01) # 短い一時停止でアニメーション表示

def calc_unstability(x, y, window_size=15, polyorder=2):

    # 1. 隣接フレーム間の座標差分を計算
    dx = np.diff(x)
    dy = np.diff(y)

    distances = np.sqrt(dx**2 + dy**2)
    move_direction_rad = np.arctan2(dy, dx)

    weighted_score = np.abs(move_direction_rad) * distances
    weighted_consistent = np.insert(weighted_score, 0, 0.0)
    
    if np.max(weighted_consistent) > 0:
        weighted_consistent /= np.max(weighted_consistent)
    else:
        pass

    # 平滑化 & 正規化
    if len(weighted_consistent) < window_size:
        return weighted_consistent
    
    denoised_score = medfilt(weighted_consistent, kernel_size=5)
    smoothed_score = savgol_filter(denoised_score, window_size, polyorder)

    s_min = np.min(smoothed_score)
    s_max = np.max(smoothed_score)
    
    if s_max - s_min > 1e-9:
        normalized_score = (smoothed_score - s_min) / (s_max - s_min)
    else:
        normalized_score = np.zeros_like(smoothed_score)
        
    return normalized_score

def simulator(param_x, param_y, param_yaw_center, param_swing_speed=7.0, param_swing_range=120, debug=False):

    # --- 設定読み込み ---
    with open('config.json', 'r') as f: conditions = json.load(f)

    # --- Config ---
    bag_path = Path(conditions['main']['bag_path'])
    gt_path = Path(conditions['main']['gt_path'])
    lidar_topic_in = conditions['lidar']['lidar_topic']
    mirror_width, mirror_height = conditions['mirror']['width'], conditions['mirror']['height']

    # 最適化のためにconfigからの読み込みはしない
    mirror_yaw_base, swing_speed, swing_range = param_yaw_center, param_swing_speed, param_swing_range
    topic_length, lidar_freq = conditions['lidar']['topic_length'], conditions['lidar']['frequency']

    # --- Load Data ---
    #num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_path) # txt file

    # moving unstability calculation
    unstability = calc_unstability(gt_x, gt_y)

    # set mirror position
    mirror_height = get_sensor_height(gt_x, gt_y, gt_z, param_x, param_y)
    mirror_center = [param_x, param_y, mirror_height]

    # --- 2. 初期姿勢(クオータニオン含む)の設定 ---
    cnt = 0

    # score record variables
    occlusion_score_num = 0
    reflect_score_num = 0
    unstability_score_num = 0
    
    # GTの1フレーム目から初期姿勢行列を作成
    initial_pos = np.array([gt_x[0], gt_y[0], gt_z[0]])
    initial_quat = [gt_qx[0], gt_qy[0], gt_qz[0], gt_qw[0]] # scipy準拠 [x,y,z,w]
    r_matrix = R.from_quat(initial_quat).as_matrix()

    global_transform = np.identity(4)
    global_transform[:3, :3] = r_matrix
    global_transform[:3, 3] = initial_pos

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC)
    load_num = 0

    #max_frame_seq = max(num_gt)
    max_frame_seq = len(gt_x)
    # min_frame_seq = min(num_gt) # only load csv file

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == lidar_topic_in]
        msg_iter = reader.messages(connections=connections)

        for i, (connection, timestamp, rawdata) in enumerate(msg_iter):

            if connection.topic == lidar_topic_in:
                # stop if we've consumed all ground-truth frames
                # max_frame_seq is len(gt_x) (number of frames). Valid indices are 0..len-1,
                # so stop when load_num >= max_frame_seq to avoid indexing past the end.
                if load_num >= max_frame_seq:
                    break

                else:

                    # (1) Deserialize
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    iteration = int(msg.data.shape[0]/topic_length)
                    bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                    lx, ly, lz = binary_to_xyz(bin_points)

                    # (2) Local -> World (GTを使用してシミュレーション用の点群を生成)
                    local_points = np.vstack((lx, ly, lz)).T # Azimuth range [-180, 180] (deg)

                    sensor_pos = [gt_x[load_num], gt_y[load_num], gt_z[load_num]]
                    sensor_quat = [gt_qw[load_num], gt_qx[load_num], gt_qy[load_num], gt_qz[load_num]]
                    wx, wy, wz = coord_trans.local_to_world(local_points, sensor_pos, sensor_quat) 
                    lidar_points_world = np.vstack((wx, wy, wz)).T # Input point cloud : size (N, 3)

                    # 正規化用変数
                    #N_total = lidar_points_world.shape[0] # スキャン全体の点群数

                    is_reflected = faster_check_intersection(
                        lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos)      

                    P_visible, P_occlusion = lidar_points_world[~is_reflected], lidar_points_world[is_reflected]         

                    #occlusion_score = P_occlusion.shape[0] / N_total
                    #occlusion_score_num += occlusion_score # 鏡によって隠れる点群のscoreを合計
                    P_virtual_fov = np.empty((0, 3))

                    mirror_yaw = decide_mirror_yaw_triangular(mirror_yaw_base, swing_range, swing_speed, cnt / lidar_freq)
                    
                    yaw_rad = np.deg2rad(mirror_yaw)
                    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])

                    P_virtual_raw, _ = reflection_sim(lidar_points_world, sensor_pos, sensor_quat, 
                                                        mirror_center, mirror_width, mirror_height, Rz)
                    
                    #simulated_points = np.vstack((P_visible, P_virtual_raw))

                    # P_visible, P_virtual_rawを水平方位角でフィルタする
                    visible_fov = conditions['lidar']['horizontal_fov']

                    P_occlusion_filtered = filter_by_horizontal_fov(P_occlusion, sensor_pos, sensor_quat, visible_fov) # FOVでフィルタされた遮蔽点群    
                    P_visible_filtered = filter_by_horizontal_fov(P_visible, sensor_pos, sensor_quat, visible_fov) # FOVでフィルタされた元点群
                    P_virtual_filtered = filter_by_horizontal_fov(P_virtual_raw, sensor_pos, sensor_quat, visible_fov) # FOVでフィルタされた鏡像点群

                    N_total = P_occlusion_filtered.shape[0] + P_visible_filtered.shape[0]
                    weight = (P_occlusion_filtered.shape[0]+P_virtual_filtered.shape[0]) / N_total # 鏡の影響を受ける点群数による重み
                    simulated_points = np.vstack((P_visible_filtered, P_virtual_filtered))

                    # 鏡像関連スコア計算
                    occlusion_score = P_occlusion_filtered.shape[0] / N_total # 遮蔽スコア
                    reflection_score = P_virtual_filtered.shape[0] / N_total # 反射スコア

                    occlusion_score_num += occlusion_score
                    reflect_score_num += reflection_score

                    # scan matching安定性スコア計算
                    unstability_score_num += weight * unstability

                    # simulation debug
                    if debug:
                        debug_visualize_2d(sensor_pos=sensor_pos[:2], lidar_points=simulated_points, mirror_center=mirror_center[:2], 
                                mirror_yaw=mirror_yaw, p_virtual=P_virtual_raw)
                    else:
                        pass

                    load_num += 1
                    cnt += 1

    #overall_score = occlusion_score_num + reflect_score_num + anisotropic_score_num - rotation_score_num
    overall_score = occlusion_score_num + reflect_score_num + unstability_score_num
    return overall_score
