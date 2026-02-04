import numpy as np
import open3d as o3d
import time, json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter, medfilt

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import coord_trans

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
    normal_world = Rz @ np.array([0, 1, 0])
    
    # 2. 光線の計算
    # 各点 P に対して ray_direction = P - O
    ray_directions = point_cloud_data - O  # (N, 3)
    
    # 3. 交点パラメータ t の一括計算
    # 線の式: L = O + t * ray_direction
    # 面の式: (L - C)・normal_world = 0
    # t = ((C - O)・normal_world) / (ray_direction・normal_world)
    
    # 分母 (N,)
    denominators = ray_directions @ normal_world
    
    # 分子 (スカラー)
    numerator = np.dot(C - O, normal_world)
    
    # ゼロ除算を避けるためのマスク
    valid_mask = np.abs(denominators) > 1e-6
    
    # t を計算 (N,)
    # 有効な分母以外は一旦 0 にして計算し、後でマスクをかける
    t = np.zeros_like(denominators)
    t[valid_mask] = numerator / denominators[valid_mask]
    
    # 4. 範囲チェック (t の条件)
    # 0 < t <= 1.0 : センサーと点 P の間に鏡がある
    t_mask = valid_mask & (t > 0.0) & (t <= 1.0)
    
    # 5. 交点の世界座標を計算
    # I = O + t * ray_direction (対象となる点のみ)
    # 効率化のため、t_mask が True の点だけ計算する
    indices = np.where(t_mask)[0]
    if len(indices) == 0:
        return np.zeros(point_cloud_data.shape[0], dtype=bool)
    
    I_world = O + t[indices, np.newaxis] * ray_directions[indices]
    
    # 6. 鏡のローカル座標系への変換と境界チェック
    # I_local = Rz^T @ (I_world - C)
    I_local_shifted = I_world - C
    # 行列演算で一括変換 (I_local_shifted @ Rz は 各行ベクトル v に v @ Rz を適用することと同等)
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

def reflection_sim(points, sensor_pos, sensor_ori, mirror_center, mirror_width, mirror_height, R):
    # 1. 鏡の法線ベクトルと平面の定義
    # 鏡のローカル座標でY軸正の方向を表面(法線)と仮定
    normal_local = np.array([0, 1, 0])
    normal_world = R @ normal_local
    C = np.array(mirror_center)
    S = np.array(sensor_pos)

    # 2. 仮想センサ位置 (S_v) の計算
    # センサを鏡面に対して対称移動させる
    dist_s = np.dot(S - C, normal_world)
    S_v = S - 2 * dist_s * normal_world

    # 3. 反射源の抽出 (仮想センサから見て、鏡の枠内にある点を特定)
    # 仮想センサから各点へのベクトル
    O_v = S_v
    ray_directions = points - O_v
    
    # 鏡面との交差判定: t = (C - O_v)・n / (ray・n)
    numerator = np.dot(C - O_v, normal_world)
    denominator = np.dot(ray_directions, normal_world)
    
    # 分母が0に近い(面と平行な光線)を除外
    valid_denom = np.abs(denominator) > 1e-6
    t = np.zeros(len(points))
    t[valid_denom] = numerator / denominator[valid_denom]
    
    # 交点が仮想センサと実体点Pの間にある (0 < t < 1) かつ、センサの「前」にある点のみ対象
    # (鏡の枠を通して点を見ている条件)
    mask = (t > 0) & (t < 1.0)
    
    # 交点 I の世界座標
    I_world = O_v + t[:, np.newaxis] * ray_directions
    
    # 4. 鏡の枠内(境界)チェック
    # 交点を鏡のローカル座標に変換
    I_local = (R.T @ (I_world - C).T).T
    
    # 鏡のローカル座標系: x=横幅方向, z=高さ方向 と仮定
    half_w = mirror_width / 2.0
    half_h = mirror_height / 2.0
    
    in_boundary = (I_local[:, 0] >= -half_w) & (I_local[:, 0] <= half_w) & \
                  (I_local[:, 2] >= -half_h) & (I_local[:, 2] <= half_h)
    
    final_mask = mask & in_boundary
    P_source = points[final_mask]

    # 5. 鏡像 (P_virtual) の生成
    # 反射源 P_source を鏡面に対して反転させる
    # P' = P - 2 * ((P - C)・n) * n
    if len(P_source) > 0:
        dist_p = np.sum((P_source - C) * normal_world, axis=1)
        P_virtual = P_source - 2 * dist_p[:, np.newaxis] * normal_world
    else:
        P_virtual = np.empty((0, 3))

    return P_virtual, P_source

def save_pointcloud_pcd(points, output_dir, filename):
    """
    ポイントクラウドをPCDファイルとして指定ディレクトリに保存
    
    Args:
        points: (N, 3) numpy array のポイントクラウドデータ
        output_dir: 出力ディレクトリパス (str または Path)
        filename: ファイル名 (拡張子なしまたは.pcd含む)
    
    Returns:
        保存されたファイルのパス (Path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイル名に .pcd 拡張子を追加
    if not filename.endswith('.pcd'):
        filename = filename + '.pcd'
    
    filepath = output_dir / filename
    
    # ポイントクラウドオブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # PCDファイルとして保存
    o3d.io.write_point_cloud(str(filepath), pcd)

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
    
def unstability(file, window_size=15, polyorder=2):
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(file)

    # 1. 隣接フレーム間の座標差分を計算
    dx = np.diff(gt_x)
    dy = np.diff(gt_y)
    
    # 2. 移動距離（＝速さ × dt）を計算
    # これが「重み」になります
    distances = np.sqrt(dx**2 + dy**2)
    
    # 3. 進行方向（ラジアン）を計算
    move_direction_rad = np.arctan2(dy, dx)
    
    # 4. 進行方向の「変化（回転）」を計算する場合（オプション）
    # 単なる方向の絶対値ではなく、方向が「どれだけ変わったか」の方が攻撃には重要かもしれません
    # direction_diff = np.diff(move_direction_rad)
    # direction_diff = (direction_diff + np.pi) % (2 * np.pi) - np.pi
    
    # 5. 重み付け：方向（の絶対値） × 移動距離
    weighted_score = np.abs(move_direction_rad) * distances
    
    # 6. 配列の長さを合わせる（先頭に0を挿入）
    # weighted_score は np.diff を2回使うなら2つ、1回なら1つ挿入
    weighted_consistent = np.insert(weighted_score, 0, 0.0)
    
    # 正規化（オプション）：最大値を1にするなど
    if np.max(weighted_consistent) > 0:
        weighted_consistent /= np.max(weighted_consistent)

    if len(weighted_consistent) < window_size:
        return weighted_consistent
    
    denoised = medfilt(weighted_consistent, kernel_size=5)

    # スパイクを抑えつつ、信号の立ち上がりなどを綺麗に残せる
    smoothed = savgol_filter(denoised, window_size, polyorder)

    # 4. 0-1 正規化 (Min-Max Scaling)
    s_min = np.min(smoothed)
    s_max = np.max(smoothed)
    
    if s_max - s_min > 1e-9:
        normalized_score = (smoothed - s_min) / (s_max - s_min)
    else:
        # すべて同じ値、または最大値が極めて小さい場合
        normalized_score = np.zeros_like(smoothed)
        
    return normalized_score

def simulator(param_x, param_y, param_yaw_center, param_swing_speed=7.0, param_swing_range=120):

    # --- 設定読み込み ---
    with open('conditions.json', 'r') as f: conditions = json.load(f)

    # --- Config ---
    bag_path = Path(conditions['main']['bag_path'])
    gt_path = Path(conditions['main']['gt_path'])
    #map_path = Path(conditions['main']['map_path'])
    lidar_topic_in = conditions['main']['lidar_topic']
    mirror_width, mirror_height = conditions['mirror']['width'], conditions['mirror']['height']

    # 最適化のためにconfigからの読み込みはしない
    mirror_yaw_base, swing_speed, swing_range = param_yaw_center, param_swing_speed, param_swing_range
    topic_length, lidar_freq = conditions['lidar']['topic_length'], conditions['lidar']['frequency']
    fov_azi = conditions['lidar']['fov'] / 2

    # --- Load Data ---
    #num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_path) # txt file
    unstability_score_array = unstability(gt_path)

    # set mirror position
    mirror_height = get_sensor_height(gt_x, gt_y, gt_z, param_x, param_y)
    mirror_center = [param_x, param_y, mirror_height]

    # --- 2. 初期姿勢(クオータニオン含む)の設定 ---
    cnt = 0

    # score record variables
    occlusion_score_num, reflect_score_num, unstability_score_num = 0, 0, 0
    
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

                    local_points_filtered = filter_by_azimuth(local_points, fov_azi)

                    sensor_pos = [gt_x[load_num], gt_y[load_num], gt_z[load_num]]
                    sensor_quat = [gt_qw[load_num], gt_qx[load_num], gt_qy[load_num], gt_qz[load_num]]
                    #wx, wy, wz = coord_trans.local_to_world(local_points, sensor_pos, sensor_quat) 
                    wx, wy, wz = coord_trans.local_to_world(local_points_filtered, sensor_pos, sensor_quat) # fovを制限
                    lidar_points_world = np.vstack((wx, wy, wz)).T # Input point cloud : size (N, 3)

                    is_reflected = faster_check_intersection(
                        lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos)       

                    P_visible, P_occlusion = lidar_points_world[~is_reflected], lidar_points_world[is_reflected] # センサから見て±90degだけ遮蔽      

                    #occlusion_score = P_occlusion.shape[0]
                    #occlusion_score_num += occlusion_score # 鏡によって隠れる点群のscoreを合計
                    P_virtual_fov = np.empty((0, 3))

                    mirror_yaw = decide_mirror_yaw_triangular(mirror_yaw_base, swing_range, swing_speed, cnt / lidar_freq)
                    
                    yaw_rad = np.deg2rad(mirror_yaw)
                    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])

                    P_virtual_raw, _ = reflection_sim(lidar_points_world, sensor_pos, sensor_quat, 
                                                        mirror_center, mirror_width, mirror_height, Rz) # センサから見て±90degだけ鏡像
                    
                    simulated_points = np.vstack((P_occlusion, P_virtual_raw))

                    # debug visualization
                    dx = mirror_center[0] - sensor_pos[0]
                    dy = mirror_center[1] - sensor_pos[1]
                    # センサから鏡への方位角 (deg)
                    mirror_dir_from_sensor = np.degrees(np.arctan2(dy, dx))

                    if np.abs(mirror_dir_from_sensor) <= 90:
                        # score calculation
                        N_total = local_points.shape[0]
                        occlusion_score_num += (P_occlusion.shape[0] / N_total)
                        reflect_score_num += (P_virtual_raw.shape[0] / N_total)
                        unstability_score_num += (((P_occlusion.shape[0] + P_virtual_raw.shape[0]) / N_total) * unstability_score_array[load_num])
                    else:
                        occlusion_score_num += 0
                        reflect_score_num += 0
                        unstability_score_num += 0

                    """
                    plt.clf()
                    # 1. シミュレーション点群のプロット（遮蔽点=青、鏡像点=赤）
                    plt.scatter(P_visible[:, 0], P_visible[:, 1], s=2, c='blue', label='Raw points')
                    plt.scatter(P_virtual_raw[:, 0], P_virtual_raw[:, 1], s=2, c='red', label='Virtual Points')
                    
                    # 2. センサー位置（緑の点）
                    plt.scatter(sensor_pos[0], sensor_pos[1], s=100, c='green', marker='o', label='Sensor')
                    
                    # 3. 鏡の中心位置（黒のX）
                    plt.scatter(mirror_center[0], mirror_center[1], s=100, c='black', marker='x', label='Mirror Center')
                    
                    # 4. 鏡の面と向き（法線方向）の描画
                    m_rad = np.deg2rad(mirror_yaw)
                    m_half_w = mirror_width / 2.0
                    # 鏡の面（実線）
                    plt.plot([mirror_center[0] - m_half_w * np.sin(m_rad), mirror_center[0] + m_half_w * np.sin(m_rad)],
                                [mirror_center[1] + m_half_w * np.cos(m_rad), mirror_center[1] - m_half_w * np.cos(m_rad)],
                                'k-', linewidth=2)
                    # 鏡の向き/法線（矢印）
                    plt.arrow(mirror_center[0], mirror_center[1], np.cos(m_rad), np.sin(m_rad),
                                head_width=0.3, fc='black', ec='black')
                    
                    # 鏡の向きに黒い鎖線を引く (任意)
                    line_len = 5.0
                    plt.plot([mirror_center[0], mirror_center[0] + line_len * np.cos(m_rad)],
                             [mirror_center[1], mirror_center[1] + line_len * np.sin(m_rad)],
                             'k--', linewidth=1, alpha=0.5)

                    plt.axis('equal')
                    # タイトルに鏡の自体の向き(Yaw)と、センサからみた方向(Mirror Dir)を表示
                    plt.title(f"Frame: {load_num} | Mirror Yaw: {mirror_yaw:.1f}$^\circ$ | Mirror Dir: {mirror_dir_from_sensor:.1f}$^\circ$")
                    plt.legend(loc='upper right')
                    plt.grid(True, linestyle=':', alpha=0.5)
                    plt.pause(0.01) # 短い一時停止でアニメーション表示
                    """

                    load_num += 1
                    cnt += 1

    print(f"occ score:{occlusion_score_num:.3f} | ref score:{reflect_score_num:.3f} | uns score:{unstability_score_num:.3f}")
    overall_score = occlusion_score_num + reflect_score_num + unstability_score_num # 鏡に隠れる点群と反射によって入り込む点群のscoreの和を計算
    return overall_score

def filter_by_azimuth(points, azimuth_limit=60):
    """
    ローカル座標系のポイントクラウドを方位角でフィルター
    
    Args:
        points: (N, 3) local_points (ローカル座標系)
        azimuth_limit: 方位角の制限 (度, デフォルト±60度)
    
    Returns:
        フィルタリングされたポイント
    """
    if len(points) == 0:
        return points
    
    x = points[:, 0]
    y = points[:, 1]
    
    azimuth = np.arctan2(y, x)
    azimuth_limit_rad = np.deg2rad(azimuth_limit)
    
    mask = np.abs(azimuth) <= azimuth_limit_rad
    
    return points[mask]