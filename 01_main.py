import numpy as np
import open3d as o3d
import json, subprocess, time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import optuna

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import coord_trans
import preprocessing
import postprocess
import objective_function
import eigen_decomposition
import error_estimate

def get_mirror_orientation_yaw(mirror_pos_x, mirror_pos_y):
    # load config
    with open('conditions.json', 'r') as f:
        config = json.load(f)
    
    gt_path = Path(config['main']['gt_path'])
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_path)
    
    # search nearest point
    gt_coords_2d = np.vstack((gt_x, gt_y)).T
    tree = KDTree(gt_coords_2d)
    query_point = [mirror_pos_x, mirror_pos_y]
    dist, idx = tree.query(query_point)

    # find yaw orientation
    target_x, target_y = gt_x[idx], gt_y[idx]
    dx, dy = target_x - mirror_pos_x, target_y - mirror_pos_y
    yaw_rad = np.arctan2(dy, dx)
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg

def binary_to_xyz(binary):
    """Livox custom binary to XYZ numpy array"""
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def get_sensor_height(gt_x, gt_y, gt_z, mirror_pos_x, mirror_pos_y):
    gt_points = np.vstack((gt_x, gt_y, gt_z)).T
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)

    [k, idx, _] = gt_tree.search_knn_vector_3d(np.array([mirror_pos_x, mirror_pos_y, 0.0]), 1)
    if k > 0:
        return gt_z[idx[0]]
    else:
        return 0

def faster_check_intersection(point_cloud_data, center, width, height, yaw_angle, sensor_pos):
    O = np.array(sensor_pos)
    C = np.array(center)
    half_width = width / 2.0
    half_height = height / 2.0

    yaw_rad = np.deg2rad(yaw_angle)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    normal_world = Rz @ np.array([0, 1, 0])

    ray_directions = point_cloud_data - O
    denominators = ray_directions @ normal_world
    numerator = np.dot(C - O, normal_world)
    
    valid_mask = np.abs(denominators) > 1e-6
    t = np.zeros_like(denominators)
    t[valid_mask] = numerator / denominators[valid_mask]
    
    t_mask = valid_mask & (t > 0.0) & (t <= 1.0)
    indices = np.where(t_mask)[0]
    if len(indices) == 0:
        return np.zeros(point_cloud_data.shape[0], dtype=bool)
    
    I_world = O + t[indices, np.newaxis] * ray_directions[indices]
    I_local = (I_world - C) @ Rz
    
    inside_mask = (np.abs(I_local[:, 0]) <= half_width) & (np.abs(I_local[:, 2]) <= half_height)
    is_reflected = np.zeros(point_cloud_data.shape[0], dtype=bool)
    is_reflected[indices[inside_mask]] = True
    return is_reflected

def decide_mirror_yaw_triangular(base_yaw, swing_range, rotation_speed, current_time):
    if swing_range == 0 or rotation_speed == 0:
        return base_yaw
    cycle_distance = 4 * swing_range
    total_distance = rotation_speed * current_time
    cycle_pos = total_distance % cycle_distance
    if cycle_pos < swing_range:
        offset = cycle_pos
    elif cycle_pos < 3 * swing_range:
        offset = 2 * swing_range - cycle_pos
    else:
        offset = cycle_pos - 4 * swing_range
    return base_yaw + offset

def reflection_sim(points, sensor_pos, sensor_ori, mirror_center, mirror_width, mirror_height, R_mat):
    normal_world = R_mat @ np.array([0, 1, 0])
    C, S = np.array(mirror_center), np.array(sensor_pos)
    S_v = S - 2 * np.dot(S - C, normal_world) * normal_world

    ray_directions = points - S_v
    denominator = ray_directions @ normal_world
    valid_denom = np.abs(denominator) > 1e-6
    t = np.zeros(len(points))
    t[valid_denom] = np.dot(C - S_v, normal_world) / denominator[valid_denom]
    
    mask = (t > 0) & (t < 1.0)
    indices = np.where(mask)[0]
    if len(indices) == 0: return np.empty((0, 3)), np.empty((0, 3))

    I_world = S_v + t[indices, np.newaxis] * ray_directions[indices]
    I_local = (I_world - C) @ R_mat
    
    in_boundary = (np.abs(I_local[:, 0]) <= mirror_width/2.0) & (np.abs(I_local[:, 2]) <= mirror_height/2.0)
    P_source = points[indices[in_boundary]]

    if len(P_source) > 0:
        dist_p = np.sum((P_source - C) * normal_world, axis=1)
        P_virtual = P_source - 2 * dist_p[:, np.newaxis] * normal_world
    else:
        P_virtual = np.empty((0, 3))
    return P_virtual, P_source

def create_pointcloud2(points, seq, stamp_ns, frame_id, typestore):
    blob = points.astype(np.float32).tobytes()
    data_array = np.frombuffer(blob, dtype=np.uint8)
    PointField = typestore.types['sensor_msgs/msg/PointField']
    fields = [
        PointField(name='x', offset=0, datatype=7, count=1),
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
    ]
    Header = typestore.types['std_msgs/msg/Header']
    Timestamp = typestore.types['builtin_interfaces/msg/Time']
    ros_time = Timestamp(sec=int(stamp_ns // 1e9), nanosec=int(stamp_ns % 1e9))
    header = Header(seq=seq, stamp=ros_time, frame_id=frame_id)
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    return PointCloud2(header=header, height=1, width=points.shape[0], fields=fields,
                       is_bigendian=False, point_step=12, row_step=12 * points.shape[0],
                       data=data_array, is_dense=True)

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

def generate_rosbag(param_x=0.0, param_y=0.0, param_yaw_center=0.0, param_swing_speed=0.0, param_swing_range=0.0, debug=False):
    with open('config.json', 'r') as f:
        config = json.load(f)

    bag_path = Path(config['main']['bag_path'])
    output_bag_path = Path(config['main']['output_bag'])
    lidar_topic_in = config['lidar']['lidar_topic']
    lidar_topic_out = config['lidar']['lidar_topic']
    imu_topic = config['imu']['imu_topic']
    topic_length = config['lidar']['topic_length']
    lidar_freq = config['lidar']['frequency']

    if output_bag_path.exists():
        output_bag_path.unlink()

    gt_pose_path = Path(config['main']['gt_path'])
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_pose_path)

    m_height = get_sensor_height(gt_x, gt_y, gt_z, param_x, param_y)
    mirror_center = [param_x, param_y, m_height]
    mirror_width, mirror_height = config['mirror']['width'], config['mirror']['height']
    max_frame_seq = len(gt_x)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    cnt = 0

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        with Writer(output_bag_path) as writer:
            lidar_conn_out = writer.add_connection(lidar_topic_out, 'sensor_msgs/msg/PointCloud2', typestore=typestore)
            imu_conn_out = writer.add_connection(imu_topic, 'sensor_msgs/msg/Imu', typestore=typestore)

            connections = [x for x in reader.connections if x.topic == lidar_topic_in or x.topic == imu_topic]

            for connection, timestamp, rawdata in reader.messages(connections=connections):

                msg = reader.deserialize(rawdata, connection.msgtype)
                msg_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

                if connection.topic == imu_topic:
                    writer.write(imu_conn_out, msg_ns, rawdata)
                    continue

                elif connection.topic == lidar_topic_in:

                    if cnt >= max_frame_seq:
                        continue

                    msg = reader.deserialize(rawdata, connection.msgtype)
                    msg_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

                    iteration = int(msg.data.shape[0]/topic_length)
                    bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                    lx, ly, lz = binary_to_xyz(bin_points)
                    local_points = np.vstack((lx, ly, lz)).T

                    sensor_pos = [gt_x[cnt], gt_y[cnt], gt_z[cnt]]
                    sensor_quat = [gt_qw[cnt], gt_qx[cnt], gt_qy[cnt], gt_qz[cnt]]
                    wx, wy, wz = coord_trans.local_to_world(local_points, sensor_pos, sensor_quat)
                    lidar_points_world = np.vstack((wx, wy, wz)).T

                    is_reflected = faster_check_intersection(lidar_points_world, mirror_center, mirror_width, mirror_height, param_yaw_center, sensor_pos)
                    P_visible = lidar_points_world[~is_reflected]

                    m_yaw = decide_mirror_yaw_triangular(param_yaw_center, param_swing_range, param_swing_speed, cnt / lidar_freq)
                    y_rad = np.deg2rad(m_yaw)
                    Rz = np.array([[np.cos(y_rad), -np.sin(y_rad), 0], [np.sin(y_rad), np.cos(y_rad), 0], [0, 0, 1]])

                    P_virtual, _ = reflection_sim(lidar_points_world, sensor_pos, sensor_quat, mirror_center, mirror_width, mirror_height, Rz)
                    #print(P_virtual.shape)
                    sim_world = np.vstack((P_visible, P_virtual))
                    sim_local = coord_trans.world_to_local(sim_world, sensor_pos, sensor_quat)

                    # simulation debug
                    if debug:
                        debug_visualize_2d(sensor_pos=sensor_pos[:2], lidar_points=sim_world, mirror_center=mirror_center[:2], 
                                mirror_yaw=m_yaw, p_virtual=P_virtual)
                    else:
                        pass

                    out_msg = create_pointcloud2(sim_local, cnt, msg_ns, msg.header.frame_id, typestore)
                    serialized_msg = typestore.serialize_ros1(out_msg, lidar_conn_out.msgtype)
                    writer.write(lidar_conn_out, msg_ns, serialized_msg)
                    cnt += 1

def run_slam(algorithm='kiss_icp'):
    cmd = ["roslaunch", "slamspoof", "mirror_sim_kiss.launch"] if algorithm == 'kiss_icp' else ["roslaunch", "slamspoof", "mirror_sim_flio.launch"]
    print("SLAMを開始します...")
    try:
        subprocess.run(cmd, check=True)
        print("SLAM終了")
    except Exception as e:
        print(f"SLAMエラー: {e}")

def calc_rotation_factor(qw, qz):
    stability_score = qw - np.abs(qz)
    return stability_score

def objective(trial):
    # 1. Trajectoryデータの読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)

    gt = config['main']['gt_path']
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt)
    matching_factor = calc_rotation_factor(gt_qw, gt_qz)

    # 2. 設置するインデックスを提案
    idx = trial.suggest_int('gt_index', 0, len(gt_x) - 1)

    # 3. 最近傍の点群を読み込み
    nearest_points = load_files.load_nearest_pcd(idx)
    anisotropy, angle = eigen_decomposition.eigen_value_decomposition_2d(nearest_points) # 局在度と長手方向を求める
    #print(f"angle:{angle}deg")

    # 最近傍点からの距離を設定(config)
    dist = float(config['simulation']['distance'])

    # 鏡のyaw角 : 長手方向から90度回転
    mirror_placement_yaw_temp = angle + 90
    mirror_placement_yaw = (mirror_placement_yaw_temp + 180) % 360 - 180 # 鏡の向き

    # 設置座標 (真上)
    mx = dist * np.cos(np.deg2rad(mirror_placement_yaw)) + gt_x[idx] # 鏡の位置(x)
    my = dist * np.sin(np.deg2rad(mirror_placement_yaw)) + gt_y[idx] # 鏡の位置(y)

    #m_yaw = angle # 長手方向を向くように設定
    m_yaw = trial.suggest_float('mirror_yaw', -180.0, 180.0) 
       
    # 4. その他の最適化パラメータ
    #speed = trial.suggest_float('mirror_swing_speed', 0.0, 20.0)
    speed = 7.0

    try:
        score = objective_function.simulator(mx, my, m_yaw, param_swing_speed=speed, param_swing_range=120)
        
        # 後のプロット用に計算値を保存
        trial.set_user_attr("mx", mx)
        trial.set_user_attr("my", my)
        trial.set_user_attr("calculated_yaw", m_yaw)
        
        return score
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

if __name__ == "__main__":
    start = time.time()
    # 1. rosbagからpcdファイル生成
    preprocessing.rosbag_writer()

    # 2. 最適化の実行
    with open('config.json', 'r') as f:
        config = json.load(f)

    n_trials = int(config['simulation']['n_trials'])
    n_random_trials = int(n_trials) * 0.30

    #study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(n_startup_trials=n_random_trials))
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # --- ベストパラメータの表示 ---
    print("\n" + "="*30)
    print("最適化が完了しました。")
    print(f"ベストスコア: {study.best_value}")
    print("ベストパラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30 + "\n")

    if bool(config['main']['save_study_result']):
        load_files.save_results(study)

    # --- 3. ベストパラメータの抽出と最終実行準備 ---
    best_trial = study.best_trial
    
    # user_attrs から座標を取得
    best_x = best_trial.user_attrs["mx"]
    best_y = best_trial.user_attrs["my"]
    
    best_yaw = best_trial.params["mirror_yaw"] 
    #best_yaw = best_trial.user_attrs["calculated_yaw"]
    best_speed = 7.0 # 固定値

    #objective_function.simulator(best_x, best_y, best_yaw, param_swing_speed=best_speed, param_swing_range=120, debug=True)
    #generate_rosbag(param_x=best_x, param_y=best_y, param_yaw_center=best_yaw, 
    #                param_swing_speed=best_speed, param_swing_range=120.0, debug=True)

    # 4. ベストパラメータで最終ROSバッグを生成
    print(f"ベスト位置({best_x:.2f}, {best_y:.2f})・方位({best_yaw:.2f}°)で最終生成中...")
    generate_rosbag(param_x=best_x, param_y=best_y, param_yaw_center=best_yaw, 
                    param_swing_speed=best_speed, param_swing_range=120.0)
    
    # 5. SLAMを実行して軌跡ファイルを生成
    run_slam(algorithm='kiss_icp')

    # --- 6. 結果の可視化 (プロット) ---
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(config['main']['gt_path']) # クオータニオンの並びはw, x, y, z
    
    # SLAMの出力パスをconfigから読み込み
    est_path = config['main']['est_path']
    est_x, est_y, est_z, est_qw, est_qx, est_qy, est_qz = load_files.load_benign_pose(est_path)

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    ax.plot(gt_x, gt_y, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Ground Truth')
    ax.plot(est_x, est_y, color='blue', linewidth=2.0, label='Spoofed Trajectory')

    # ベスト設置位置
    ax.scatter(best_x, best_y, color='red', marker='*', s=300, edgecolor='darkred', zorder=5, label='Best Mirror')

    # 鏡の向きを矢印で表示
    arrow_length = 5.0 
    rad = np.radians(best_yaw)
    ax.arrow(best_x, best_y, arrow_length * np.cos(rad), arrow_length * np.sin(rad), 
             width=0.5, head_width=2.0, head_length=2.5, fc='red', ec='red', alpha=0.8, zorder=6)

    ax.set_title(f"SLAM Optimization Result\nBest Score: {study.best_value:.4f}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./images/slam_result_{timestamp}.png"

    # savefigはshowの前に呼ぶ必要があります
    plt.savefig(filename, dpi=150)
    print(f"プロットを保存しました: {filename}")
    print(f"Processing time :{time.time()-start}sec")

    plt.show()

    # 位置ずれを定量的に評価
    RPE = error_estimate.evo_rpe_eval_results(config['main']['gt_path'], config['main']['est_path'])
    print(f"RPE is {RPE} m")

    # 7. クリーンアップ
    postprocess.cleanup_files()
