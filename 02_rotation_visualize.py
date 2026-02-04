import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import savgol_filter, medfilt

# load external files
import load_files
import eigen_decomposition

def apply_sliding_window(scores, window_size=10):
    # 窓内の合計を計算
    kernel = np.ones(window_size)
    accumulated = np.convolve(scores, kernel, mode='same')
    return accumulated

def calc_rotation_factor(file):
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
    direction_diff = np.insert(np.diff(move_direction_rad), 0, 0.0)
    direction_diff = (direction_diff + np.pi) % (2 * np.pi) - np.pi

    direction_diff * distances

    direction_diff = np.insert(direction_diff, 0, 0.0)
    
    # 5. 重み付け：方向（の絶対値） × 移動距離
    #weighted_score = np.abs(move_direction_rad) * distances
    
    # 6. 配列の長さを合わせる（先頭に0を挿入）
    # weighted_score は np.diff を2回使うなら2つ、1回なら1つ挿入
    #weighted_consistent = np.insert(weighted_score, 0, 0.0)
    #weighted_consistent = np.insert(np.abs(move_direction_rad), 0, 0.0)
    #distances = np.insert(distances, 0, 0.0)
    
    # 正規化（オプション）：最大値を1にするなど
    #if np.max(weighted_consistent) > 0:
    #    weighted_consistent /= np.max(weighted_consistent)

    return gt_x, gt_y, direction_diff

def get_yaw_from_quaternion_np(x, y, z, w):
    """
    クオータニオン(x, y, z, w)からYaw角[rad]を計算する (NumPy版)
    """
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    
    # math.atan2(y, x) と同様に np.arctan2(y, x) を使用
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    
    return yaw

def get_yaw_list(file):
    # データの読み込み
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(file)
    
    # 全フレームの絶対Yaw角を一括計算 (NumPy配列)
    yaw_frames = get_yaw_from_quaternion_np(gt_qx, gt_qy, gt_qz, gt_qw)

    # 1. 隣り合うフレーム間の差分を計算 (この時点で長さは len - 1 になる)
    yaw_diff = np.diff(yaw_frames)

    # 2. 角度の折り返し補正 (±πを跨いだ瞬間に大きな値にならないようにする)
    # 例: 179度から-179度へ動いた場合、差分を+2度として扱う
    yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi

    # 3. 先頭に 0.0 を挿入して、元の gt_x と同じ長さに揃える
    yaw_diff_consistent = np.insert(yaw_diff, 0, 0.0)

    return yaw_diff_consistent

def smooth_score_savgol(score, window_size=15, polyorder=4):
    """
    Savitzky-Golayフィルタによる平滑化
    window_size: 奇数である必要あり
    polyorder: 適合させる多項式の次数（2〜3が一般的）
    """
    if len(score) < window_size:
        return score
    
    denoised = medfilt(score, kernel_size=15)

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

def anis(file):

    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(file)

    anis_array = []
    
    for cnt in range(len(gt_x)):
        nearest_points = load_files.load_nearest_pcd(cnt)
        anisotropy, angle = eigen_decomposition.eigen_value_decomposition_2d(nearest_points)
        anis_array.append(anisotropy)
    
    return gt_x, gt_y, anis_array

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # データ読み込み
    file = "/home/rokuto/lidar_mirror_sim/01_31_code/01_28_benign.txt"
    #yaw_frames = get_yaw_list(file)
    x, y, score1 = calc_rotation_factor(file)

    score1 = apply_sliding_window(score1, window_size=30)
    
    #x, y, score2 = anis(file)

    score = smooth_score_savgol(score1)

    plt.rcdefaults()

    # --- 第1図: 軌跡のヒートマップ ---
    fig1, ax1 = plt.subplots(figsize=(10, 6), layout='constrained')
    sc = ax1.scatter(x, y, c=score, cmap='viridis', s=15, alpha=0.8, edgecolors='none', label='Vulnerability Score')
    
    cbar = fig1.colorbar(sc, ax=ax1)
    cbar.set_label('Rotation Factor (Score)', fontsize=12)
    
    ax1.plot(x, y, color='gray', alpha=0.2, linewidth=1, zorder=1)
    ax1.set_title("Trajectory Heatmap (Spatial Distribution)", fontsize=14)
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- 第2図: スコアの分布 (ヒストグラム) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6), layout='constrained')
    
    # ヒストグラムの描画
    #n, bins, patches = ax2.hist(score, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 平均値などの統計情報を追加
    #mean_val = np.mean(score)
    #ax2.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
    #ax2.plot(score, color="blue")
    ax2.plot(score, color="red")
    
    ax2.set_title("Score Distribution (Frequency)", fontsize=14)
    ax2.set_xlabel("Rotation Factor (Score)")
    ax2.set_ylabel("Frequency (Count)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)

    # 両方の図を表示
    plt.show()
