
import numpy as np
from scipy.spatial import cKDTree

from evo.core import metrics
from evo.core.units import Unit
from evo.tools import log
log.configure_logging(verbose=False, debug=False, silent=False)

import pprint
from evo.tools import plot
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False

plot.apply_settings(SETTINGS)

from evo.tools import file_interface
from evo.core import sync

def calc_trans_error(ground_truth, estimated_traj):
    tree = cKDTree(estimated_traj)
    distance, index = tree.query(ground_truth)

    return distance

def load_traj(est_file, ref_file, convert_timestamp_to_relative=False):
    max_diff = 0.01
    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    # Convert absolute timestamps to relative time (from first timestamp)
    if convert_timestamp_to_relative:
        traj_ref.timestamps = traj_ref.timestamps - traj_ref.timestamps[0]
        traj_est.timestamps = traj_est.timestamps - traj_est.timestamps[0]

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
    traj_est.align(traj_ref, correct_scale=False, correct_only_scale=False)
    return traj_est, traj_ref

def get_ape(traj_est, traj_ref):
    pose_relation = metrics.PoseRelation.translation_part
    #pose_relation = metrics.PoseRelation.rotation_part
    #pose_relation = metrics.PoseRelation.full_transformation
    #pose_relation = metrics.PoseRelation.rotation_angle_deg

    data = (traj_est, traj_ref)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    return ape_stat

def get_rpe(traj_est, traj_ref):
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_est, traj_ref)

    # normal mode
    delta = 5
    delta_unit = Unit.meters
    all_pairs = False

    rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    rpe_metric.process_data(data)
    rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.max)
    return rpe_stat

def evo_eval_result(estimated_traj, ground_truth, convert_timestamp_to_relative=False):
    traj_est, traj_ref = load_traj(estimated_traj, ground_truth, convert_timestamp_to_relative)
    ape = get_ape(traj_est, traj_ref)
    return ape

def evo_rpe_eval_results(estimated_traj, ground_truth, convert_timestamp_to_relative=False):
    traj_est, traj_ref = load_traj(estimated_traj, ground_truth, convert_timestamp_to_relative)
    rpe = get_rpe(traj_est, traj_ref)
    return rpe
    