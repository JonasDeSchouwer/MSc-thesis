import os
import os.path as osp
import numpy as np
import sys
import argparse
from torch_geometric.graphgym.utils.io import json_to_dict_list

parser = argparse.ArgumentParser(description='Report epoch times')
parser.add_argument('dir', type=str, help='Directory containing the runs')


def report_epoch_times2(dir):
    """
    Reads the epoch times from the subdir of each split of each run in the given `dir`
    And writes the averages and standard deviations into each dir/agg/*split*/time.txt
    """

    agg_dir = osp.join(dir, "agg")
    splits = os.listdir(agg_dir)
    runs = [subdir for subdir in os.listdir(dir) if subdir != "agg" and osp.isdir(osp.join(dir, subdir))]

    for split in splits:

        epoch_time_means = []

        # get the mean for each split
        for run in runs:
            run_split_dir = osp.join(dir, run, split)
            fname_stats = osp.join(run_split_dir, "stats.json")
            stats_list = json_to_dict_list(fname_stats)

            epoch_time_mean = np.mean([x['time_epoch'] for x in stats_list])
            epoch_time_means.append(epoch_time_mean)

        overall_epoch_time_mean = np.mean(epoch_time_means)
        overall_epoch_time_std = np.std(epoch_time_means)

        # save the overall mean and std of the epoch times for this split in a new file 'time2.txt'
        with open(osp.join(agg_dir, split, "time2.txt"), "w") as f:
            f.write(f"{overall_epoch_time_mean} ± {overall_epoch_time_std}")


if __name__ == "__main__":
    args = parser.parse_args()
    report_epoch_times2(args.dir)