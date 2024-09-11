import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphgps.utils import report_epoch_times2

parser = argparse.ArgumentParser(description='Report epoch times')
parser.add_argument('dir', type=str, help='Directory containing the runs')

if __name__ == "__main__":
    args = parser.parse_args()
    report_epoch_times2(args.dir)