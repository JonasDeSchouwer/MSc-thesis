from graphgps.utils import report_epoch_times2
import argparse

parser = argparse.ArgumentParser(description='Report epoch times')
parser.add_argument('dir', type=str, help='Directory containing the runs')

if __name__ == "__main__":
    args = parser.parse_args()
    report_epoch_times2(args.dir)