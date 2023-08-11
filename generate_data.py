"""
python generate_data.py --sample_size=30000 --datasets=ss00,ss01,ss02,regr01,regr02
"""
import sys
import argparse
import os
from datetime import datetime

from helpers import Simulator

parser = argparse.ArgumentParser(description='generate synthetic datasets')
parser.add_argument('--sample_size', help='sample size for the high frequency variable', type=int, default=30000)
parser.add_argument('--replica_id', help='replica_id; toggle to set seed', type=int, default=0)
parser.add_argument('--datasets', '--list', help='delimited list input for dgp settings', type=str, default='ss00,ss01,ss02,regr01,regr02')

def main():

    global args
    args = parser.parse_args()
    setattr(args,'data_folder', 'data_sim')
    setattr(args,'datasets', args.datasets.split(','))
    setattr(args,'replica_id',int(args.replica_id))

    try:
        os.mkdir(args.data_folder)
        print(f'folder {args.data_folder}/ created')
    except OSError as error:
        print(f'folder {args.data_folder}/ already exists')
        pass

    simulator = Simulator()
    for ds_id in args.datasets:
        print(f'>>>> generating synthetic dataset {ds_id}')
        _ = simulator.generate_dataset(ds_id=ds_id, n=args.sample_size, replica_id=args.replica_id)
        
if __name__ == "__main__":
    print('=============================================================')
    print(f'[{sys.argv[0]}] started execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'[{sys.argv[0]}] finished execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')
