# This file will read all tract files and determine which tracts are within the lesion mask

import os
import sys
import subprocess
import argparse

import pandas as pd

sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

tracts_df = pd.DataFrame()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine which tracts are within the lesion mask')
    parser.add_argument('tract_dir', type=str, help='Directory containing tract files')
    parser.add_argument('lesion', type=str, help='Lesion mask file')
    parser.add_argument('output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    tmp_dir = args.output_dir

    # Tract file are organised in VOI folders
    # Loop through all VOI folders
    for voi in list(filter(lambda x: 'custom' not in x and os.path.isdir(os.path.join(args.tract_dir, x)), os.listdir(args.tract_dir))):
        save_dir = os.path.join(tmp_dir, voi)
        os.makedirs(save_dir, exist_ok=True)
        # Loop through all tract files
        tck_files = list(filter(lambda x: x.endswith('.tck'), os.listdir(os.path.join(args.tract_dir, voi))))
        for tck_file in tck_files:
            tck_include_cmd = f'tckedit {os.path.join(args.tract_dir, voi, tck_file)} -include {args.lesion} {os.path.join(save_dir, tck_file)} -force -quiet'
            subprocess.run(tck_include_cmd, shell=True)
            track_count = subprocess.run(f'tckinfo {os.path.join(save_dir, tck_file)} -count', shell=True, capture_output=True).stdout.decode('utf-8')
            track_count = int(track_count.split('\n')[-2].split(' ')[-1])
            if track_count == 0:
                # Remove the tract file
                rm_cmd = f'rm {os.path.join(save_dir, tck_file)}'
                subprocess.run(rm_cmd, shell=True)
            else:
                # Add the track count to the DataFrame
                tracts_df.loc[voi, tck_file] = track_count

        # If the VOI folder is empty, remove it
        if len(os.listdir(save_dir)) == 0:
            rm_cmd = f'rm -r {save_dir}'
            subprocess.run(rm_cmd, shell=True)

    # Remove all rows with zeros
    tracts_df = tracts_df.loc[~(tracts_df == 0).all(axis=1)]

    # Print the DataFrame
    print(tracts_df)