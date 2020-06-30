import os
import glob
import struct
import itertools
import json
import pandas as pd
from typing import List, Dict


# Fields meanings: <source root>/av1/encoder/firstpass.h
fields = ['frame', 'weight', 'intra_error', 'frame_avg_wavelet_energy', 'coded_error', 'sr_coded_error', 'tr_coded_error',
         'pcnt_inter', 'pcnt_motion', 'pcnt_second_ref', 'pcnt_third_ref', 'pcnt_neutral', 'intra_skip_pct', 'inactive_zone_rows',
         'inactive_zone_cols', 'MVr', 'mvr_abs', 'MVc', 'mvc_abs', 'MVrv', 'MVcv', 'mv_in_out_count', 'new_mv_count', 'duration', 'count', 'raw_error_stdev']


def read_first_pass(log_path):
    """
    Reads libaom first pass log into a list of dictionaries.

    :param log_path: the path to the log file
    :return: A list of dictionaries. The keys are the fields from aom_keyframes.py
    """
    frame_stats = []
    with open(log_path, 'rb') as file:
        frame_buf = file.read(208)
        while len(frame_buf) > 0:
            stats = struct.unpack('d' * 26, frame_buf)
            p = dict(zip(fields, stats))
            frame_stats.append(p)
            frame_buf = file.read(208)
    return frame_stats


def read_fpfs(dir):
    
    fpfs = [f for f in glob.glob(f"{dir}/*.log")]
    
    raw_fpfs = [read_first_pass(f) for f in fpfs]
    noeos = [lst[:-1] for lst in raw_fpfs]
    
    flatten_fpf = itertools.chain(*noeos)
    
    df = pd.DataFrame(flatten_fpf)
    return df


def read_json(jfile):
    with open(jfile) as f:
      data = json.load(f)
    
    frame_lst = [frm['metrics'] for frm in data['frames']]
    
    df = pd.DataFrame(frame_lst)
    return df


def main():
    cq_value = 60
    dataset_name = f's1_{cq_value!s}'
    fp = read_fpfs(f"dataset/firstpass/{dataset_name}/split")
    metrics = read_json(f"dataset/vmaf/{dataset_name}.json")
    df = pd.concat([fp, metrics], axis=1)
    cq_col = [cq_value] * len(df)
    df['cq_value'] = cq_col
    df.to_csv(f'dataset/csv/{dataset_name}.csv')


if __name__ == '__main__':
    main()
