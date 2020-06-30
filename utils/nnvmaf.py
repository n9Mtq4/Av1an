import struct
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


# Fields meanings: <source root>/av1/encoder/firstpass.h
fields = ['frame', 'weight', 'intra_error', 'frame_avg_wavelet_energy', 'coded_error', 'sr_coded_error', 'tr_coded_error',
         'pcnt_inter', 'pcnt_motion', 'pcnt_second_ref', 'pcnt_third_ref', 'pcnt_neutral', 'intra_skip_pct', 'inactive_zone_rows',
         'inactive_zone_cols', 'MVr', 'mvr_abs', 'MVc', 'mvc_abs', 'MVrv', 'MVcv', 'mv_in_out_count', 'new_mv_count', 'duration', 'count', 'raw_error_stdev']


X_COLS = [
    'weight', 'frame_avg_wavelet_energy',
    'MVr', 'MVc',
    'pcnt_inter', 'pcnt_motion', 'pcnt_second_ref', 
    'pcnt_third_ref', 'pcnt_neutral',
    'mv_in_out_count',
    'cq_value',
#     'psnr'
]


X_COLS = X_COLS + [f'nxt_{s}' for s in X_COLS]

Y_COLS = [
    'ms_ssim', 'psnr', 'ssim', 'vmaf'
#     'vmaf'
#     'ms_ssim', 'psnr', 'ssim'
]


X_MEANS = [
    1.200821e+00,
    1.694160e+07,
    1.010128e+00,
   -1.041517e+01,
    9.724755e-01,
    4.179238e-01,
    1.683899e-01,
    1.706898e-01,
    5.463165e-01,
   -1.086130e-02,
    4.500000e+01,
    1.200817e+00,
    1.694225e+07,
    1.012099e+00,
   -1.041595e+01,
    9.724387e-01,
    4.179052e-01,
    1.683853e-01,
    1.706898e-01,
    5.462896e-01,
   -1.086159e-02,
    4.500000e+01,
]


X_STDS = [
    5.847092e-02,
    7.887933e+06,
    4.513486e+01,
    1.146930e+02,
    1.139054e-01,
    2.679447e-01,
    1.136325e-01,
    1.678372e-01,
    1.904098e-01,
    1.689994e-01,
    1.000003e+01,
    5.847152e-02,
    7.888192e+06,
    4.513365e+01,
    1.146929e+02,
    1.140588e-01,
    2.679563e-01,
    1.136368e-01,
    1.678372e-01,
    1.904357e-01,
    1.689993e-01,
    1.000003e+01,
]


Y_MEANS = [
    0.992680,
   45.292758,
    0.993688,
   90.276615,
]


Y_STDS = [
    0.006177,
    3.848677,
    0.005999,
    6.904385,
]

X_MEAN_DS = pd.Series(dict(zip(X_COLS, X_MEANS)))
X_STD_DS = pd.Series(dict(zip(X_COLS, X_STDS)))
Y_MEAN_DS = pd.Series(dict(zip(Y_COLS, Y_MEANS)))
Y_STD_DS = pd.Series(dict(zip(Y_COLS, Y_STDS)))


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


def get_chunk_vmaf(fpf, cq):
    # construct pandas df from fpf
    df = pd.DataFrame(fpf)
    df['cq_value'] = [cq] * len(fpf)
    df = pd.concat([df, df.shift(1).add_prefix('nxt_')], axis=1)
    df = df[X_COLS].dropna()
    df = (df - X_MEAN_DS) / X_STD_DS
    pred = MODEL.predict(df)
    vmafs = [((l * Y_STD_DS) + Y_MEAN_DS)[3] for l in pred]
    avg_vmaf = sum(vmafs) / len(vmafs)
    return avg_vmaf


def binary_search_for_qual(fpf, target):
    pm_cq = 10
    mincq = 40 - pm_cq
    maxcq = 40 + pm_cq
    for i in range(6):
        test_cq = (mincq + maxcq) / 2
        probe_vmaf = get_chunk_vmaf(fpf, test_cq)
        if probe_vmaf > target:
            mincq = test_cq
        else:
            maxcq = test_cq
    return test_cq


def replace_cq(command: str, cq: int):
    """Return command with new cq value"""
    mt = '--cq-level='
    cmd = command[:command.find(mt) + 11] + str(cq) + command[command.find(mt) + 13:]
    return cmd


# (
# '-i .temp/split/00217.mkv   -strict -1 -pix_fmt yuv420p -f yuv4mpegpipe - | aomenc --passes=2 --pass=1 --threads=6 --row-mt=1 --cpu-used=5 --end-usage=q --cq-level=40 --lag-in-frames=25 --bit-depth=10 --tune=psnr --ivf --fpf=.temp/split/00217.log -o /dev/null - ',
# '-i .temp/split/00217.mkv   -strict -1 -pix_fmt yuv420p -f yuv4mpegpipe - | aomenc --passes=2 --pass=2 --threads=6 --row-mt=1 --cpu-used=5 --end-usage=q --cq-level=40 --lag-in-frames=25 --bit-depth=10 --tune=psnr --ivf --fpf=.temp/split/00217.log -o .temp/encode/00217.ivf - ',
# (PosixPath('.temp/split/00217.mkv'), PosixPath('.temp/encode/00217.ivf'))
# )
def nn_qual(commands, target_vmaf):
    # print(repr(commands))
    # commands[-1][0]
    fpf_file = commands[-1][0].with_suffix('.log')
    # print(fpf_file)
    fpf = read_first_pass(fpf_file)
    tg_cq = binary_search_for_qual(fpf, target_vmaf)
    print(f"F: {fpf_file.as_posix()} CQ: {tg_cq}")
    tg_cq = round(tg_cq)
    cm1 = replace_cq(commands[0], tg_cq)
    cm2 = replace_cq(commands[1], tg_cq)
    new_commands = (cm1, cm2) + commands[2:]
    return new_commands


def load_my_model():
    model_path = '/mnt/L4/workspace/Av1an/my_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model


MODEL = load_my_model()


if __name__ == '__main__':
    fpf = read_first_pass('/mnt/L4/media/av1ancrf/s1_60/split/00001.log')[:-1]
    vmaf = get_chunk_vmaf(fpf, 40)
    
    cq = binary_search_for_qual(fpf, 92)
    
    print()
