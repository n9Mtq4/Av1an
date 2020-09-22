#!/bin/env python

from math import isnan, sqrt

from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_squared_error, average_precision_score
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.utils.extmath import safe_sparse_dot
import scipy.optimize as opt


def logistic(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b


class LogCurve:
    def __init__(self, x, y):
        (a_, b_, c_, d_), _ = opt.curve_fit(logistic, x, y, maxfev=10000000)
        self.a = a_
        self.b = b_
        self.c = c_
        self.d = d_
    
    def __call__(self, *args, **kwargs):
        return logistic(args[0], self.a, self.b, self.c, self.d)


class VideoDataChunk:
    def __init__(self, name, qs, vmafs):
        self.frames = 0
        self.name = name
        self.qs = qs
        self.vmafs = vmafs
        self.probes = 0
        self.q_to_vmaf = LogCurve(self.qs, self.vmafs)
        self.vmaf_to_q = LogCurve(self.vmafs, self.qs)
    
    def get_vmaf_at_q(self, q):
        self.probes += 1
        vmaf_pred = self.q_to_vmaf(q)
        return np.clip(vmaf_pred, 0, 99.99)


class VmafTestArgs:
    def __init__(self, vmaf_target):
        self.min_q = 25
        self.max_q = 50
        self.vmaf_target = vmaf_target
        self.vmaf_plots = False
        self.vmaf_steps = 4


ARG_LIST = [VmafTestArgs(88), VmafTestArgs(90), VmafTestArgs(92), VmafTestArgs(94), VmafTestArgs(96), VmafTestArgs(98)]


def log(s):
    print(s)


def target_vmaf_routine(args, chunk):
    """
    Applies target vmaf to this chunk. Determines what the cq value should be and sets the
    vmaf_target_cq for this chunk

    :param args: the Args
    :param chunk: the Chunk
    :return: None
    """
    chunk.vmaf_target_cq = target_vmaf(chunk, args)


def gen_probes_names(chunk, q):
    """Make name of vmaf probe
    """
    return chunk.fake_input_path.with_name(f'v_{q}{chunk.name}').with_suffix('.ivf')



def get_target_q(scores, vmaf_target):
    """
    Interpolating scores to get Q closest to target VMAF
    Interpolation type for 2 probes changes to linear
    """
    x = [x[1] for x in sorted(scores)]
    y = [float(x[0]) for x in sorted(scores)]

    if len(x) > 2:
        interpolation = 'quadratic'
    else:
        interpolation = 'linear'
    f = interpolate.interp1d(x, y, kind=interpolation)
    xnew = np.linspace(min(x), max(x), max(x) - min(x))
    tl = list(zip(xnew, f(xnew)))
    q = min(tl, key=lambda l: abs(l[1] - vmaf_target))

    return int(q[0]), round(q[1], 3)


def interpolate_data(vmaf_cq: list, vmaf_target):
    x = [x[1] for x in sorted(vmaf_cq)]
    y = [float(x[0]) for x in sorted(vmaf_cq)]

    # Interpolate data
    f = interpolate.interp1d(x, y, kind='quadratic')
    xnew = np.linspace(min(x), max(x), max(x) - min(x))

    # Getting value closest to target
    tl = list(zip(xnew, f(xnew)))
    vmaf_target_cq = min(tl, key=lambda l: abs(l[1] - vmaf_target))
    return vmaf_target_cq, tl, f, xnew


def plot_probes(args, vmaf_cq, chunk, frames):
    # Saving plot of vmaf calculation

    x = [x[1] for x in sorted(vmaf_cq)]
    y = [float(x[0]) for x in sorted(vmaf_cq)]

    cq, tl, f, xnew = interpolate_data(vmaf_cq, args.vmaf_target)
    matplotlib.use('agg')
    plt.ioff()
    plt.plot(xnew, f(xnew), color='tab:blue', alpha=1)
    plt.plot(x, y, 'p', color='tab:green', alpha=1)
    plt.plot(cq[0], cq[1], 'o', color='red', alpha=1)
    plt.grid(True)
    plt.xlim(args.min_q, args.max_q)
    vmafs = [int(x[1]) for x in tl if isinstance(x[1], float) and not isnan(x[1])]
    plt.ylim(min(vmafs), max(vmafs) + 1)
    plt.ylabel('VMAF')
    plt.title(f'Chunk: {chunk.name}, Frames: {frames}')
    plt.xticks(np.arange(args.min_q, args.max_q + 1, 1.0))
    temp = args.temp / chunk.name
    plt.savefig(f'{temp}.png', dpi=200, format='png')
    plt.close()


def vmaf_probe(chunk, q, args):
    """
    Make encoding probe to get VMAF that Q returns

    :param chunk: the Chunk
    :param q: Value to make probe
    :param args: the Args
    :return :
    """
    return chunk.get_vmaf_at_q(q)


def get_closest(q_list, q, positive=True):
    """
    Returns closest value from the list, ascending or descending

    :param q_list: list of q values that been already used
    :param q:
    :param positive: search direction, positive - only values bigger than q
    :return: q value from list
    """
    if positive:
        q_list = [x for x in q_list if x > q]
    else:
        q_list = [x for x in q_list if x < q]

    return min(q_list, key=lambda x: abs(x - q))


def weighted_search(num1, vmaf1, num2, vmaf2, target):
    """
    Returns weighted value closest to searched

    :param num1: Q of first probe
    :param vmaf1: VMAF of first probe
    :param num2: Q of second probe
    :param vmaf2: VMAF of first probe
    :param target: VMAF target
    :return: Q for new probe
    """

    dif1 = abs(target - vmaf2)
    dif2 = abs(target - vmaf1)

    tot = dif1 + dif2

    new_point = int(round(num1 * (dif1 / tot) + (num2 * (dif2 / tot))))
    return new_point


def target_vmaf(chunk, args):
    vmaf_cq = []
    frames = chunk.frames
    q_list = []
    score = 0

    # Make middle probe
    middle_point = (args.min_q + args.max_q) // 2
    q_list.append(middle_point)
    last_q = middle_point

    score = vmaf_probe(chunk, last_q, args)
    vmaf_cq.append((score, last_q))

    # Branch
    if score < args.vmaf_target:
        next_q = args.min_q
        q_list.append(args.min_q)
    else:
        next_q = args.max_q
        q_list.append(args.max_q)

    # Edge case check
    score = vmaf_probe(chunk, next_q, args)
    vmaf_cq.append((score, next_q))

    if next_q == args.min_q and score < args.vmaf_target:
        log(f"Chunk: {chunk.name}, Fr: {frames}\n"
            f"Q: {sorted([x[1] for x in vmaf_cq])}, Early Skip Low CQ\n"
            f"Vmaf: {sorted([x[0] for x in vmaf_cq], reverse=True)}\n"
            f"Target Q: {vmaf_cq[-1][1]} Vmaf: {vmaf_cq[-1][0]}\n\n")
        return next_q

    elif next_q == args.max_q and score > args.vmaf_target:
        log(f"Chunk: {chunk.name}, Fr: {frames}\n"
            f"Q: {sorted([x[1] for x in vmaf_cq])}, Early Skip High CQ\n"
            f"Vmaf: {sorted([x[0] for x in vmaf_cq], reverse=True)}\n"
            f"Target Q: {vmaf_cq[-1][1]} Vmaf: {vmaf_cq[-1][0]}\n\n")
        return next_q

    # VMAF search
    for _ in range(args.vmaf_steps - 2):
        new_point = weighted_search(vmaf_cq[-2][1], vmaf_cq[-2][0], vmaf_cq[-1][1], vmaf_cq[-1][0], args.vmaf_target)
        if new_point in [x[1] for x in vmaf_cq]:
            break
        last_q = new_point

        q_list.append(new_point)
        score = vmaf_probe(chunk, new_point, args)
        next_q = get_closest(q_list, last_q, positive=score >= args.vmaf_target)
        vmaf_cq.append((score, new_point))

    q, q_vmaf = get_target_q(vmaf_cq, args.vmaf_target)

    log(f'Chunk: {chunk.name}, Fr: {frames}\n'
        f'Q: {sorted([x[1] for x in vmaf_cq])}\n'
        f'Vmaf: {sorted([x[0] for x in vmaf_cq], reverse=True)}\n'
        f'Target Q: {q} Vmaf: {q_vmaf}\n\n')

    # Plot Probes
    if args.vmaf_plots and len(vmaf_cq) > 3:
        plot_probes(args, vmaf_cq, chunk, frames)

    return q



def get_wget_source():
    sources = ["aspen_1080p_60f.y4m", "blue_sky_360p_60f.y4m", "dark70p_60f.y4m", "DOTA2_60f_420.y4m",
               "ducks_take_off_1080p50_60f.y4m", "gipsrestat720p_60f.y4m", "kirland360p_60f.y4m",
               "KristenAndSara_1280x720_60f.y4m", "life_1080p30_60f.y4m", "MINECRAFT_60f_420.y4m",
               "Netflix_Aerial_1920x1080_60fps_8bit_420_60f.y4m", "Netflix_Boat_1920x1080_60fps_8bit_420_60f.y4m",
               "Netflix_Crosswalk_1920x1080_60fps_8bit_420_60f.y4m",
               "Netflix_DrivingPOV_1280x720_60fps_8bit_420_60f.y4m",
               "Netflix_FoodMarket_1920x1080_60fps_8bit_420_60f.y4m",
               "Netflix_PierSeaside_1920x1080_60fps_8bit_420_60f.y4m",
               "Netflix_RollerCoaster_1280x720_60fps_8bit_420_60f.y4m",
               "Netflix_SquareAndTimelapse_1920x1080_60fps_8bit_420_60f.y4m",
               "Netflix_TunnelFlag_1920x1080_60fps_8bit_420_60f.y4m", "niklas360p_60f.y4m", "red_kayak_360p_60f.y4m",
               "rush_hour_1080p25_60f.y4m", "shields_640x360_60f.y4m", "speed_bag_640x360_60f.y4m",
               "STARCRAFT_60f_420.y4m", "thaloundeskmtg360p_60f.y4m", "touchdown_pass_1080p_60f.y4m",
               "vidyo1_720p_60fps_60f.y4m", "vidyo4_720p_60fps_60f.y4m", "wikipedia_420.y4m"]
    url_base = "https://beta.arewecompressedyet.com/runs/ref_cpu6_master@2020-07-24T12:33:14.837Z/objective-1-fast/"
    for src in sources:
        print(f"wget {url_base}{src}-daala.out")


if __name__ == '__main__':
    total_probes = 0
    cq_preds = []
    cq_reals = []
    
    vmaf_dir = Path('./vmaf_test')
    vmaf_dir_files = list(vmaf_dir.iterdir())
    for fname in vmaf_dir_files:
        f = open(fname, 'r')
        lines = f.readlines()
        vmaf_data = {}
        for line in lines:
            lt = line.split(" ")
            q = lt[0]
            vmaf = lt[15]
            vmaf_data[q] = vmaf
        qs = list(map(float, vmaf_data.keys()))
        vmafs = list(map(float, vmaf_data.values()))
        video_data_chunk = VideoDataChunk(fname, qs, vmafs)
        for arg in ARG_LIST:
            print(f'{arg.vmaf_target=}')
            cq_pred = target_vmaf(video_data_chunk, arg)
            cq_real = video_data_chunk.vmaf_to_q(arg.vmaf_target)
            cq_preds.append(cq_real)
            cq_reals.append(cq_pred)
            total_probes += video_data_chunk.probes
    
    error = list(map(lambda ys: ys[0] - ys[1], zip(cq_reals, cq_preds)))
    wrong_cqs = len(list(filter(lambda e: e >= 1, error)))
    print(f"Performed {len(ARG_LIST) * len(vmaf_dir_files)} target vmaf searches")
    print(f"Total Probes: {total_probes}")
    print(f"RMSE: {sqrt(mean_squared_error(cq_reals, cq_preds))}")
    print(f"Averge diff: {sum(map(abs, error)) / len(error)}")
    print(f"Number of cqs wrong: {wrong_cqs}")
