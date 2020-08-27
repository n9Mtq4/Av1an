#!/bin/env python


import sys
from math import isnan

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from .arg_parse import Args
from .bar import process_pipe
from .chunk import Chunk
from .commandtypes import CommandPair
from .logger import log
from .utils import terminate
from .vmaf import call_vmaf, read_vmaf_json
from .encoders import ENCODERS


def target_vmaf_routine(args: Args, chunk: Chunk):
    """
    Applies target vmaf to this chunk. Determines what the cq value should be and sets the
    vmaf_target_cq for this chunk

    :param args: the Args
    :param chunk: the Chunk
    :return: None
    """
    chunk.vmaf_target_cq = target_vmaf(chunk, args)


def gen_probes_names(chunk: Chunk, q):
    """Make name of vmaf probe
    """
    return chunk.fake_input_path.with_name(f'v_{q}{chunk.name}').with_suffix('.ivf')


def probe_pipe(args: Args, chunk: Chunk, q):
    probe_name = gen_probes_names(chunk, q).with_suffix('.ivf').as_posix()
    pipe = ENCODERS[args.encoder].make_pipes(args, chunk, 1, 1, probe_name, q)

    return pipe


def get_target_q(scores, vmaf_target):
    x = [x[1] for x in sorted(scores)]
    y = [float(x[0]) for x in sorted(scores)]
    f = interpolate.interp1d(x, y, kind='quadratic')
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


def plot_probes(args, vmaf_cq, chunk: Chunk, frames):
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


def vmaf_probe(chunk: Chunk, q, args: Args):

    pipe = probe_pipe(args, chunk, q)
    process_pipe(pipe)

    file = call_vmaf(chunk, gen_probes_names(chunk, q), args.n_threads, args.vmaf_path, args.vmaf_res,
                     vmaf_rate=args.vmaf_rate)
    score = read_vmaf_json(file, 20)

    return score


def get_closest(q_list, q, positive=True):
    """Returns closest value from the list, ascending or descending
    """
    if positive:
        q_list = [x for x in q_list if x > q]
    else:
        q_list = [x for x in q_list if x < q]

    return min(q_list, key=lambda x: abs(x - q))


def weighted_search(q1, v1, q2, v2, target) -> int:
    """
    Returns cq value that should be closest to the target vmaf value with a linear interpolation.

    :param q1: cq of point1
    :param v1: vmaf of point1
    :param q2: cq of point2
    :param v2: vmaf of point2
    :param target: the vmaf to aim for
    :return: a q value that should be close to the target vmaf
    """
    # point-slope form (x=vmaf, y=q), solved for y=q
    m = (q2 - q1) / (v2 - v1)
    new_q = m * (target - v1) + q1
    return round(new_q)


def target_vmaf_search(chunk: Chunk, frames, args: Args):
    vmaf_cq = []
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
        return vmaf_cq, True

    elif next_q == args.max_q and score > args.vmaf_target:
        return vmaf_cq, True

    for _ in range(args.vmaf_steps - 2):
        new_cq = weighted_search(vmaf_cq[-2][1], vmaf_cq[-2][0], vmaf_cq[-1][1], vmaf_cq[-1][0], args.vmaf_target)
        # If the weighted search suggests we try a point we've already done, we can assume that the cq is very close,
        # and we can exit early.
        if new_cq in [x[1] for x in vmaf_cq]:
            # we need to put this point at the end of the list
            # get the vmaf from the last time we tried this cq
            new_vmaf = vmaf_cq[[x[1] for x in vmaf_cq].index(new_cq)][1]
            vmaf_cq.append((new_vmaf, new_cq))
            return vmaf_cq, True

        last_q = new_cq

        q_list.append(new_cq)
        score = vmaf_probe(chunk, new_cq, args)
        next_q = get_closest(q_list, last_q, positive=score >= args.vmaf_target)
        vmaf_cq.append((score, new_cq))

    return vmaf_cq, False


def target_vmaf(chunk: Chunk, args: Args):
    frames = chunk.frames
    vmaf_cq = []

    try:
        vmaf_cq, skip = target_vmaf_search(chunk, frames, args)
        if skip or len(vmaf_cq) == 2:
            if vmaf_cq[-1][1] == args.max_q:
                log(f"Chunk: {chunk.name}, Fr: {frames}\n"
                    f"Q: {sorted([x[1] for x in vmaf_cq])}, Early Skip High CQ\n"
                    f"Vmaf: {sorted([x[0] for x in vmaf_cq], reverse=True)}\n"
                    f"Target Q: {args.max_q} Vmaf: {vmaf_cq[-1][0]}\n\n")

            elif vmaf_cq[-1][1] == args.min_q:
                log(f"Chunk: {chunk.name}, Fr: {frames}\n"
                    f"Q: {sorted([x[1] for x in vmaf_cq])}, Early Skip Low CQ\n"
                    f"Vmaf: {sorted([x[0] for x in vmaf_cq], reverse=True)}\n"
                    f"Target Q: {args.min_q} Vmaf: {vmaf_cq[-1][0]}\n\n")

            else:
                log(f"Chunk: {chunk.name}, Fr: {frames}\n"
                    f"Q: {sorted([x[1] for x in vmaf_cq])}, Early Skip Hit VMAF\n"
                    f"Vmaf: {sorted([x[0] for x in vmaf_cq], reverse=True)}\n"
                    f"Target Q: {args.min_q} Vmaf: {vmaf_cq[-1][0]}\n\n")

            return vmaf_cq[-1][1]

        q, q_vmaf = get_target_q(vmaf_cq, args.vmaf_target)

        log(f'Chunk: {chunk.name}, Fr: {frames}\n'
            f'Q: {sorted([x[1] for x in vmaf_cq])}\n'
            f'Vmaf: {sorted([x[0] for x in vmaf_cq], reverse=True)}\n'
            f'Target Q: {q} Vmaf: {q_vmaf}\n\n')

        if args.vmaf_plots:
            plot_probes(args, vmaf_cq, chunk, frames)

        return q

    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        print(f'Error in vmaf_target {e} \nAt line {exc_tb.tb_lineno}')
        terminate()
