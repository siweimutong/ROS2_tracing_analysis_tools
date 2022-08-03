#!/usr/bin/env python3

import datetime
import itertools
import os
import sys
import time

from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# import tracetools_anlysis_Galactic
from tracetools_analysis.loading import load_file
from tracetools_analysis.processor.ros2 import Ros2Handler
from tracetools_analysis.utils.ros2 import Ros2DataModelUtil

trace_name = None
data_util = None
callback_symbols = None
TimeRange = Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]
TimeRanges = List[TimeRange]


def get_timer_callback_ranges(timer_node_name: str)->TimeRanges:
    """ Get timer callback instance ranges """

    # Get timer object
    objs_and_owner = {
        obj: data_util.get_callback_owner_info(obj)
        for obj, _ in callback_symbols.items()
    }
    for obj, owner_info in objs_and_owner.items():
        print(obj)
        print(owner_info)

    timer_objs = [
        obj for obj, owner_info in objs_and_owner.items() if 'Timer' in owner_info
    ]
    # assert 1 == len(timer_objs), f'len={len(timer_objs)}'
    print(f'{timer_objs = }')
    timer_obj = timer_objs[0]

    # get callback_durations
    callback_durations = data_util.get_callback_durations(timer_obj)

    # Convert to simple list of tuples
    ranges = []
    for _, row in callback_durations.iterrows():
        begin = row['timestamp'].value
        duration = row['duration'].value
        ranges.append((begin, begin+duration, duration))

    return ranges


def get_sub_callback_ranges(
        sub_topic_name: str,
        node_name: Optional[str] = None,
) -> TimeRanges:
    """ Get Callback Object """
    objs_and_owners = {
        obj: data_util.get_callback_owner_info(obj)
        for obj, _ in callback_symbols.items()
    }

    sub_objs = [
        obj for obj, owner_info in objs_and_owners.items()
        if sub_topic_name in owner_info and (node_name in owner_info if node_name is not None else True)
    ]
    print(sub_objs)
    ranges = []
    # assert 1 == len(sub_objs), f'len={len(sub_objs)}'
    if len(sub_objs) == 0:
        return ranges

    # Get callback durations
    if(len(sub_objs) == 1):
        sub_obj = sub_objs[0]

        # Get callback durations
        callback_durations = data_util.get_callback_durations(sub_obj)

        # Convert to simple list of tuples

        for _, row in callback_durations.iterrows():
            begin = row['timestamp'].value
            duration = float(row['duration'].value)
            ranges.append((begin, begin + duration, duration))
        return ranges
    else:
        callback_durations = []
        for sub_obj in sub_objs:
            callback_durations.append(data_util.get_callback_durations(sub_obj))

        # Convert to simple list of tuples
        ranges = []
        for callback_duration in callback_durations:
            signal_range = []
            for _, row in callback_duration.iterrows():
                begin = row['timestamp'].value
                duration = row['duration'].value
                signal_range.append((begin, begin + duration, duration))
            ranges.append(signal_range)
        return ranges


def get_sub_callback_times(
        sub_topic_name: str,
        node_name: Optional[str] = None,
) -> List[pd.Timestamp]:
    """ Get subscription callback """

    ranges = get_sub_callback_ranges(sub_topic_name, node_name)
    return [r for r in ranges]


def draw_scatter_graph(
        timers_pub, timers_sub1, timers_sub2
):
    x = []
    y = []
    for i in range(5):
        x.append('pub')
        y.append(timers_pub[i][0])
    plt.scatter(y, x, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
                linewidths=None, edgecolors=None, plotnonfinite=False, data=None)

    for i in range(5):
        x.append('sub_1')
        y.append(timers_sub1[i][0])
    plt.scatter(y, x, s=None, c='g', marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
                linewidths=None, edgecolors=None, plotnonfinite=False, data=None)

    for i in range(5):
        x.append('sub_2')
        y.append(timers_sub2[i][0])
    plt.scatter(y, x, s=None, c='r', marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
                linewidths=None, edgecolors=None, plotnonfinite=False, data=None)
    plt.show()


def draw_arange_graph(timers_pub, timers_sub1):
    diff_s1 = []
    diff_s2 = []
    red_diff_s1 = []
    duration_s1 = []
    duration_s2 = []
    red_duration_s1 = []
    duration_s2_step2 = []


    diff_diff = []
    diff_diff_avg = []
    diff_timer = []
    diff_sum = 0
#    print(f'{timers_pub[5][0] = }')
#    print(f'{timers_sub1[4][0] = }')
#    print(f'{timers_sub2[4][0] = }')
#     for i in range(30):
#         diff_timer.append(timers_pub[i + 1][0] - timers_pub[i][0])
#         diff_s1.append(timers_sub1[i + 1][0] - timers_sub1[i][0])
#         diff_s2.append(timers_sub2[i + 1][0] - timers_pub[i][0])
#         diff_diff.append(diff_s1[i] - diff_s2[i])
#         diff_sum = diff_diff[i] + diff_sum
#         diff_diff_avg.append(diff_sum/(i + 1))

    # for i in range(len(timers_pub) - 1):
    #     diff_timer.append(timers_pub[i+1][0] - timers_pub[i][0])
    for i in range(len(timers_sub1) - 1):
        diff_s1.append(timers_sub1[i+1][0] - timers_sub1[i][0])
        if i < 25 or i > 78:
            red_diff_s1.append(1e8)
        else:
            red_diff_s1.append(timers_sub1[i+1][0] - timers_sub1[i][0])

    sum_s1: float = 0
    for i in range(len(timers_sub1) - 1):
        duration_s1.append(timers_sub1[i][2])
        sum_s1 += float(timers_sub1[i][2])
        if i < 26 or i > 78:
            red_duration_s1.append(500000)
        else:
            red_duration_s1.append(timers_sub1[i][2])
    sum_s1 = sum_s1 / (len(timers_sub1) - 1)


    avg_sum1 = []
    avg_sum2 = []

    for i in range(len(timers_sub1) - 1):
        avg_sum1.append(sum_s1)


    x = np.arange(0, len(diff_s1), 1)
    plt.subplot(1, 2, 1)
    line1, = plt.plot(x, diff_s1, label='sub1')
    # line2, = plt.plot(x, red_diff_s1, color='red', label='sub1')



    x = np.arange(0, len(duration_s1), 1)
    plt.subplot(1, 2, 2)
    plt.scatter(x, duration_s1, label='sub1')
    line5, = plt.plot(x, avg_sum1, label='red_sub1')


    #line4, = plt.plot(x, diff_diff, label='time drift')
    #line5, = plt.plot(x, diff_diff_avg,  label='avg time drift')
    #plt.legend(handles=[line1, line2], labels=['time drift', 'avg time drift'])
    print('---------------------------------')
    plt.savefig(f'{trace_name}.svg')
    plt.show()


def main(argv=sys.argv[1:])->int:
    """ Input tracing file path"""

    if len(argv) != 1:
        print('[Error')
        return 1

    global trace_name
    trace_name = argv[0].strip('/')
    print(f'Trace Directory:{trace_name}')

    """ Processing trace data """

    # Note 'f' is the fromat strings
    path = f'{trace_name}/ust'
    events = load_file(path)

    handler = Ros2Handler.process(events)

    global data_util
    data_util = Ros2DataModelUtil(handler.data)
    global callback_symbols
    callback_symbols = data_util.get_callback_symbols()

    timers_pub = get_timer_callback_ranges('sub')
    print(timers_pub)
    timers_sub_1 = get_sub_callback_times('/topic_1', 'sub')



#   timers_sub2 = get_sub_callback_times('/topic', 'sub')

#    print(timers_pub)
#    print(timers_sub1)
#    print(timers_sub2)

    # Plot
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif', size=14)
    #plt.rc('axes', titlesize=20)

    #draw_scatter_graph(timers_pub, timers_sub1, timers_sub2)
    draw_arange_graph(timers_pub, timers_sub_1)


# Press the green button in the gutter to run the script
if __name__ == '__main__':
    sys.exit(main())
