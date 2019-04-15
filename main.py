import csv
import math
import pathlib
import numpy

import matplotlib.pyplot as plt


def read_accelerometer_file(filepath):
    with open(filepath) as file:
        reader = csv.reader(file)
        return [list(map(float, (row[0], row[1], row[2], row[3]))) for index, row in enumerate(reader) if index > 3]


def chunks(data, chunk_size):
    accum = []
    for index, value in enumerate(data):
        accum.append(value)
        if index % chunk_size == 0:
            yield accum
            accum = []
    yield accum


def downsample_average(data, downsampled_size=None, chunk_size=None):
    if chunk_size and downsampled_size:
        raise Exception()

    if downsampled_size:
        data = list(data)
        actual_length = len(data)
        chunk_size = math.floor(actual_length / downsampled_size)
        if chunk_size < 2:
            raise Exception()

    return [sum(chunk) / len(chunk) for chunk in chunks(data, chunk_size)]


def analyze_windows(data, size):
    windowed = []
    buffer = []
    for value in data:
        buffer.append(value)
        if len(buffer) > size:
            buffer = buffer[1:]
        windowed.append(buffer)

    average = map(lambda a: sum(a) / len(a), windowed)
    median = map(numpy.median, windowed)
    std = map(numpy.std, windowed)
    min_v = list(map(min, windowed))
    max_v = list(map(max, windowed))
    difference = map(lambda a: a[1] - a[0], zip(min_v, max_v))

    return {
        f'average{size}': average,
        f'median{size}': median,
        f'std{size}': std,
        f'min{size}': min_v,
        f'max{size}': max_v,
        f'diff{size}': difference,
    }


def extract_features(data):
    transposed = numpy.transpose(data)

    time = transposed[0]
    x = transposed[1]
    y = transposed[2]
    z = transposed[3]

    def abs_sum(a):
        return sum(map(math.fabs, a))

    combined = list(map(abs_sum, zip(x, y, z)))

    return {
        'time': time,
        'combined': combined,
        **analyze_windows(combined, 100),
    }


datapath = pathlib.Path("data")

for filepath in datapath.iterdir():
    data = read_accelerometer_file(filepath)
    measurements = len(data)
    features = extract_features(data)

    display_width = 3000
    chunk_size = math.floor(measurements / display_width)

    time = downsample_average(features['time'], chunk_size=chunk_size)

    def plot_feature(feature_key, color):
        feature = downsample_average(features[feature_key], chunk_size=chunk_size)
        plt.plot(time, feature, color=color, label=feature_key)

    plot_feature('std100', color='black')
    # plot_feature('min100', color='red')
    # plot_feature('max100', color='blue')
    plot_feature('diff100', color='green')

    plt.legend(loc='best')
    plt.show()
