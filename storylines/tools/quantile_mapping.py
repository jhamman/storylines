#!/usr/bin/env python
import numpy as np


def calc_quantiles_and_sorted_data(data):
    """calculate quantiles for each point in data and sort
    return arrays of the sorted data and corresponding quantiles"""
    outputdata = np.zeros((data.shape[0], 2))
    outputdata[:, 0] = np.sort(data)
    outputdata[:, 1] = np.arange(len(data))/float(len(data))

    return outputdata


def linterp(input_data, match, datacol=0, weightcol=1):
    """linearly interpolate between input_data[0,datacol] and input_data[1,datacol]
    interpolation is based on where match falls relative to the values in
    input_data[:,weightcol]
    """
    weight = (input_data[0, weightcol] - match) / (input_data[1, weightcol] -
                                                   input_data[0, weightcol])
    return weight*input_data[0, datacol] + (1-weight)*input_data[1, datacol]


def lookup_data(data, quantile_data, datacol=0, weightcol=1):
    """look up where data falls in quantile_data and linearly interpolate to
    get an improved value"""
    n = quantile_data.shape[0]
    i = n / 2
    step = float(i/2)
    done = False
    count = 0

    # perform a log(N) binary search
    while not done:
        if step < 1:
            done = True
        else:
            if quantile_data[i, weightcol] < data:
                i += np.ceil(step)
                step = np.ceil(step)/2.0
            elif quantile_data[i, weightcol] == data:
                step = 0
            else:
                i -= np.ceil(step)
                step = np.ceil(step)/2.0
        # just in case, enforce that i can be greater than the length of quantile_data
        i = max(0, min(n - 1, i))
        count = count+1
        if (count > n):
            raise ValueError("We are taking too long to converge, what is wrong?!?")

    # check a small region around the "found" point to avoid edge cases
    i = max(0, i - 2)
    done = False
    while not done:
        if (quantile_data[i, weightcol] > data):
            done = True
        else:
            i += 1

        if (i == n):
            done = True
            i = n - 1

    # if the data point is past the bounds of the data, just return the extreme value
    if (i == 0) or (quantile_data[i, weightcol] <= data):
        return quantile_data[0, datacol]
    # otherwise, linearly interpolate between the two nearest values
    else:
        return linterp(quantile_data[i-1:i+1, :], data, weightcol=weightcol,
                       datacol=datacol)


# perform a quantile mapping of data from the input_quantiles to the output_quantiles
# assumes inputdata is a 1D array and quantile data were calculated from 1D
# arrays by calc_quantiles_and_sorted_data
def map_data_with_quantiles(inputdata, input_quantile_data,
                            output_quantile_data):
    """perform quantile mapping of data from input_quantiles to output_quantiles"""
    outputdata = np.zeros(inputdata.shape)
    for i in range(len(inputdata)):
        # find the current quantile in input_quantile_data
        quantile = lookup_data(inputdata[i], input_quantile_data,
                               weightcol=0, datacol=1)
        # find the corresponding data in output_quantile_data
        qm_data = lookup_data(quantile, output_quantile_data,
                              weightcol=1, datacol=0)
        outputdata[i] = qm_data

    return outputdata


def map_data(input_data, data_to_match):
    # just develop the training data from the first half of the input_data
    input_quantile_data = calc_quantiles_and_sorted_data(input_data)
    output_quantile_data = calc_quantiles_and_sorted_data(data_to_match)

    output_data = map_data_with_quantiles(input_data, input_quantile_data,
                                          output_quantile_data)

    return output_data

if __name__ == '__main__':
    # simple example test case
    n = 1000

    input_data = np.random.randn(n * 2) * 10 + 2
    data_to_match = np.random.randn(n) * 100 - 50

    # just develop the training data from the first half of the input_data
    input_quantile_data = calc_quantiles_and_sorted_data(input_data[:n])
    output_quantile_data = calc_quantiles_and_sorted_data(data_to_match)

    output_data = map_data_with_quantiles(input_data, input_quantile_data,
                                          output_quantile_data)

    if n <= 10:
        print("Here are the random input values")
        print(input_data)
        print("Here are the random values to be match")
        print(data_to_match)
        print("Here are the quantile mapped input values")
        print(output_data)
    else:
        try:
            import matplotlib.pyplot as plt
            plt.hist(input_data,   bins=int(n/10))
            plt.savefig("input_data.png")
            plt.clf()
            plt.hist(data_to_match, bins=int(n/10))
            plt.savefig("data_to_match.png")
            plt.clf()
            plt.hist(output_data,  bins=int(n/10))
            plt.savefig("output_data.png")
        except ImportError:
            print("matplotlib doesn't seem to be installed, skipping output figures.")
            print("Here are some of the random input values")
            print(input_data[:10])
            print("Here are some of the random values to be match")
            print(data_to_match[:10])
            print("Here are some of the quantile mapped input values")
            print(output_data[:10])
