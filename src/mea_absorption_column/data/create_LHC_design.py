from scipy.stats import qmc
import numpy as np
import pandas as pd


def LHC_design(n):

    avg = np.array([314, 320, 29, 3.5, .28, .300, .013, .175])
    std = np.array([  3,   3,  3,   .5, .02, .020, .003, .01])

    l_bounds = avg - std
    u_bounds = avg + std

    sampler = qmc.LatinHypercube(d=len(avg))

    sample = sampler.random(n=n)
    sample = qmc.scale(sample, l_bounds, u_bounds)

    parameters = np.loadtxt('data/parameters_baseline.txt')
    columns = list(np.loadtxt('data/column_names.txt', dtype=str))

    data = np.zeros((len(sample), len(columns)))

    for i in range(len(sample)):
        data[i] = np.append(sample[i], parameters)
    index = np.arange(1, len(sample)+1)
    df = pd.DataFrame(data, columns=columns, index=index)
    df.index.name = 'Run'
    df.to_csv('data/LHC_design_w_SRP_cases.csv')

    return df
