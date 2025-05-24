import os
import numpy as np
import pandas as pd
from ..BVP.ABS_Column import abs_column
from openpyxl import Workbook, load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt


# Put outputs into dictionary and dataframe

def make_dfs_dict(output_dict, keys_dict, stages):
    sheetnames = list(keys_dict.keys())
    dfs_dict = {}
    for k1 in sheetnames:
        d = {}
        keys = keys_dict[k1]
        array = output_dict[k1]
        for k2, v in zip(keys, array.T):
            d[k2] = v[::-1]
        df = pd.DataFrame(d, index=stages[::-1])
        df.index.name = 'Position'
        dfs_dict[k1] = df
    return dfs_dict


def save_run_outputs(Y_scaled, z, parameters, save_run_results=True, plot_temperature=False):
    n = len(z)
    outputs_0, keys_dict = abs_column(z[0], Y_scaled.T[0], parameters, run_type='saving', column_names=True)
    sheetnames = list(keys_dict.keys())

    # Initialize output arrays
    output_dict = {k: np.zeros((n, len(outputs_0[k]))) for k in sheetnames}

    # Populate output arrays
    for i in range(n):
        outputs, _ = abs_column(z[i], Y_scaled.T[i], parameters, run_type='saving')
        for k in sheetnames:
            output_dict[k][i] = outputs[k]

    # Convert to DataFrame dict
    dfs_dict = make_dfs_dict(output_dict, keys_dict, z)

    if save_run_results:
        # Locate or create the Excel workbook
        base = os.path.dirname(__file__)
        results_dir = os.path.abspath(os.path.join(base, '..', 'data', 'Results'))
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, 'Profiles.xlsx')

        if os.path.exists(path):
            wb = load_workbook(path)
        else:
            wb = Workbook()
            default = wb.active
            wb.remove(default)

        # Remove sheets not in current run
        existing = set(wb.sheetnames)
        wanted = set(dfs_dict.keys())
        for name in existing - wanted:
            del wb[name]

        # Create/populate sheets
        for sheetname, df in dfs_dict.items():
            if sheetname in wb.sheetnames:
                del wb[sheetname]
            ws = wb.create_sheet(title=sheetname)
            for row in dataframe_to_rows(df, index=False, header=True):
                ws.append(row)
            # Freeze top row
            ws.freeze_panes = 'A2'

        # Reorder sheets
        wb._sheets = [wb[name] for name in dfs_dict.keys()]

        # Save workbook
        wb.save(path)

    return dfs_dict
