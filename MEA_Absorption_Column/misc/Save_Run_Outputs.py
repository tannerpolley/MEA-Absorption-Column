import numpy as np
import pandas as pd
from MEA_Absorption_Column.BVP.ABS_Column import abs_column
import xlwings as xw

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


def save_run_outputs(Y, Fl_MEA, Fv_N2, Fv_O2, z, A, P, df_param, n):

    run_type = 'saving'

    outputs_0, keys_dict = abs_column(z[0], Y.T[0], Fl_MEA, Fv_N2, Fv_O2, P, A, df_param, run_type)

    sheetnames = list(keys_dict.keys())

    # Initializes each output array in the shape (n, m) where is the # of relevant properties in a group
    # and puts it into a list of output arrays
    output_dict = {}
    for k in sheetnames:
        output_dict[k] = np.zeros((n, len(outputs_0[k])))

    # Updates each output array and the (i, j) height step (i) for relevant group (j)
    for i in range(n):
        outputs, _ = abs_column(z[i], Y.T[i], Fl_MEA, Fv_N2, Fv_O2, P, A, df_param, run_type)

        for k in sheetnames:
            output_dict[k][i] = outputs[k]

    # Converts the Outputs dictionary into a dictionary of dataframes
    dfs_dict = make_dfs_dict(output_dict, keys_dict, z)

    # Updates each sheet in the Excel file with the new data from the df
    wb = xw.Book('data/Results/Profiles.xlsx', read_only=False)

    for sheetname, df in dfs_dict.items():
        try:
            wb.sheets[sheetname].clear()
        except:
            wb.sheets.add(sheetname)
        wb.sheets[sheetname].range("A1").value = df
    for sheet in wb.sheets:
        if sheet.name not in sheetnames:
            sheet.delete()
    wb.save(path=r'data/Results/Profiles.xlsx')

