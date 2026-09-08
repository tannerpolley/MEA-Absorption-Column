from mea_absorption_column.Thermodynamics.thermo_models import guarded_compute_fugacity


def fugacity(
    x,
    y,
    x_true,
    Cl_true,
    Tl,
    Tv,
    alpha,
    H_CO2_mix,
    P,
    P_sat_H2O,
    thermo_model='ideal_henry',
    diagnostics=None,
):

    kwargs = {
        "model": thermo_model,
        "y": y,
        "x_true": x_true,
        "Cl_true": Cl_true,
        "Tl": Tl,
        "Tv": Tv,
        "H_CO2_mix": H_CO2_mix,
        "P": P,
        "P_sat_H2O": P_sat_H2O,
    }
    kwargs["diagnostics"] = diagnostics
    Pl_CO2, Pv_CO2, Pl_H2O, Pv_H2O = guarded_compute_fugacity(
        **kwargs,
    )

    fl_CO2 = Pl_CO2
    fv_CO2 = Pv_CO2
    fl_H2O = Pl_H2O
    fv_H2O = Pv_H2O
    DF_CO2 = (fv_CO2 - fl_CO2)
    DF_H2O = (fv_H2O - fl_H2O)

    return fl_CO2, fv_CO2, fl_H2O, fv_H2O, [DF_CO2, H_CO2_mix], [DF_H2O, P_sat_H2O]
