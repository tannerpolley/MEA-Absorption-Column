

def molar_flux(fl_CO2, fv_CO2, fl_H2O, fv_H2O, kv_CO2, kv_H2O, a_eA, Psi_H):

    Nv_CO2 = -kv_CO2 * a_eA * (fv_CO2 - fl_CO2) * Psi_H  # mol/(s*m)
    Nv_H2O = -kv_H2O * a_eA * (fv_H2O - fl_H2O) # mol/(s*m)

    # Vapor
    Nl_CO2 = -Nv_CO2   # mol/(s*m)
    Nl_H2O = -Nv_H2O # mol/(s*m)

    return Nv_CO2, Nv_H2O, Nl_CO2, Nl_H2O


def enthalpy_flux(Nl_CO2, Hl_CO2, Nl_H2O, Hl_H2O, Nv_CO2, Hv_CO2, Nv_H2O, Hv_H2O, UT, a_eA, Tv, Tl):

    Hl_CO2_trn = Nl_CO2 * Hl_CO2  # J/(s*m)
    Hl_H2O_trn = Nl_H2O * Hl_H2O  # J/(s*m)
    Hv_CO2_trn = Nv_CO2 * Hv_CO2  # J/(s*m)
    Hv_H2O_trn = Nv_H2O * Hv_H2O  # J/(s*m)

    Hv_trn = Nv_CO2 * Hv_CO2 + Nv_H2O * Hv_H2O  # J/(s*m)
    # Hl_trn = Nl_CO2 * Hl_CO2 + Nl_H2O * Hl_H2O  # J/(s*m)
    Hl_trn = -Hv_trn  # J/(s*m)

    qv = -UT * a_eA * (Tv - Tl)  # J/(s*m) =  J/(s*K*m) * K * m or W/(K*m) * K
    ql = UT * a_eA * (Tv - Tl)  # J/(s*m) =  J/(s*K*m) * K * m or W/(K*m) * K
    # ql = -qv  # J/(s*m) =  J/(s*K*m) * K * m or W/(K*m) * K

    Hv_flux = Hv_trn + qv
    Hl_flux = Hl_trn + ql

    return Hv_flux, Hl_flux, qv, ql, Hv_trn, Hl_trn, Hv_CO2_trn, Hv_H2O_trn, Hl_CO2_trn, Hl_H2O_trn
