from numpy import exp


def enhancement_factor(Tl, Cl, Cl_MEA_true, kl_CO2, Dl_CO2):

    k2b = 3.95e10 * exp(-6864 / Tl)  # Greer -
    k2c = 10 ** (10.49 - 2200 / Tl) / 1000  # Hakita - Reaction rate of CO2 in aqueous MEA-AMP solution: Experiment and modeling
    k2d = exp(10 - 904.6 / Tl) / 1000  # Freguia - Modeling of CO2 Capture by Aqueous Monoethanolamine
    k2e = 3.1732e9 * exp(-4936.6 / Tl) * Cl_MEA_true * 1e-6  # Putta, Svendsen, Knuutila 2017 Eqn. 42

    # Akula Model Development, Validation, and Optimization of an MEA-Based
    # Post-Combustion CO2 Capture Process under Part-Load and Variable
    # Capture Operations
    k_MEA = 2.003e10*exp(-4742/Tl)
    k_H2O = 4.147e6*exp(-3110/Tl)
    k2f = (k_MEA*Cl[1] + k_H2O*Cl[2])*5e-8 # kmol -> mol added fudge factor, still in development

    # print(k2e, k2f)

    # Choose the rate constant to use
    k_rxn = k2f

    # Can be seen from Eq 8: Kvamsdal - Effects of the Temperature Bulge in CO2 Absorption from Flue Gas by Aqueous Monoethanolamine
    # Also from Moore 2021
    # Akula Model Development, Validation, and Optimization of an MEA-Based Post-Combustion CO2 Capture Process under Part-Load and Variable Capture Operations
    Ha = (k_rxn * Cl_MEA_true * Dl_CO2) ** .5 / kl_CO2

    E = Ha

    kinetics = [k_rxn, Ha, E]

    return E, kinetics
