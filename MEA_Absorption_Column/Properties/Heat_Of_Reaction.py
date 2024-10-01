
def heat_of_reaction(alpha):
    if alpha < .55:
        delH_rxn = (-2.8*alpha**5 + 1.65*alpha**4 - .17*alpha**3 - .045*alpha**2 + .0084*alpha + .085)*1e6

    elif alpha >= .55:
        delH_rxn = (-.1256*alpha**5 + .6377*alpha**4 - 1.2818*alpha**3 + 1.2757*alpha**2 + -.6319*alpha + .129077)*1e7

    else:
        delH_rxn = 2e4

    return -delH_rxn