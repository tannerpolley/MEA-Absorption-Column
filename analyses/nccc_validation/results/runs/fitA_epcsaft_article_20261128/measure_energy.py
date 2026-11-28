"""Evaluate the retained fit-A collocation polynomials with Engine enthalpy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path('analyses/nccc_validation/scripts')))
from measure_epcsaft_energy import measure

if __name__ == '__main__':
    measure(Path(sys.argv[1]))
