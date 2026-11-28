"""3C-5C of ladder_epcsaft_henry.py: the [0, 1] bracket end fails G - D m > 0 and [0, 0.5] holds no root, so try upper ends 0.9, 0.8, 0.7, 0.6
(first converged 17-node start wins; the brentq root does not depend on the bracket once it is enclosed), then 33 -> 65 -> 129 on those inputs.
Run from the repository root."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ladder_epcsaft_henry import OUT, narrowed, solve  # noqa: E402  (module guard below keeps its pool from running)


def chain(case):
    for upper in (0.9, 0.8, 0.7, 0.6):
        inputs = narrowed(case, upper)
        seed, _ = solve(f"{case}_n17_fresh_b{upper}", *inputs, 17, None)
        if seed:
            break
    for nodes in (33, 65, 129):
        if seed is None:
            return
        seed, _ = solve(f"{case}_n{nodes}", *inputs, nodes, seed)


with ThreadPoolExecutor(3) as pool:
    list(pool.map(chain, ("3C", "4C", "5C")))
