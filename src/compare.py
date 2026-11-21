"""Paired comparison of two models scored on the SAME patients (e.g. 3D vs 2D).

  python -m src.compare --a results/tables/nnunet_3d_fullres_fold0_per_case.csv --name-a 3D \
                        --b results/tables/nnunet_2d_fold0_per_case.csv --name-b 2D
Reports per region: mean Dice of each, mean paired difference with a patient-level bootstrap 95% CI,
and the Wilcoxon signed-rank p-value. Statistical significance is not clinical significance.
"""
import argparse
from pathlib import Path
import pandas as pd
from .metrics import REGIONS
from .stats import bootstrap_ci, paired_wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A"); ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    A, B = pd.read_csv(a.a).set_index("case"), pd.read_csv(a.b).set_index("case")
    common = A.index.intersection(B.index)
    assert len(common) == len(A) == len(B), "the two models must be scored on exactly the same patients"
    A, B = A.loc[common], B.loc[common]
    rows = []
    for r in REGIONS:
        d = A[f"dice_{r}"] - B[f"dice_{r}"]
        lo, hi = bootstrap_ci(d.values)
        stat, p = paired_wilcoxon(A[f"dice_{r}"].values, B[f"dice_{r}"].values)
        rows.append({"region": r, "n": len(common), f"dice_{a.name_a}": A[f"dice_{r}"].mean(),
                     f"dice_{a.name_b}": B[f"dice_{r}"].mean(), "mean_diff": d.mean(),
                     "diff_ci95_low": lo, "diff_ci95_high": hi, "wilcoxon_p": p})
    out = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(a.out) / f"compare_{a.name_a}_vs_{a.name_b}.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
