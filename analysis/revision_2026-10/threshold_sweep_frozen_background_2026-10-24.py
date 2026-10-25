#!/usr/bin/env python3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG = "2026-10-24"


def main():
    outp = HERE / f"THRESHOLD_SWEEP_FROZEN_BG_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    per, src = {}, {}
    for c, code in (("S1", "S1"), ("S3", "X1")):
        p = HERE / f"THRESHOLD_SWEEP_NET_{c}_{TAG}.json"
        d = json.loads(p.read_text()); src[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        per[code] = {}
        for R in ("A", "B"):
            per[code][R] = {}
            for k, r in d["readers"][R].items():
                assert r["capped_cells"] == 0 and r["cap_term_mm3"] == 0.0, (c, R, k)
                per[code][R][k] = {"net_at_primary_background_mm3": round(r["gross_at_primary_background_mm3"] - r["implant_mm3"]
                                                                          - r["pin_metal_mm3"], 1),
                                   "net_remeasured_background_mm3": r["net_mm3"], "qc_flags_remeasured": r["qc_flags"]}
    lv = list(per["S1"]["A"])
    pairs = {R: {k: {"X1_minus_S1_at_primary_background_mm3": round(per["X1"][R][k]["net_at_primary_background_mm3"]
                                                                   - per["S1"][R][k]["net_at_primary_background_mm3"], 1),
                     "X1_minus_S1_remeasured_background_mm3": round(per["X1"][R][k]["net_remeasured_background_mm3"]
                                                                   - per["S1"][R][k]["net_remeasured_background_mm3"], 1)}
                 for k in lv} for R in ("A", "B")}
    out = {"_what": "",
           "per_case": per, "pairs": pairs,
           "levels_S1_gt_X1_at_primary_background": {R: [k for k in lv if pairs[R][k]["X1_minus_S1_at_primary_background_mm3"] < 0]
                                                     for R in ("A", "B")},
           "inputs_sha256": src, "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(outp, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["pairs"], indent=0)[:3000]); print(out["levels_S1_gt_X1_at_primary_background"])


if __name__ == "__main__":
    main()

