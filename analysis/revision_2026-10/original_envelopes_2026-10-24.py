#!/usr/bin/env python3
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
TAG = "2026-10-24"
CASES = ("S1", "S2", "S3", "S4")
PAPER = {"S1": "S1", "S3": "X1", "S2": "S2", "S4": "X2"}
ALL6 = [("S1", "S2"), ("S1", "S3"), ("S1", "S4"), ("S2", "S3"), ("S2", "S4"), ("S3", "S4")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def env(members):
    vals = [m["value_mm3"] for m in members]
    lo, hi = min(vals), max(vals)
    return {"min_mm3": lo, "max_mm3": hi, "n_members": len(vals),
            "argmin": [m["name"] for m in members if m["value_mm3"] == lo],
            "argmax": [m["name"] for m in members if m["value_mm3"] == hi]}


def relation(e1, e2):
    lo, hi = max(e1["min_mm3"], e2["min_mm3"]), min(e1["max_mm3"], e2["max_mm3"])
    return {"overlap": bool(lo <= hi), "signed_gap_mm3": round(lo - hi, 1)}


def main(spec_tag):
    outp = OUT / f"ORIGINAL_ENVELOPES_{TAG}{spec_tag}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    inputs, cases = {}, {}
    for c in CASES:
        pa, pb = ROOT / f"analysis/{c}/results.json", ROOT / f"analysis/{c}/readerB/results.json"
        ps, po = ROOT / f"analysis/{c}/specification_curve{spec_tag}.json", ROOT / f"analysis/{c}/outline_compare.json"
        for p in (pa, pb, ps, po):
            inputs[str(p.relative_to(ROOT))] = sha(p)
        A = json.loads(pa.read_text())["volume"]["PRIMARY_net_bone_gain_mm3"]
        B = json.loads(pb.read_text())["volume"]["PRIMARY_net_bone_gain_mm3"]
        spec = json.loads(ps.read_text())
        op = json.loads(po.read_text())["calibrated_windowed_mm3"]
        mA = ([{"name": "frozen_A", "value_mm3": A}] +
              [{"name": f"spec_stored_step{s['step']}_frac{s['frac']}_gap{s['gap']:.0f}", "value_mm3": s["net_mm3"]}
               for s in spec["specifications"] if s["roi_w"] == 10 and s["roi_d"] == 10] +
              [{"name": "operator_scaled_A", "value_mm3": op}])
        mB = [{"name": "frozen_B", "value_mm3": B}]
        cases[c] = {"paper_code": PAPER[c], "members": {"A": mA, "B": mB},
                    "envelope": {"A": env(mA), "B": env(mB), "case": env(mA + mB)}}
    pairs = {v: {f"{a}_{b}": relation(cases[a]["envelope"][v], cases[b]["envelope"][v]) for a, b in ALL6}
             for v in ("A", "B", "case")}
    out = {"_what": "",
           "specification_file_tag": spec_tag, "cases": cases, "pairs": pairs, "inputs_sha256": inputs,
           "script_sha256": sha(Path(__file__).resolve()), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(outp, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for c in CASES:
        e = cases[c]["envelope"]["case"]
        print(PAPER[c], e["min_mm3"], e["max_mm3"], e["argmin"], e["argmax"])
    print({k: v["signed_gap_mm3"] for k, v in pairs["case"].items()})


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[a.index("--spec-tag") + 1] if "--spec-tag" in a else "")

