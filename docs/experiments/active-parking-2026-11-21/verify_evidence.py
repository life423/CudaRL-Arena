"""Independently check native parking traces. No simulation or training."""
import csv
import gzip
import hashlib
from collections import defaultdict
from pathlib import Path

root = Path(__file__).parent

def digest(path):
    raw = path.read_bytes()
    if path.suffix not in ('.gz', '.bin'):
        raw = raw.replace(b'\r\n', b'\n')
    return hashlib.sha256(raw).hexdigest()

def verify_manifest():
    for line in (root / "sha256.txt").read_text().splitlines():
        checksum, path = line.split("  ", 1)
        assert digest(root / path) == checksum, path

def verify_trials():
    data = defaultdict(list)
    with gzip.open(root / "endurance/trace.csv.gz", "rt") as trace:
        for row in csv.DictReader(trace):
            data[int(row["case"])].append(row)
    tier = defaultdict(list)
    with (root / "endurance/endurance.csv").open(newline="") as f:
        for case in csv.DictReader(f):
            rows = data.pop(int(case["case"]))
            assert [int(r["control"]) for r in rows] == list(range(1, 1281))
            safe = all(r["safe"] == "1" for r in rows)
            assert safe == bool(int(case["all_safe"]))
            streak = maximum = 0
            for r in rows:
                quiet = all((r["safe"] == "1", float(r["radial_mm"]) <= 1,
                             float(r["speed_mm_s"]) <= .5,
                             float(r["max_angular_rad_s"]) <= .035))
                assert quiet == bool(int(r["quiet"]))
                streak = streak + 1 if quiet else 0
                maximum = max(maximum, streak)
            duration = max(0, streak - 1) * .05
            assert abs(duration - float(case["final_quiet_s"])) < 1e-4
            assert abs(max(0, maximum - 1) * .05 - float(case["best_quiet_s"])) < 1e-4
            assert float(case["final_radial_mm"]) == float(rows[-1]["radial_mm"])
            passed = safe and duration >= 16
            assert passed == bool(int(case["pass"]))
            tier[round(float(case["initial_speed_mm_s"]), 2)].append(passed)
    assert not data
    assert [(s, sum(v), len(v)) for s,v in tier.items()] == [(0.5,6,6),(.55,6,6),(.6,5,6)]
    assert (root / "endurance/swim-reference.csv.gz").read_bytes() == (root / "endurance/swim-candidate.csv.gz").read_bytes()
    print("Verified hashes, 18 x 1280 hold controls, 6/6, 6/6, 5/6, identical Swim.")

verify_manifest()
verify_trials()
