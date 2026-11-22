import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
status = json.loads((root / "status.json").read_text())
assert (status["status"], status["controls"], status["substeps"], status["optimizer_steps"]) == (
    "complete", 640, 6400, 0
)
with (root / "summary.csv").open(newline="") as file:
    cases = list(csv.DictReader(file))
assert len(cases) == 8 and len({case["id"] for case in cases}) == 8
assert {(c["mode"], int(c["pose"]), float(c["start_mm_s"])) for c in cases} == {
    (mode, pose, speed)
    for mode in ("fixed_target", "frozen_336")
    for pose in (1, 2)
    for speed in (0.5, 1.0)
}
for case in cases:
    with (root / "traces" / (case["id"] + ".csv")).open(newline="") as file:
        trace = list(csv.DictReader(file))
    assert [int(sample["control"]) for sample in trace] == list(range(81))
    assert abs(float(trace[0]["forward_mm"]) + 2) < 0.001
    assert abs(float(trace[0]["signed_vx_mm_s"]) - float(case["start_mm_s"])) < 0.001
    entry = next(i for i, sample in enumerate(trace) if float(sample["radial_mm"]) <= 1)
    assert entry == int(case["first_entry_control"])
    assert abs(float(trace[entry]["speed_mm_s"]) - float(case["entry_mm_s"])) < 1e-6
    safe = all(
        float(r["radial_mm"]) <= 6 and float(r["speed_mm_s"]) <= 3
        and abs(float(r["depth_mm"])) <= 1 and abs(float(r["side_mm"])) <= 2
        and 0 <= float(r["tilt_deg"]) <= 10 for r in trace
    )
    held = all(float(r["radial_mm"]) <= 1 and float(r["speed_mm_s"]) <= 0.5 for r in trace[60:])
    assert (safe and held) == (case["pass"] == "1")
    assert (safe and held) == (case["mode"] == "fixed_target" and case["start_mm_s"] == "0.5")
print("Verified eight recorded cases: fixed-target 2/2 at 0.5 mm/s, 0/2 at 1 mm/s; frozen 336 0/4.")
