"""Audit recorded holding data. No physics, optimizer, or checkpoint writes.

Python/NumPy are offline analysis tools here, never the training/runtime path.
Run from the worktree root: python tools/audit_hold_saved.py
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/ppo/rest-hold-duration-pair-2026-11-21-v1"
ROWS, OBS, ACT = 10240, 41, 4
ROW = np.arange(ROWS)
TASK = ((ROW % 32) // 2) % 2
FIT = ROW % 32 % 8 < 4
CELL = (TASK * 2 + ROW % 2) * 320 + ROW // 32


class Reader:
    def __init__(self, data): self.data, self.at = data, 0
    def value(self, fmt):
        result = struct.unpack_from("<" + fmt, self.data, self.at)[0]
        self.at += struct.calcsize("<" + fmt)
        return result
    def array(self, dtype="<f4"):
        n = self.value("Q")
        dt = np.dtype(dtype)
        result = np.frombuffer(self.data, dt, n, self.at)
        self.at += n * dt.itemsize
        assert self.at <= len(self.data)
        return result.copy()
    def finish(self): assert self.at == len(self.data), (self.at, len(self.data))


def read_envelope(path):
    raw = path.read_bytes()
    magic, count, expected = struct.unpack_from("<QQQ", raw)
    assert magic == 0x31504B434C524443 and count == len(raw) - 24
    data = raw[24:]
    checksum = 14695981039346656037
    for b in data: checksum = ((checksum ^ b) * 1099511628211) & 0xffffffffffffffff
    assert checksum == expected, path
    return data


def read_batch(path):
    outer = Reader(read_envelope(path))
    assert outer.value("I") == 1
    inner = Reader(outer.array("u1").tobytes())
    assert [inner.value("I") for _ in range(4)] == [1, ROWS, OBS, ACT]
    names = ["obs", "prevobs", "prevaction", "history", "action", "logp", "value", "reward", "done"]
    d = {name: inner.array() for name in names}
    inner.finish()
    for name, width in [("obs", OBS), ("prevobs", OBS), ("prevaction", ACT), ("action", ACT)]:
        d[name] = d[name].reshape(ROWS, width)
    d["costs"] = outer.array().reshape(ROWS, 5)
    for _ in range(3):
        width = outer.value("I")
        outer.array(np.dtype((np.void, width)))
    outer.finish()
    assert all(np.isfinite(x).all() for x in d.values())
    return d


def read_actor(path):
    r = Reader(read_envelope(path))
    assert [r.value("I"), r.value("I")] == [0x50505241, 1]
    protocol = r.value("Q")
    arm, update, steps = r.value("i"), r.value("i"), r.value("i")
    params = r.array()
    r.array("u1")
    r.finish()
    assert len(params) == 2532
    return params, {"protocol": protocol, "arm": arm, "update": update, "steps": steps}


def read_fixed(path):
    r = Reader(read_envelope(path))
    assert r.value("I") == 0x50464252
    identity = r.value("Q")
    means, counts = r.array("<f8"), r.array("<i4")
    scales, ns = [], []
    for _ in range(2): scales.append(r.value("d")); ns.append(r.value("i"))
    r.finish()
    assert len(means) == 1280 and np.all(counts == 8) and ns == [5120, 5120]
    return means, np.array(scales), identity


def forward(params, x):
    at = 0
    for index, (ni, no) in enumerate([(41, 32), (32, 32), (32, 4)]):
        w = params[at:at + ni * no].reshape(no, ni); at += ni * no
        b = params[at:at + no]; at += no
        x = x @ w.T + b
        if index < 2: x = np.tanh(x)
    assert at == len(params)
    return x


def returns(d):
    rewards, done = d["reward"].reshape(320, 32), d["done"].reshape(320, 32)
    result = np.empty_like(rewards, dtype=np.float64)
    last = np.zeros(32)
    for t in range(319, -1, -1):
        last = rewards[t] + float(np.float32(.99)) * last * (1 - done[t])
        result[t] = last
    return result.ravel()


def stats(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "SD": float(x.std()), "RMS": float(np.sqrt(np.mean(x * x))),
            "min": float(x.min()), "max": float(x.max())}


def covariance(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xc, yc = x - x.mean(axis=0), y - y.mean()
    cov = np.mean(xc * yc[:, None], axis=0)
    scale = np.sqrt(np.mean(xc * xc, axis=0) * np.mean(yc * yc))
    return {"covariance": cov.tolist(), "correlation": np.divide(cov, scale, out=np.zeros_like(cov), where=scale > 0).tolist()}


def sequences(mask, task, history):
    result, active, longest = [], 0, 0
    shaped, hist = mask.reshape(320, 32), history.reshape(320, 32)
    for world in range(32):
        if int(((world // 2) % 2)) != task: continue
        length = 0
        for t in range(320):
            if not hist[t, world] and length:
                result.append(length); length = 0
            if shaped[t, world]:
                active += 1; length += 1; longest = max(longest, length)
            elif length:
                result.append(length); length = 0
        if length: result.append(length)
    return {"rows": active, "fraction": active / 5120, "longest_controls": longest,
            "longest_seconds": longest * .05, "runs": len(result),
            "runs_at_least_4controls": sum(x >= 4 for x in result),
            "runs_at_least_20controls": sum(x >= 20 for x in result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "build/audit-hold-saved-2026-11-22")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [RUN / "pooled-fixed.bin", RUN / "approved-protocol.md", Path(__file__)]
    batches = {}
    for arm in ["A20", "B20_60"]:
        for stage, updates in [("calibration", [336]), ("training", [336, 337, 338])]:
            home = RUN / (f"calibration-{'A' if arm == 'A20' else 'B'}" if stage == "calibration" else f"phase-B-started/{arm}")
            for u in updates:
                bp, rp = home / f"batch-{u}.bin", home / f"rows-{u}.csv"
                inputs.extend([bp, rp]); batches[(arm, stage, u)] = (read_batch(bp), rp)
                if stage == "training": inputs.append(home / f"actor-only-update-{u}.bin")
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    means, scales, identity = read_fixed(RUN / "pooled-fixed.bin")
    cal = {arm: returns(batches[(arm, "calibration", 336)][0]) for arm in ["A20", "B20_60"]}
    pooled = np.zeros(1280)
    count = np.zeros(1280, int)
    for raw in cal.values():
        np.add.at(pooled, CELL[FIT], raw[FIT]); np.add.at(count, CELL[FIT], 1)
    pooled /= count
    assert np.max(abs(pooled - means)) < 2e-7
    specific = {}
    for arm, raw in cal.items():
        m = np.zeros(1280); n = np.zeros(1280, int)
        np.add.at(m, CELL[FIT], raw[FIT]); np.add.at(n, CELL[FIT], 1)
        assert np.all(n == 4)
        specific[arm] = m / n
    records = []
    for (arm, stage, u), (d, csvpath) in batches.items():
        rows = list(csv.DictReader(csvpath.open(newline="")))
        raw = np.array([float(row["return"]) for row in rows])
        independent = returns(d)
        err = float(np.max(abs(raw - independent))); assert err < 2e-5
        assert np.max(abs(d["reward"] - np.array([float(row["reward"]) for row in rows]))) < 1e-10
        assert np.array_equal(d["done"], [float(row["done"]) for row in rows])
        assert np.array_equal(d["history"], [float(row["history"]) for row in rows])
        speed = np.array([float(row["speed_m_s"]) for row in rows])
        decoded = np.tanh(d["action"].astype(np.float64))
        residual = raw - means[CELL]
        corrected = raw - specific[arm][CELL]
        record = {"arm": arm, "stage": stage, "collection_update": u, "return_reconstruction_max_error": err, "tasks": {}}
        innovations, conditional_probabilities = None, {}
        if stage == "training":
            home = RUN / f"phase-B-started/{arm}"
            params, meta = read_actor(home / f"actor-only-update-{u}.bin")
            assert meta["protocol"] == identity and meta["update"] == u
            mu = forward(params, d["obs"]); prev = forward(params, d["prevobs"])
            conditional = mu + np.float32(.8) * d["history"][:, None] * (d["prevaction"] - prev)
            variance = np.exp(np.float32(-2)) * (1 - np.float32(.8) ** 2 * d["history"])
            innovations = d["action"] - conditional
            logp = -.5 * np.sum(innovations ** 2 / variance[:, None] + np.log(2 * np.pi * variance[:, None]), axis=1)
            error = float(np.max(abs(logp - d["logp"]))); assert error < 1e-4
            record["independent_AR_logp_max_error"] = error
            for threshold in [.03, .1]:
                bound = np.arctanh(threshold)
                sd = np.sqrt(variance)[:, None]
                upper = (bound - conditional) / sd / np.sqrt(2)
                lower = (-bound - conditional) / sd / np.sqrt(2)
                cdfdiff = np.fromiter((.5 * (math.erf(a) - math.erf(b)) for a, b in zip(upper.ravel(), lower.ravel())), float).reshape(ROWS, 4)
                conditional_probabilities[str(threshold)] = np.prod(cdfdiff, axis=1)
        for task, label in [(0, "swim"), (1, "hold")]:
            mask = TASK == task
            held = mask & ~FIT
            r, c = residual[held], corrected[held]
            out = {"raw_return": stats(raw[mask]), "pooled_scaled_signal": stats(residual[mask] / scales[task]),
                   "pooled_positive_signal_fraction": float(np.mean(residual[mask] > 0)),
                   "heldout_pooled_residual": stats(r), "heldout_arm_specific_residual": stats(c),
                   "heldout_sign_flip_fraction_if_arm_specific": float(np.mean((r > 0) != (c > 0))),
                   "heldout_pooled_mean_offset_energy_fraction": float(r.mean() ** 2 / np.mean(r * r)),
                   "near_rest_speed_fraction": float(np.mean(speed[mask] <= .0005)),
                   "sampled_command_RMS": float(np.sqrt(np.mean(decoded[mask] ** 2))),
                   "sampled_command_mean_by_axis": decoded[mask].mean(axis=0).tolist(),
                   "near_zero_all4_command_runs": {str(t): sequences(np.max(abs(decoded), axis=1) <= t, task, d["history"]) for t in [.03, .05, .1, .2]},
                   "sampled_command_vs_raw_return": covariance(decoded[mask], raw[mask]),
                   "command_magnitude_vs_return": covariance(np.sqrt(np.mean(decoded[mask] ** 2, axis=1))[:, None], raw[mask]),
                   "command_magnitude_vs_arm_specific_residual": covariance(np.sqrt(np.mean(decoded[mask] ** 2, axis=1))[:, None], corrected[mask])}
            if innovations is not None:
                out["conditional_innovation_vs_arm_specific_residual"] = covariance(innovations[mask], corrected[mask])
                out["conditional_all4_near_zero_probability_on_visited_states"] = {
                    key: {"mean": float(prob[mask].mean()), "sum_expected_rows": float(prob[mask].sum()),
                          "initial_rest_mean": float(prob[mask & (d["history"] == 0)].mean())}
                    for key, prob in conditional_probabilities.items()}
            record["tasks"][label] = out
        records.append(record)
    after = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    assert after == manifest
    result = {"scope": "Saved-data arithmetic and network inference only; zero physical controls, optimizer steps, or checkpoint modifications",
              "run": str(RUN.relative_to(ROOT)), "pooled_scale_swim_hold": scales.tolist(),
              "notes": ["Near-zero command thresholds are descriptive, not physical acceptance gates.",
                        "Sampled target tanh(latent) is before the physical filter; no trajectory is inferred from commands alone.",
                        "Conditional all-axis interval probability is exact for the saved Gaussian at visited states, not a future sequence success probability.",
                        "AR-correlated rows are not independent trials; covariance is observational, not causal.",
                        "Arm-specific baseline comparison uses only original calibration FIT worlds, no held-out fitting or optimizer."],
              "records": records, "input_manifest": manifest, "input_hashes_unchanged": True}
    target = args.output / "audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(target), "batches": len(records), "inputs": len(manifest), "unchanged": True}))


if __name__ == "__main__": main()
