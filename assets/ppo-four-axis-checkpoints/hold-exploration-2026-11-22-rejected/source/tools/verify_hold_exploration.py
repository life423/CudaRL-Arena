"""Independent saved-artifact verification. Never simulate or apply an optimizer."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import verify_hold_credit as common

ROWS = 10240


def initial_depth_mm(pose):
    """Pure float32 geometry for original yaw-only reset; no time integration."""
    f = np.float32
    assert pose["quat_x"] == pose["quat_z"] == pose["angle1"] == pose["angle3"] == 0
    pi, radius, length = f(3.141592653589793), f(.0015), f(.026)
    head = f(f(f(f(f(4)*pi)*f(f(.02)*f(.5)))*f(.0064))*f(.008))/f(3)
    rod = f(f(f(pi*radius)*radius)*length)
    tail = f(rod+f(f(f(.0097)*length)*f(.0001)))
    masses = [f(head*f(1000)), f(rod*f(1000)), f(tail*f(1000))]
    total_mass = f(0)
    for mass in masses: total_mass = f(total_mass+mass)
    def com_y(y):
        center = f(f(y)*f(.04)); weighted = f(0)
        for mass in masses: weighted = f(weighted+f(center*mass))
        return f(weighted*f(f(1)/f(total_mass*f(.04))))
    return float(f(f(abs(f(com_y(pose["position_y_model"])-com_y(-2)))*f(.04))*f(1000)))


def hold_assessment(directory):
    """Include the reset sample, which can be the peak for a quieter actor."""
    trace = common.rows(directory/"rest/controls.csv")
    summary = common.rows(directory/"rest/summary.csv")
    poses = list(csv.DictReader(common.read(directory/"rest/poses.csv").decode().splitlines()))
    for pose in poses:
        for key in pose:
            if key != "category": pose[key] = float(pose[key])
    reset = {int(x["case"]): x for x in poses if x["control"] == 0}
    breaches = {int(x["case"]): x for x in common.rows(directory/"first-hold-breach.csv")}
    assert len(trace) == 7680 and len(summary) == 6 and set(reset) == set(breaches) == set(range(6))
    details = []
    for case in range(6):
        seq = [x for x in trace if x["case"] == case]
        assert [x["control"] for x in seq] == list(range(1, 1281))
        depth0 = initial_depth_mm(reset[case])
        row = next(x for x in summary if x["case"] == case)
        assert abs(depth0-abs(row["offset_mm"])) < 2e-5
        at56 = seq[1119]["forward_mm"]
        peaks = {"speed": max(x["speed_mm_s"] for x in seq),
                 "anchor": max(depth0, max(x["anchor_error_mm"] for x in seq)),
                 "depth": max(depth0, max(abs(x["depth_mm"]) for x in seq)),
                 "lane": max(x["lane_mm"] for x in seq), "tilt": max(x["tilt_deg"] for x in seq),
                 "excursion": max(abs(x["forward_mm"]-at56) for x in seq[1119:])}
        limits = {"speed": .5, "anchor": 2, "depth": 1, "lane": 2, "tilt": 10, "excursion": 2}
        gates = {key: value <= limits[key] for key, value in peaks.items()}
        first_time, first_cause = None, "none"
        for x in seq:
            causes = []
            if x["speed_mm_s"] > .5: causes.append("speed")
            if x["anchor_error_mm"] > 2: causes.append("anchor")
            if abs(x["depth_mm"]) > 1: causes.append("depth")
            if x["lane_mm"] > 2: causes.append("lane")
            if x["tilt_deg"] > 10: causes.append("tilt")
            if x["control"] > 1120 and abs(x["forward_mm"]-at56) > 2: causes.append("final8_excursion")
            if causes: first_time, first_cause = x["time_s"], "|".join(causes); break
        assert breaches[case]["cause"] == first_cause
        if first_time is not None: common.near(breaches[case]["time_s"], first_time, 1e-9)
        assert row["passed"] == int(all(gates.values()))
        for key, field in [("speed", "max_speed_mm_s"), ("anchor", "max_anchor_error_mm"),
                           ("depth", "max_depth_mm"), ("lane", "max_lane_mm"), ("tilt", "max_tilt_deg"),
                           ("excursion", "final8_forward_excursion_mm")]:
            common.near(peaks[key], row[field], 1e-6)
        details.append({"case": case, "passed": all(gates.values()), "gates": gates, "peaks": peaks,
                        "initial_depth_mm": depth0, "first_breach_s": first_time, "first_breach_cause": first_cause,
                        "final8_mean_speed_mm_s": float(np.mean([x["speed_mm_s"] for x in seq[1120:]])),
                        "final8_forward_delta_mm": seq[-1]["forward_mm"]-at56, "net_forward_mm": seq[-1]["forward_mm"],
                        "final_speed_mm_s": seq[-1]["speed_mm_s"]})
    return {"passes": sum(x["passed"] for x in details), "cases": details, "controls": len(trace)}


# Reuse the prior independently checked swim parser, while this study's reset
# poses supply the missing initial sample for the stricter hold-peak arithmetic.
common.hold_assessment = hold_assessment


def schedule():
    row = np.arange(ROWS)
    world, slot = row % 32, row // 32
    hold = (world // 2) % 2 == 1
    age = np.where(slot % 80 < 20, slot % 80, slot % 80 - 20)
    length = np.where(slot % 80 < 20, 20, 60)
    bins = (world // 8 + slot // 80) % 4
    cell = np.where(hold, 640+(world % 2)*320+bins*80+np.where(length == 20, 0, 20)+age,
                    (world % 2)*320+slot)
    return {"mode": hold.astype(int), "hold": hold, "fit": world % 8 < 4,
            "slot": slot, "world": world, "bin": np.where(hold, bins, -1), "cell": cell,
            "age": np.where(hold, age, slot), "length": np.where(hold, length, 320)}


def migrated(path, protocol):
    r = common.envelope(path)
    assert r.value("I") == 0x35415050 and r.value("I") == 5
    assert r.value("Q") == protocol and r.value("I") == 1247037243 and r.value("i") == 336
    assert [r.value("I") for _ in range(9)] == [6, 6, 3, 9, 4, 41, 4, 2532, 2433]
    assert r.value("f") == float(np.float32(.8))
    assert np.array_equal(r.array(size=4), np.full(4, -1))
    base, critic = r.array(size=2532), r.array(size=2433)
    assert np.all(base[:1312].reshape(32, 41)[:, 40] == 0)
    ages = []
    for n in [2532, 2433]:
        cfg = [r.value("f") for _ in range(5)]
        assert np.all(np.isfinite(cfg))
        ages.append(r.value("i"))
        assert np.isfinite(r.array(size=n)).all()
        assert np.all(r.array(size=n) >= 0)
    assert ages == [21700, 21700] and [r.value("i"), r.value("i")] == [0, 0]
    r.finish()
    return base.astype(np.float64)


def read_policy(path, protocol, parent_checksum, expected_base):
    r = common.envelope(path)
    assert [r.value("I"), r.value("I")] == [0x32454841, 1]
    assert r.value("Q") == protocol and r.value("Q") == parent_checksum
    update, steps = r.value("i"), r.value("i")
    assert [r.value("I"), r.value("I")] == [41, 4]
    base, gains = r.array(size=2532), r.array(size=4)
    a = common.Reader(r.array("u1").tobytes()); r.finish()
    cfg = np.array([a.value("f") for _ in range(5)], dtype=np.float32)
    assert np.array_equal(cfg, np.array([.0025, .9, .999, .00001, .5], dtype=np.float32))
    age = a.value("i")
    first, second = a.array(size=4), a.array(size=4); a.finish()
    assert age == steps and 0 <= steps <= 2560 and np.array_equal(base, expected_base)
    assert np.isfinite(gains).all() and np.isfinite(first).all() and np.all(np.isfinite(second) & (second >= 0))
    if update == 336:
        assert steps == 0 and np.all(gains == 1) and np.all(first == 0) and np.all(second == 0)
    return gains.astype(np.float64), {"update": update, "steps": steps, "gains": gains.tolist()}


def baseline(path, protocol):
    r = common.envelope(path)
    assert [r.value("I"), r.value("I")] == [0x32454842, 1] and r.value("Q") == protocol
    means, counts = r.array("<f8", 1280), r.array("<i4", 1280)
    scales = np.array([r.value("d"), r.value("d")]); r.finish()
    assert np.all(counts == 4) and np.all(np.isfinite(means)) and np.all(np.isfinite(scales) & (scales > 0))
    return means, scales


def signal(data, means, scales, state):
    out = common.signal(data, means, scales, state)
    residual = data["returns"]-means[state["cell"]]
    groups = {}
    for duration in [20, 60]:
        for gain_bin in range(4):
            mask = ~state["fit"] & state["hold"] & (state["length"] == duration) & (state["bin"] == gain_bin)
            x = residual[mask]
            assert len(x) == (160 if duration == 20 else 480)
            mse = float(np.mean(x*x))
            groups[f"duration{duration}_bin{gain_bin}"] = {"rows": len(x), "bias": float(x.mean()), "MSE": mse,
                                                          "offset_energy_fraction": float(x.mean()**2/mse) if mse > 0 else None}
    out["by_duration_and_gain_bin"] = groups
    scales_out = {}
    for task, label in [(0, "swim"), (1, "hold")]:
        values = abs(data["returns"][state["mode"] == task])
        p99 = float(np.sort(values)[int(np.ceil(.99*len(values)))-1])
        scales_out[label] = {"absolute_p99": p99, "absolute_max": float(values.max()),
                             "passed": bool(p99 <= 5 and values.max() <= 20)}
    out["return_scale"] = scales_out
    return out


def batch(directory, update, state, base, expected_gains):
    d = common.read_batch(directory/f"batch-{update}.bin", state)
    r = common.envelope(directory/f"exploration-evidence-{update}.bin")
    assert [r.value("I"), r.value("I")] == [0x31504548, 1]
    cur, prev = r.array(size=ROWS*4).reshape(-1, 4), r.array(size=ROWS*4).reshape(-1, 4)
    amplitude, bins = r.array(size=ROWS), r.array(size=ROWS)
    post = r.array(size=ROWS*5).reshape(-1, 5)
    gains = r.array(size=4); r.finish()
    assert np.array_equal(gains, expected_gains) and np.isfinite(post).all() and np.all(post >= 0)
    base_error = max(np.max(abs(cur-common.forward(base, d["obs"]))),
                     np.max(abs(prev-common.forward(base, d["prevobs"]))))
    assert base_error < 2e-5
    assert np.array_equal(bins, state["bin"])
    hold, hist = state["hold"], d["history"]
    assert np.all(amplitude[~hold] == 1)
    logs = np.log(amplitude[hold]) / np.log(100) + 1
    assert np.all(logs >= bins[hold]/4-1e-6) and np.all(logs <= (bins[hold]+1)/4+1e-6)
    rho = float(np.float32(.8))
    conditional = cur.astype(float) + rho*hist[:, None]*(d["prevaction"]-prev)
    conditional[hold] = (amplitude[hold, None]*gains*(cur[hold]-rho*hist[hold, None]*prev[hold])
                         + rho*hist[hold, None]*d["prevaction"][hold])
    variance = amplitude.astype(float)**2*np.exp(-2)*(1-rho*rho*hist)
    logp = -.5*np.sum((d["action"]-conditional)**2/variance[:, None]+np.log(2*np.pi*variance[:, None]), axis=1)
    logp_error = float(np.max(abs(logp-d["logp"]))); assert logp_error < 1e-4
    selected = np.flatnonzero(hold)[[0, 16, 320, 640, 1000, 3000, 5000]]
    bdiff = cur[selected]-rho*hist[selected, None]*prev[selected]
    dc = amplitude[selected, None]*bdiff
    analytic = np.mean((d["action"][selected]-conditional[selected])/variance[selected, None]*dc, axis=0)
    numeric = []
    for axis in range(4):
        objectives = []
        for delta in [1e-6, -1e-6]:
            shifted = conditional[selected].copy(); shifted[:, axis] += delta*dc[:, axis]
            lp = -.5*np.sum((d["action"][selected]-shifted)**2/variance[selected, None]+np.log(2*np.pi*variance[selected, None]), axis=1)
            objectives.append(float(lp.mean()))
        numeric.append((objectives[0]-objectives[1])/2e-6)
    fd_error = float(np.max(abs(analytic-numeric)))
    assert fd_error < 1e-5*max(1, float(np.max(abs(analytic))))
    csvrows = common.rows(directory/f"rows-{update}.csv")
    for field, values in [("amplitude", amplitude), ("gain_bin", bins), ("post_speed_m_s", post[:, 0]),
                          ("post_anchor_m", post[:, 1]), ("post_depth_m", post[:, 2]),
                          ("post_lane_m", post[:, 3]), ("post_tilt_deg", post[:, 4])]:
        assert np.max(abs(np.array([x[field] for x in csvrows])-values)) < 1e-9
    complete, groups, details = 0, np.zeros((2, 2), int), []
    for world in np.flatnonzero(hold[:32]):
        for start in [0, 20, 80, 100, 160, 180, 240, 260]:
            count = 20 if start % 80 == 0 else 60
            indices = np.arange(start, start+count)*32+world
            assert np.all(amplitude[indices] == amplitude[indices[0]])
            good = np.all(d["speed"][indices] <= .0005) and np.all(post[indices] <= np.array([.0005, .002, .001, .002, 10]))
            if good: complete += 1; groups[world % 2, int(count == 60)] += 1
            details.append({"world": int(world), "start": start, "duration": count,
                            "amplitude": float(amplitude[indices[0]]), "bin": int(bins[indices[0]]), "quiet": bool(good)})
    quiet = int(np.sum(hold & (d["speed"] <= .0005)))
    heldquiet = int(np.sum(hold & ~state["fit"] & (d["speed"] <= .0005)))
    decoded = np.tanh(d["action"])
    command_counts = {}
    for threshold in [.03, .1]:
        mask = np.max(abs(decoded), axis=1) <= threshold
        longest = 0
        for segment in details:
            span = np.arange(segment["start"], segment["start"]+segment["duration"])*32+segment["world"]
            streak = 0
            for value in mask[span]:
                streak = streak+1 if value else 0
                longest = max(longest, streak)
        command_counts[str(threshold)] = {"rows": int(np.sum(mask & hold)), "longest_controls": longest}
    return d, {"collection_update": update, "gains": gains.tolist(), "oldlogp_max_error": logp_error,
               "backbone_inference_max_error": float(base_error), "return_error": d["return_error"],
               "independent_gain_score_finite_difference_error": fd_error,
               "quiet_rows": quiet, "heldout_quiet_rows": heldquiet, "complete_quiet_segments": complete,
               "quiet_pose_duration_groups": groups.tolist(), "segments": details, "commands": command_counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("build/ppo/hold-exploration-2026-11-22-v1"))
    p.add_argument("--calibration-only", action="store_true")
    args = p.parse_args(); root = args.run.resolve()
    common.HASHES.clear()
    report = {"scope": "Independent saved-artifact verification; zero physics/optimizer steps", "passed": False,
              "limitations": ["Frozen-source and full prior-artifact preservation are coordinator checks.",
                             "CPU assessment action interpretation g=1 follows frozen native source; metrics are independently recomputed.",
                             "Short swimming velocity RMSE uses saved native summary; physical gates and long RMSE are independently checked."]}
    try:
        common.read(Path(__file__))
        bound_protocol = common.envelope(root/"protocol.bin")
        protocol = common.fnv(bound_protocol.data)
        assert [bound_protocol.value("I"), bound_protocol.value("I")] == [0x32454850, 1]
        bound_protocol.array("u1")
        assert bound_protocol.value("Q") == common.fnv(common.read(root/"approved-protocol.md"))
        assert [bound_protocol.value("i") for _ in range(14)] == [20,60,80,320,32,41,4,1280,20960,691680,6916800,32,2560,120]
        assert np.array_equal(np.array([bound_protocol.value("f") for _ in range(3)], np.float32), np.array([.01,1,.0025], np.float32))
        assert bound_protocol.value("I") == 0x61a10e11
        bound_protocol.array("u1"); bound_protocol.finish()
        parent = common.envelope(root/"parent-checkpoint.bin")
        parent_checksum = common.fnv(parent.data)
        assert hashlib.sha256((root/"parent-checkpoint.bin").read_bytes()).hexdigest().upper() == "810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0"
        base = migrated(root/"migrated-initial.bin", protocol)
        initial, original = common.assessment(root/"baseline")
        assert initial["short_passes"] == 4 and initial["hold"]["passes"] == 0
        assert original == {(pose, t) for pose in [0, 3] for t in range(8, 61, 4)}
        state = schedule(); cal_result = common.js(root/"calibration-result.json")
        assert cal_result["closed"] and not cal_result["fault"]
        assert cal_result["controls"] == 20960 and cal_result["substeps"] == 209600
        assert cal_result["actor_steps"] == cal_result["critic_steps"] == cal_result["critic_forward_calls"] == 0
        data, cal = batch(root/"calibration", 336, state, base, np.ones(4))
        means, scales = baseline(root/"fixed-baseline.bin", protocol)
        total, counts = np.zeros(1280), np.zeros(1280, int)
        np.add.at(total, state["cell"][state["fit"]], data["returns"][state["fit"]])
        np.add.at(counts, state["cell"][state["fit"]], 1)
        assert np.all(counts == 4) and np.max(abs(total/counts-means)) < 1e-10
        residual = data["returns"]-means[state["cell"]]
        rms = np.array([np.sqrt(np.mean(residual[state["fit"] & (state["mode"] == task)]**2)) for task in [0, 1]])
        assert np.max(abs(rms-scales)) < 1e-10
        sig = signal(data, means, scales, state)
        native = common.js(root/"calibration-report.json")
        gate = cal["heldout_quiet_rows"] >= 256 and .1 <= sig["hold_share"] <= .9 and cal["complete_quiet_segments"] >= 4 and np.all(np.array(cal["quiet_pose_duration_groups"]) > 0)
        gate = bool(gate and all(sig["return_scale"][goal]["passed"] for goal in ["hold", "swim"]))
        assert native["passed"] == gate
        assert native["complete_quiet_segments"] == cal["complete_quiet_segments"] and native["heldout_hold_quiet"] == cal["heldout_quiet_rows"]
        common.near(native["hold_share"], sig["hold_share"], 1e-9)
        for pose in [0, 1]:
            for duration_index, duration in enumerate([20, 60]):
                assert native[f"quiet_pose{pose*3}_len{duration}"] == cal["quiet_pose_duration_groups"][pose][duration_index]
        for threshold, field in [("0.03", "0p03"), ("0.1", "0p1")]:
            assert native[f"neutral_{field}_rows"] == cal["commands"][threshold]["rows"]
            assert native[f"neutral_{field}_longest_controls"] == cal["commands"][threshold]["longest_controls"]
        report.update({"baseline": initial, "calibration": cal, "calibration_signal": sig, "calibration_passed": gate, "scales": scales.tolist()})
        controls = 20960; updates, policies = [], []
        train_path = root/"training-result.json"
        if train_path.exists() and not args.calibration_only:
            training = common.js(train_path); assert training["closed"] and not training["fault"]
            directory = root/"training-started"
            head = {}
            for path in sorted(directory.glob("exploration-policy-update-*.bin")):
                gains, meta = read_policy(path, protocol, parent_checksum, base)
                head[meta["update"]] = gains; policies.append(meta)
            assert [x["update"] for x in policies] == list(range(336, 337+training["PPO_updates"]))
            rows = [json.loads(line) for line in common.read(root/"updates.jsonl").decode().splitlines() if line]
            for path in sorted(directory.glob("batch-*.bin")):
                update = int(path.stem.split("-")[-1])
                current, detail = batch(directory, update, state, base, head[update])
                detail["signal"] = signal(current, means, scales, state)
                controls += 10240
                if update == 336 and training["PPO_updates"] > 0:
                    assert detail["heldout_quiet_rows"] >= 256 and detail["complete_quiet_segments"] >= 4
                    assert np.all(np.array(detail["quiet_pose_duration_groups"]) > 0)
                    assert .1 <= detail["signal"]["hold_share"] <= .9
                    assert all(x["passed"] for x in detail["signal"]["return_scale"].values())
                if (directory/f"assessment-{update+1}").exists():
                    measured, _ = common.assessment(directory/f"assessment-{update+1}", original)
                    controls += measured["controls"]
                    native = next(x for x in rows if x["update"] == update+1)
                    assert native["short"] == measured["short_passes"] and native["holds"] == measured["hold"]["passes"]
                    assert native["long"] == measured["long_passes"] and native["retained"] == measured["retained"]
                    assert native["complete_quiet_segments"] == detail["complete_quiet_segments"] and native["controls"] == controls
                    assert native["actor_steps"] == next(x["steps"] for x in policies if x["update"] == update+1)
                    for relative in ["frozen-controls.csv", "frozen-evaluation.csv", "long/sustained-controls.csv", "long/sustained-windows.csv"]:
                        assert common.read(directory/f"assessment-{update+1}"/relative) == common.read(root/"baseline"/relative), "Frozen swimming changed"
                    detail["assessment"] = measured
                updates.append(detail)
            assert len(rows) == training["PPO_updates"] and controls == training["controls"] and controls*10 == training["substeps"]
            assert training["actor_steps"] == policies[-1]["steps"] and training["PPO_updates"] <= 32
            assert training["critic_steps"] == training["critic_forward_calls"] == 0
            assert not training["runtime_adopted"] and training["retries"] == 0
            assert all(40 <= b["steps"]-a["steps"] <= 80 for a, b in zip(policies, policies[1:]))
            assert all(x["assessment"]["retained"] for x in updates[:-1] if "assessment" in x)
            if training["qualified"]:
                assert updates[-1]["assessment"]["hold"]["passes"] == 6 and updates[-1]["assessment"]["retained"]
            if training["gate"] == "regressed_swimming": assert not updates[-1]["assessment"]["retained"]
            assert training["active_seconds"]+training["prior_active_seconds"] <= 120
            report["training"] = training
        elif not args.calibration_only:
            assert not gate, "Training result is pending; use --calibration-only for Phase A"
        assert controls <= 691680
        for path, digest in list(common.HASHES.items()): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        report.update({"passed": True, "phase": "calibration" if args.calibration_only else "closed_attempt", "updates": updates,
                       "policies": policies, "counted_controls": controls, "counted_substeps": controls*10, "input_sha256": common.HASHES})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["input_sha256"] = common.HASHES
        (root/"independent-verification.json").write_text(json.dumps(report, indent=2)+"\n")
        raise
    (root/"independent-verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "phase": report["phase"], "controls": controls, "updates": len(updates), "input_files": len(common.HASHES)}))


if __name__ == "__main__": main()
