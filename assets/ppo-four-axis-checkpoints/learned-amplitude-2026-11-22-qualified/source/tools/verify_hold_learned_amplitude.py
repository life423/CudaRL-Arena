"""Read-only verification of whole-episode amplitude PPO. No physics/optimizer."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import verify_hold_credit as common
from verify_hold_exploration import initial_depth_mm, migrated

N, T = 256, 1280
LIMITS = np.array([.0005, .002, .001, .002, 10.])
VALIDATED = set()


def fast_envelope(path):
    path = Path(path).resolve()
    raw = common.read(path)
    magic, size, _ = struct.unpack_from("<QQQ", raw)
    assert magic == 0x31504B434C524443 and size == len(raw)-24
    assert path in VALIDATED, f"Envelope was not independently checked: {path}"
    return common.Reader(raw[24:])


def validate_envelopes(root, helper):
    paths = sorted(root.rglob("*.bin"))
    listing = root/"independent-envelope-paths.txt"
    listing.write_text("".join(str(x.resolve())+"\n" for x in paths), encoding="utf-8")
    common.read(helper)
    result = subprocess.run([str(helper.resolve()), str(listing.resolve())], capture_output=True, text=True,
                            creationflags=0x08000000 if os.name == "nt" else 0, check=True)
    status = json.loads(result.stdout)
    assert status["passed"] and status["files"] == len(paths)
    assert status["physical_controls"] == status["optimizer_steps"] == 0
    VALIDATED.update(x.resolve() for x in paths)
    return status


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -x))


def q_logp(parameters, latent):
    mu, v = parameters
    sigma = .05+7.95*float(sigmoid(v))
    return -.5*((np.asarray(latent)-mu)/sigma)**2-math.log(sigma)-.5*math.log(2*math.pi)


def loo(returns, scale):
    answer = np.empty(N)
    for case in range(6):
        indices = np.arange(case, N, 6)
        assert len(indices) == (43 if case < 4 else 42)
        total = np.sum(returns[indices], dtype=np.float64)
        answer[indices] = (returns[indices]-(total-returns[indices])/(len(indices)-1))/scale
    return answer


def hold_assessment(directory):
    root = directory/"rest" if (directory/"rest").exists() else directory
    trace, summary, metadata = common.rows(root/"controls.csv"), common.rows(root/"summary.csv"), common.rows(root/"cases.csv")
    poses = list(csv.DictReader(common.read(root/"poses.csv").decode().splitlines()))
    for row in poses:
        for key in row:
            if key != "category": row[key] = float(row[key])
    reset = {int(x["case"]): x for x in poses if x["control"] == 0}
    assert len(summary) == len(metadata) == len(reset)
    details, controls = [], 0
    for meta in metadata:
        case, count = int(meta["case"]), int(meta["controls"])
        sequence = [x for x in trace if int(x["case"]) == case]
        assert [x["control"] for x in sequence] == list(range(1, count+1))
        assert count in [1280, 6000]
        controls += count
        depth = initial_depth_mm(reset[case])/1000
        expected_initial = [0, depth, depth, 0, 0]
        recorded_initial = [meta[f"initial_{name}"] for name in ["speed_m_s", "anchor_m", "depth_m", "lane_m", "tilt_deg"]]
        assert np.max(abs(np.array(expected_initial)-recorded_initial)) < 2e-8
        forward_start = sequence[count-161]["forward_mm"]
        excursion = max(abs(x["forward_mm"]-forward_start) for x in sequence[count-161:])
        peaks = [max(x["speed_mm_s"] for x in sequence), max(depth*1000, max(x["anchor_error_mm"] for x in sequence)),
                 max(depth*1000, max(abs(x["depth_mm"]) for x in sequence)), max(x["lane_mm"] for x in sequence),
                 max(x["tilt_deg"] for x in sequence)]
        scales = np.array([1000, 1000, 1000, 1000, 1])
        gates = (np.array(peaks) <= LIMITS*scales).tolist()+[excursion <= 2]
        native = next(x for x in summary if int(x["case"]) == case)
        assert bool(native["passed"]) == all(gates)
        for value, key in zip(peaks, ["max_speed_mm_s", "max_anchor_error_mm", "max_depth_mm", "max_lane_mm", "max_tilt_deg"]):
            common.near(value, native[key], 1e-6)
        common.near(excursion, native["final8_forward_excursion_mm"], 1e-6)
        first = next((x for x in sequence if x["speed_mm_s"] > .5 or x["anchor_error_mm"] > 2 or abs(x["depth_mm"]) > 1
                      or x["lane_mm"] > 2 or x["tilt_deg"] > 10
                      or (x["control"] >= count-160 and abs(x["forward_mm"]-forward_start) > 2)), None)
        details.append({"case": case, "pose": int(meta["pose"]), "offset_mm": meta["offset_mm"], "controls": count,
                        "gain": meta["gain"], "passed": all(gates), "gates": gates, "peaks_mm_and_degrees": peaks,
                        "final8_excursion_mm": excursion, "first_breach_s": first["time_s"] if first else None,
                        "final8_mean_speed_mm_s": float(np.mean([x["speed_mm_s"] for x in sequence[-160:]]))})
    assert controls == len(trace)
    return {"passes": sum(x["passed"] for x in details), "cases": details, "controls": controls}


common.hold_assessment = hold_assessment


def policy(path, protocol, parent, frozen):
    r = fast_envelope(path)
    assert [r.value("I"), r.value("I")] == [0x314d4148, 1]
    assert r.value("Q") == protocol and r.value("Q") == parent
    assert [r.value("I") for _ in range(7)] == [6, 6, 3, 4, 41, 4, 336]
    update, age = r.value("i"), r.value("i")
    base, parameters = r.array(size=2532), r.array(size=2)
    stored_gain, stored_sigma = r.value("f"), r.value("f")
    assert abs(stored_gain-float(sigmoid(parameters[0]))) < 2e-7
    assert abs(stored_sigma-(.05+7.95*float(sigmoid(parameters[1])))) < 2e-6
    optimizer = common.Reader(r.array("u1").tobytes()); r.finish()
    config = np.array([optimizer.value("f") for _ in range(5)], np.float32)
    assert np.array_equal(config, np.array([.05, .9, .999, .00001, .5], np.float32))
    assert optimizer.value("i") == age and 0 <= age <= 512
    first, second = optimizer.array(size=2), optimizer.array(size=2); optimizer.finish()
    assert np.array_equal(base, frozen) and np.isfinite(parameters).all()
    assert np.isfinite(first).all() and np.all(np.isfinite(second) & (second >= 0))
    if update == 0: assert age == 0 and np.all(first == 0) and np.all(second == 0)
    return parameters.astype(float), {"update": update, "steps": age, "parameters": parameters.tolist(),
                                      "deterministic_gain": float(sigmoid(parameters[0])),
                                      "sigma": .05+7.95*float(sigmoid(parameters[1]))}


def read_batch(path, frozen, expected_parameters=None):
    r = fast_envelope(path)
    assert [r.value("I"), r.value("I"), r.value("I"), r.value("I")] == [0x31424148, 1, N, T]
    ticks = r.value("i"); assert ticks == T
    assert [r.value("I") for _ in range(3)] == [41, 4, 6]
    parameters, latent, gain, oldp, returns = [r.array() for _ in range(5)]
    assert len(parameters) == 2 and all(len(x) == N for x in [latent, gain, oldp, returns])
    if expected_parameters is not None: assert np.array_equal(parameters, expected_parameters)
    abi = r.value("I")
    raw_counts = r.array(np.dtype((np.void, abi))); assert len(raw_counts) == N and abi >= 60
    dtype = np.dtype({"names": ["controls", "substeps", "sum", "peaks", "speed_square", "forward56", "excursion", "failure"],
                      "formats": ["<u8", "<u8", "<f8", ("<f4", 5), "<f4", "<f4", "<f4", "<i4"],
                      "offsets": [0, 8, 16, 24, 44, 48, 52, 56], "itemsize": abi})
    counts = np.frombuffer(raw_counts.tobytes(), dtype=dtype)
    initial = r.array(size=N*5).reshape(N, 5)
    obs = r.array(size=N*T*41).reshape(-1, 41)
    base = r.array(size=N*T*4).reshape(-1, 4)
    action = r.array(size=N*T*4).reshape(T, N, 4)
    rewards = r.array(size=N*T).reshape(T, N)
    metrics = r.array(size=N*T*6).reshape(T, N, 6)
    for _ in range(2):
        width, count = r.value("I"), r.value("Q")
        assert count == N; r.at += width*count
    r.finish()
    for x in [parameters, latent, gain, oldp, returns, initial, obs, base, action, rewards, metrics]: assert np.isfinite(x).all()
    assert np.all(counts["controls"] == T) and np.all(counts["substeps"] == T*10) and np.all(counts["failure"] == 0)
    assert np.max(abs(gain-sigmoid(latent))) < 2e-7 and np.all((gain >= 0) & (gain <= 1))
    probability_error = float(np.max(abs(oldp-q_logp(parameters, latent)))); assert probability_error < 1e-4
    sums = np.sum(rewards, axis=0, dtype=np.float64)
    assert np.max(abs(sums-counts["sum"])) < 1e-7+1e-7*max(1, float(np.max(abs(sums))))
    assert np.array_equal(np.float32(sums/64), returns)
    assert np.array_equal(action, gain[None, :, None]*base.reshape(T, N, 4))
    expected_clock = np.maximum(np.float32(0), np.float32(1)-np.arange(T, dtype=np.float32)/np.float32(40))
    assert np.max(abs(obs[:, 34].reshape(T, N)-expected_clock[:, None])) < 1e-7
    assert np.all(obs[:, 3:6] == 0) and np.all(obs[:N, 12:18] == 0) and np.all(obs[:N, 40] == 0)
    backbone_error = 0.
    for start in range(0, N*T, 8192):
        stop = min(start+8192, N*T)
        predicted = common.forward(frozen, obs[start:stop].astype(float))
        backbone_error = max(backbone_error, float(np.max(abs(predicted-base[start:stop]))))
    assert backbone_error < 2e-5
    peaks = np.maximum(initial, np.max(metrics[:, :, :5], axis=0))
    assert np.max(abs(peaks-counts["peaks"])) < 1e-7
    excursion = np.max(abs(metrics[T-161:, :, 5]-metrics[T-161, :, 5]), axis=0)
    assert np.max(abs(excursion-counts["excursion"])) < 2e-7
    physical = np.all(peaks <= LIMITS.astype(np.float32), axis=1) & (excursion <= np.float32(.002))
    episode_path = path.with_name(path.name.replace("-batch-", "-episodes-").replace(".bin", ".csv"))
    table = common.rows(episode_path); assert len(table) == N
    for i, row in enumerate(table):
        assert row["episode"] == i and row["case"] == i % 6 and row["pose"] == (3 if i % 2 else 0)
        assert row["controls"] == T and row["substeps"] == T*10 and bool(row["passed"]) == bool(physical[i])
        for key, value in [("latent_u", latent[i]), ("gain", gain[i]), ("old_logp", oldp[i]), ("return", returns[i]), ("reward_sum", sums[i])]:
            common.near(row[key], float(value), 1e-9)
    return returns.astype(float), {"parameters": parameters.tolist(), "quiet": int(physical.sum()),
                                  "quiet_pose0": int(np.sum(physical[np.arange(N) % 2 == 0])),
                                  "quiet_pose3": int(np.sum(physical[np.arange(N) % 2 == 1])),
                                  "probability_error": probability_error, "backbone_error": backbone_error,
                                  "return_error": float(np.max(abs(sums-counts["sum"]))),
                                  "gain_min": float(gain.min()), "gain_max": float(gain.max()),
                                  "quiet_gain_max": float(gain[physical].max()) if np.any(physical) else None,
                                  "controls": N*T}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("build/ppo/learned-amplitude-2026-11-22-v1"))
    p.add_argument("--helper", type=Path, default=Path("build/learned-amplitude-preparation-2026-11-22/check-evidence.exe"))
    p.add_argument("--calibration-only", action="store_true")
    args = p.parse_args(); root = args.run.resolve(); common.HASHES.clear()
    report = {"scope": "Independent saved evidence only; zero physics/optimizer steps", "passed": False,
              "limitations": ["Frozen-source/full-prior-artifact preservation are coordinator checks.",
                             "Short-swim RMSE uses native summary; physical gates and long-swim RMSE are independently checked."]}
    try:
        common.read(Path(__file__))
        report["envelopes"] = validate_envelopes(root, args.helper)
        common.envelope = fast_envelope
        protocol = common.fnv(fast_envelope(root/"protocol.bin").data)
        parent = common.fnv(fast_envelope(root/"parent-checkpoint.bin").data)
        frozen = migrated(root/"migrated-initial.bin", protocol)
        assert hashlib.sha256((root/"parent-checkpoint.bin").read_bytes()).hexdigest().upper() == "810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0"
        baseline, original = common.assessment(root/"baseline")
        assert baseline["short_passes"] == 4 and baseline["hold"]["passes"] == 0
        assert original == {(pose, t) for pose in [0, 3] for t in range(8, 61, 4)}
        report["baseline"] = baseline
        initial_parameters, initial_policy = policy(root/"initial-policy/amplitude-policy-update-0.bin", protocol, parent, frozen)
        initial = hold_assessment(root/"initial-hold")
        assert initial["passes"] == 0 and initial["controls"] == 7680
        assert all(abs(x["gain"]-initial_policy["deterministic_gain"]) < 2e-7 for x in initial["cases"])
        cal_result = common.js(root/"calibration-result.json")
        assert cal_result["closed"] and not cal_result["fault"]
        assert cal_result["controls"] == 346080 and cal_result["substeps"] == 3460800
        assert cal_result["actor_steps"] == cal_result["critic_steps"] == cal_result["critic_forward_calls"] == 0
        returns, calibration = read_batch(root/"calibration-started/calibration-batch-0.bin", frozen, initial_parameters)
        scale_reader = fast_envelope(root/"fixed-scale.bin")
        assert [scale_reader.value("I"), scale_reader.value("I")] == [0x31534148, 1] and scale_reader.value("Q") == protocol
        scale = scale_reader.value("d")
        means = np.array([scale_reader.value("d") for _ in range(6)])
        counts = [scale_reader.value("i") for _ in range(6)]; scale_reader.finish()
        assert counts == [43, 43, 43, 43, 42, 42]
        expected_means = np.array([returns[np.arange(N) % 6 == case].mean() for case in range(6)])
        assert np.max(abs(means-expected_means)) < 1e-12
        expected_scale = float(np.sqrt(np.mean((returns-means[np.arange(N) % 6])**2)))
        assert abs(expected_scale-scale) < 1e-12 and scale > 0
        native_cal = common.js(root/"calibration-report.json")
        bounds = np.sort(abs(returns)); p99 = float(bounds[int(math.ceil(.99*N))-1])
        gate = calibration["quiet"] >= 4 and calibration["quiet_pose0"] > 0 and calibration["quiet_pose3"] > 0 and p99 <= 5 and bounds[-1] <= 20
        assert native_cal["passed"] == gate
        for key, value in [("quiet_complete_64s", calibration["quiet"]), ("quiet_pose0", calibration["quiet_pose0"]),
                           ("quiet_pose3", calibration["quiet_pose3"]), ("fixed_global_residual_RMS", scale),
                           ("return_abs_p99", p99), ("return_abs_max", float(bounds[-1]))]:
            common.near(native_cal[key], value, 1e-9)
        report.update({"initial_policy": initial_policy, "initial_hold": initial, "calibration": calibration,
                       "calibration_passed": bool(gate), "calibration_scale": scale, "calibration_return_p99": p99,
                       "calibration_return_max": float(bounds[-1]), "counted_controls": 346080, "counted_substeps": 3460800})
        controls = 346080
        updates, policies = [], []
        if not args.calibration_only:
            training_path = root/"training-result.json"
            assert training_path.exists() or not gate, "Training is still pending; use --calibration-only"
            if training_path.exists():
                training = common.js(training_path)
                assert training["closed"] and not training["fault"], "A partial hard fault needs its recorded-attempt decoder"
                assert training["critic_steps"] == training["critic_forward_calls"] == 0
                assert not training["runtime_adopted"] and training["retries"] == 0
                directory = root/"training-started"
                parameters_by_update = {}
                checkpoint_paths = sorted(directory.glob("amplitude-policy-update-*.bin"), key=lambda x: int(x.stem.rsplit("-", 1)[1]))
                for path in checkpoint_paths:
                    parameters, meta = policy(path, protocol, parent, frozen)
                    parameters_by_update[meta["update"]] = parameters; policies.append(meta)
                assert [x["update"] for x in policies] == list(range(training["PPO_updates"]+1))
                assert np.array_equal(parameters_by_update[0], initial_parameters)
                assert all(4 <= b["steps"]-a["steps"] <= 16 for a, b in zip(policies, policies[1:]))
                native_updates = [json.loads(line) for line in common.read(root/"updates.jsonl").decode().splitlines() if line]
                batches = sorted(directory.glob("training-batch-*.bin"), key=lambda x: int(x.stem.rsplit("-", 1)[1]))
                for path in batches:
                    update = int(path.stem.rsplit("-", 1)[1])
                    episode_returns, detail = read_batch(path, frozen, parameters_by_update[update-1])
                    controls += detail["controls"]
                    detail["update"] = update
                    bounds = np.sort(abs(episode_returns))
                    if update == 1 and training["PPO_updates"] > 0:
                        assert detail["quiet"] >= 4 and detail["quiet_pose0"] > 0 and detail["quiet_pose3"] > 0
                        assert bounds[int(math.ceil(.99*N))-1] <= 5 and bounds[-1] <= 20
                    signal_path = directory/f"advantages-{update}.bin"
                    if signal_path.exists():
                        reader = fast_envelope(signal_path)
                        assert [reader.value("I"), reader.value("I")] == [0x31474148, 1]
                        assert reader.value("i") == update and reader.value("d") == scale
                        stored_returns, advantages = reader.array(size=N), reader.array(size=N); reader.finish()
                        assert np.array_equal(stored_returns, episode_returns)
                        expected = loo(episode_returns, scale)
                        advantage_error = float(np.max(abs(advantages-expected)))
                        assert advantage_error < 1e-6
                        detail["LOO_advantage_max_error"] = advantage_error
                        detail["LOO_case_mean_advantages"] = [float(advantages[np.arange(N) % 6 == case].mean()) for case in range(6)]
                        assert max(abs(x) for x in detail["LOO_case_mean_advantages"]) < 1e-6
                    assessment_path = directory/f"assessment-{update}"
                    if assessment_path.exists():
                        assessment, _ = common.assessment(assessment_path, original)
                        controls += assessment["controls"]
                        native = next(x for x in native_updates if x["update"] == update)
                        assert native["short"] == assessment["short_passes"] and native["holds"] == assessment["hold"]["passes"]
                        assert native["long"] == assessment["long_passes"] and native["retained"] == assessment["retained"]
                        assert native["quiet_training_episodes"] == detail["quiet"] and native["controls"] == controls
                        meta = next(x for x in policies if x["update"] == update)
                        assert native["actor_steps"] == meta["steps"]
                        common.near(native["mu"], meta["parameters"][0], 1e-9)
                        common.near(native["variance_coordinate"], meta["parameters"][1], 1e-9)
                        common.near(native["sigma"], meta["sigma"], 2e-7)
                        common.near(native["deterministic_gain"], meta["deterministic_gain"], 2e-7)
                        assert all(abs(x["gain"]-meta["deterministic_gain"]) < 2e-7 for x in assessment["hold"]["cases"])
                        for relative in ["frozen-controls.csv", "frozen-evaluation.csv", "long/sustained-controls.csv", "long/sustained-windows.csv"]:
                            assert common.read(assessment_path/relative) == common.read(root/"baseline"/relative), "Frozen swimming changed"
                        detail["assessment"] = assessment
                    updates.append(detail)
                assert len(native_updates) == training["PPO_updates"]
                assert len([x for x in updates if "assessment" in x]) == training["PPO_updates"]
                assert all(x["assessment"]["retained"] for x in updates[:-1] if "assessment" in x)
                assert all(x["assessment"]["hold"]["passes"] < 6 for x in updates[:-1] if "assessment" in x)
                if training["original_success"]:
                    assert updates[-1]["assessment"]["hold"]["passes"] == 6 and updates[-1]["assessment"]["retained"]
                    withheld = hold_assessment(root/"reserved-validation/withheld")
                    endurance = hold_assessment(root/"reserved-validation/endurance")
                    assert len(withheld["cases"]) == 6 and len(endurance["cases"]) == 2
                    assert {(x["pose"], x["offset_mm"], x["controls"]) for x in withheld["cases"]} == {(pose, offset, 1280) for pose in [1,2] for offset in [0,-.25,.25]}
                    assert {(x["pose"], x["offset_mm"], x["controls"]) for x in endurance["cases"]} == {(0,0,6000), (3,0,6000)}
                    assert withheld["controls"] == 7680 and endurance["controls"] == 12000
                    controls += 19680
                    assert training["withheld_passes"] == withheld["passes"] and training["endurance_passes"] == endurance["passes"]
                    assert training["qualified"] == (withheld["passes"] == 6 and endurance["passes"] == 2)
                    report["reserved_withheld"] = withheld; report["reserved_endurance"] = endurance
                else:
                    assert not training["qualified"] and not training["reserve_consumed"]
                assert training["controls"] == controls and training["substeps"] == controls*10
                assert training["actor_steps"] == policies[-1]["steps"] and training["PPO_updates"] <= 32
                assert training["active_seconds"]+training["prior_active_seconds"] <= 900
                report["training"] = training
            report["updates"], report["policies"] = updates, policies
        assert controls <= 11194560
        for path, digest in list(common.HASHES.items()): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        report.update({"passed": True, "phase": "calibration" if args.calibration_only else "closed_attempt",
                       "counted_controls": controls, "counted_substeps": controls*10, "input_sha256": common.HASHES})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["input_sha256"] = common.HASHES
        (root/"independent-verification.json").write_text(json.dumps(report, indent=2)+"\n")
        raise
    (root/"independent-verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "phase": report["phase"], "quiet": report["calibration"]["quiet"],
                      "input_files": len(common.HASHES), "controls": report["counted_controls"]}))


if __name__ == "__main__": main()
