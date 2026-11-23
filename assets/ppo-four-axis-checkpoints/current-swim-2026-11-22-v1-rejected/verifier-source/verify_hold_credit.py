"""Independently verify closed hold-credit artifacts; never simulate or optimize.

Only reads the run and writes independent-verification.json. Frozen-input and
preservation manifests are checked by the coordinator separately.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def value(self, fmt):
        value = struct.unpack_from("<" + fmt, self.data, self.at)[0]
        self.at += struct.calcsize("<" + fmt)
        return value

    def array(self, dtype="<f4", size=None):
        n = self.value("Q")
        if size is not None:
            assert n == size, (n, size)
        dtype = np.dtype(dtype)
        assert self.at + n * dtype.itemsize <= len(self.data)
        out = np.frombuffer(self.data, dtype=dtype, count=n, offset=self.at)
        self.at += n * dtype.itemsize
        return out

    def finish(self):
        assert self.at == len(self.data), (self.at, len(self.data))


HASHES = {}


def read(path):
    data = path.read_bytes()
    HASHES[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return data


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def envelope(path):
    data = read(path)
    magic, size, checksum = struct.unpack_from("<QQQ", data)
    assert magic == 0x31504B434C524443 and size == len(data) - 24
    assert fnv(data[24:]) == checksum, path
    return Reader(data[24:])


def js(path):
    return json.loads(read(path))


def rows(path):
    out = list(csv.DictReader(read(path).decode("utf-8").splitlines()))
    for row in out:
        for key, value in row.items():
            if key not in {"cause"} and value != "":
                row[key] = float(value)
                assert np.isfinite(row[key]), (path, key, value)
    return out


def near(a, b, tolerance=2e-5):
    assert abs(a - b) <= tolerance * max(1, abs(a), abs(b)), (a, b, tolerance)


def schedule():
    r = np.arange(10240)
    world, slot = r % 32, r // 32
    mode, pose = (world // 2) % 2, world % 2
    length = np.where(slot % 80 < 20, 20, 60)
    age = np.where(slot % 80 < 20, slot % 80, slot % 80 - 20)
    cell = np.where(mode == 1, 640 + pose * 80 + np.where(length == 20, 0, 20) + age, pose * 320 + slot)
    return {"mode": mode, "hold": mode == 1, "fit": world % 8 < 4, "slot": slot,
            "age": np.where(mode == 1, age, slot), "length": np.where(mode == 1, length, 320), "cell": cell}


def raw_returns(rewards, done):
    rewards, done = rewards.reshape(320, 32), done.reshape(320, 32)
    result = np.zeros_like(rewards)
    value = np.zeros(32)
    for k in range(319, -1, -1):
        value = rewards[k] + float(np.float32(.99)) * (1 - done[k]) * value
        result[k] = value
    return result.ravel()


def read_batch(path, state):
    outer = envelope(path)
    assert outer.value("I") == 1
    inner = Reader(outer.array("u1").tobytes())
    assert [inner.value("i") for _ in range(4)] == [1, 10240, 41, 4]
    names = ["obs", "prevobs", "prevaction", "history", "action", "logp", "value", "reward", "done"]
    d = {name: inner.array().astype(np.float64) for name in names}
    inner.finish()
    costs = outer.array(size=51200).astype(np.float64).reshape(-1, 5)
    for _ in range(3):
        abi = outer.value("I")
        count = outer.value("Q")
        assert count == 32
        outer.at += abi * count
    outer.finish()
    for key, width in [("obs", 41), ("prevobs", 41), ("prevaction", 4), ("action", 4)]:
        d[key] = d[key].reshape(-1, width)
    for values in d.values():
        assert np.isfinite(values).all()
    assert np.all(d["value"] == 0)
    assert np.array_equal(d["done"], (state["age"] + 1 == state["length"]).astype(float))
    assert np.array_equal(d["history"], (state["age"] != 0).astype(float))
    assert np.max(np.abs(d["reward"] + .05 * costs.sum(axis=1))) < 2e-5
    assert np.max(np.abs(d["obs"][:, 34] - np.maximum(0, 1-state["age"]/40))) < 1e-6
    assert np.all(d["obs"][state["hold"], 3:6] == 0)
    reset = state["age"] == 0
    assert np.all(d["prevobs"][reset] == 0) and np.all(d["prevaction"][reset] == 0)
    table_path = path.with_name(path.name.replace("batch-", "rows-").replace(".bin", ".csv"))
    table = rows(table_path)
    assert len(table) == 10240
    csv_return = np.asarray([row["return"] for row in table])
    error = np.max(np.abs(csv_return - raw_returns(d["reward"], d["done"])))
    assert error < 2e-5
    for key, actual in [("done", d["done"]), ("history", d["history"]), ("reward", d["reward"]),
                        ("logp", d["logp"]), ("cell", state["cell"]), ("local_age", state["age"]), ("segment_length", state["length"])]:
        assert np.max(np.abs(np.asarray([row[key] for row in table]) - actual)) < 1e-8
    d["returns"] = csv_return
    d["return_error"] = float(error)
    d["speed"] = np.asarray([row["speed_m_s"] for row in table])
    return d


def read_baseline(path, protocol):
    reader = envelope(path)
    assert reader.value("I") == 0x31434842 and reader.value("I") == 1 and reader.value("Q") == protocol
    means, count = reader.array("<f8", 800), reader.array("<i4", 800)
    scales = np.asarray([reader.value("d"), reader.value("d")])
    reader.finish()
    assert np.isfinite(means).all() and np.all(np.isfinite(scales) & (scales > 0))
    assert np.array_equal(count, np.r_[np.full(640, 4), np.full(160, 16)])
    return means, scales


def signal(data, means, scales, state):
    residual = data["returns"] - means[state["cell"]]
    held = ~state["fit"]
    energy = [np.sum((residual[held & (state["mode"] == m)]/scales[m])**2) for m in [0, 1]]
    by_duration = {}
    for length in [20, 60]:
        x = residual[held & state["hold"] & (state["length"] == length)]
        assert len(x) == (640 if length == 20 else 1920)
        by_duration[str(length)] = {"bias": float(x.mean()), "mse": float(np.mean(x*x)),
                                     "offset_energy_fraction": float(x.mean()**2/np.mean(x*x))}
    return {"hold_share": float(energy[1]/sum(energy)),
            "heldout_hold_quiet": int(np.sum(held & state["hold"] & (data["speed"] <= .0005))),
            "by_duration": by_duration}


def read_adam(reader):
    config = [reader.value("f") for _ in range(5)]
    expected = np.asarray([.000125, .9, .999, .00001, .5], dtype=np.float32)
    assert np.array_equal(np.asarray(config, dtype=np.float32), expected)
    age = reader.value("i")
    first, second = reader.array(size=2532), reader.array(size=2532)
    assert np.isfinite(first).all() and np.all(np.isfinite(second) & (second >= 0))
    reader.finish()
    return age


def read_actor(path, protocol, update):
    reader = envelope(path)
    assert reader.value("I") == 0x31434841 and reader.value("I") == 1 and reader.value("Q") == protocol
    assert reader.value("i") == update
    fresh = reader.value("i")
    assert reader.value("I") == 41 and reader.value("I") == 4
    params = reader.array(size=2532).astype(np.float64)
    adam = Reader(reader.array("u1").tobytes())
    reader.finish()
    age = read_adam(adam)
    assert age == 21700 + fresh and 0 <= fresh <= 640 and np.isfinite(params).all()
    return params, {"update": update, "fresh_steps": fresh, "inherited_steps": age,
                    "sha256": HASHES[str(path.resolve())]}


def forward(params, x):
    at = 0
    for layer, (ni, no) in enumerate([(41, 32), (32, 32), (32, 4)]):
        w = params[at:at+ni*no].reshape(no, ni)
        at += ni*no
        b = params[at:at+no]
        at += no
        x = x @ w.T + b
        if layer < 2:
            x = np.tanh(x)
    assert at == 2532
    return x


def old_logp(data, params):
    rho = float(np.float32(.8))
    mu = forward(params, data["obs"]) + rho * data["history"][:, None] * (data["prevaction"] - forward(params, data["prevobs"]))
    var = np.exp(-2) * (1-rho*rho*data["history"])
    lp = -.5*np.sum((data["action"]-mu)**2/var[:,None] + np.log(2*np.pi*var[:,None]), axis=1)
    error = float(np.max(np.abs(lp-data["logp"])))
    assert error < 1e-4
    return error


def hold_assessment(directory):
    trace, summary = rows(directory/"rest"/"controls.csv"), rows(directory/"rest"/"summary.csv")
    native_breach = {int(row["case"]): row for row in rows(directory/"first-hold-breach.csv")}
    assert len(trace) == 7680 and len(summary) == 6 and set(native_breach) == set(range(6))
    details = []
    for case in range(6):
        sequence = [row for row in trace if row["case"] == case]
        assert len(sequence) == 1280 and [row["control"] for row in sequence] == list(range(1,1281))
        anchor56 = sequence[1119]["forward_mm"]
        excursions = [abs(row["forward_mm"]-anchor56) for row in sequence[1119:]]
        peaks = {"speed": max(row["speed_mm_s"] for row in sequence), "anchor": max(row["anchor_error_mm"] for row in sequence),
                 "depth": max(abs(row["depth_mm"]) for row in sequence), "lane": max(row["lane_mm"] for row in sequence),
                 "tilt": max(row["tilt_deg"] for row in sequence), "excursion": max(excursions)}
        limits = {"speed": .5, "anchor": 2, "depth": 1, "lane": 2, "tilt": 10, "excursion": 2}
        gates = {key: value <= limits[key] for key, value in peaks.items()}
        first_time, first_cause = None, "none"
        for row in sequence:
            causes = []
            if row["speed_mm_s"] > .5: causes.append("speed")
            if row["anchor_error_mm"] > 2: causes.append("anchor")
            if abs(row["depth_mm"]) > 1: causes.append("depth")
            if row["lane_mm"] > 2: causes.append("lane")
            if row["tilt_deg"] > 10: causes.append("tilt")
            if row["control"] > 1120 and abs(row["forward_mm"]-anchor56) > 2: causes.append("final8_excursion")
            if causes:
                first_time, first_cause = row["time_s"], "|".join(causes)
                break
        native = native_breach[case]
        assert native["cause"] == first_cause
        if first_time is not None: near(native["time_s"], first_time, 1e-9)
        original = next(row for row in summary if row["case"] == case)
        assert original["passed"] == int(all(gates.values()))
        for key, saved in [("speed","max_speed_mm_s"),("anchor","max_anchor_error_mm"),("depth","max_depth_mm"),
                           ("lane","max_lane_mm"),("tilt","max_tilt_deg"),("excursion","final8_forward_excursion_mm")]:
            near(peaks[key], original[saved], 1e-6)
        details.append({"case":case, "passed":all(gates.values()), "gates":gates, "peaks":peaks,
                        "first_breach_s":first_time, "first_breach_cause":first_cause,
                        "final8_mean_speed_mm_s":float(np.mean([row["speed_mm_s"] for row in sequence[1120:]])),
                        "final8_forward_delta_mm":sequence[-1]["forward_mm"]-anchor56,
                        "net_forward_mm":sequence[-1]["forward_mm"],
                        "final_speed_mm_s":sequence[-1]["speed_mm_s"]})
    return {"passes":sum(row["passed"] for row in details), "cases":details, "controls":len(trace)}


def swim_gate(row, short=False):
    return (7.2 <= row["steady_mm_s" if short else "mean_mm_s"] <= 8.8 and row["RMSE_mm_s"] <= 2.5
            and row["lateral_mm"] <= 2 and row["depth_mm"] <= 1 and row["tilt_deg"] <= 10
            and row["path_deg"] <= 5 and row["work_J_per_m"] <= .001)


def assessment(directory, original=None):
    short = rows(directory/"frozen-evaluation.csv")
    assert len(short) == 4 and {row["pose"] for row in short} == {0,1,2,3}
    short_cases = []
    for row in short:
        assert row["passed"] == int(swim_gate(row,True)), row
        gates = {"speed":7.2<=row["steady_mm_s"]<=8.8,"RMSE":row["RMSE_mm_s"]<=2.5,
                 "lane":row["lateral_mm"]<=2,"depth":row["depth_mm"]<=1,"tilt":row["tilt_deg"]<=10,
                 "path":row["path_deg"]<=5,"work":row["work_J_per_m"]<=.001}
        short_cases.append({"pose":int(row["pose"]),"passed":bool(row["passed"]),"gates":gates,
                            "failed_gates":[key for key,value in gates.items() if not value],"native_metrics":row})
    short_trace = rows(directory/"frozen-controls.csv")
    assert len(short_trace) == 640
    for pose in range(4):
        assert [row["control"] for row in short_trace if row["pose"] == pose] == list(range(1,161))
    long = rows(directory/"long"/"sustained-windows.csv")
    assert len(long) == 30
    assert {(int(row["pose"]),int(row["end_s"])) for row in long} == {(p,t) for p in [0,3] for t in range(4,61,4)}
    passed = set()
    for row in long:
        yes = swim_gate(row)
        assert row["passed"] == int(yes)
        if yes: passed.add((int(row["pose"]),int(row["end_s"])))
    long_trace = rows(directory/"long"/"sustained-controls.csv")
    assert len(long_trace) == 2400
    for pose in [0,3]:
        assert [row["control"] for row in long_trace if row["pose"] == pose] == list(range(1,1201))
        sequence = [row for row in long_trace if row["pose"] == pose]
        initial_yaw = (pose-1.5)*.05
        reference_x, reference_z = 8*np.cos(initial_yaw), -8*np.sin(initial_yaw)
        for saved in [row for row in long if row["pose"] == pose]:
            end = int(saved["end_s"]*20)
            span = sequence[40:end] if end == 80 else sequence[end-80:end]
            rmse = np.sqrt(np.mean([(row["vx_mm_s"]-reference_x)**2+row["vy_mm_s"]**2+(row["vz_mm_s"]-reference_z)**2 for row in span]))
            near(rmse,saved["RMSE_mm_s"],1e-5)
            for col, metric in [("lane_mm","lateral_mm"),("signed_depth_mm","depth_mm"),("tilt_deg","tilt_deg")]:
                near(max(abs(row[col]) for row in sequence[:end]), saved[metric],1e-5)
    hold = hold_assessment(directory)
    short_pass = sum(row["passed"] for row in short)
    retained = short_pass == 4 and (original is None or original <= passed)
    return {"short_passes":int(short_pass), "short_cases":short_cases,"long_passes":len(passed), "long_pass_keys":sorted(passed),
            "retained":retained, "hold":hold, "controls":640+2400+hold["controls"]}, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("build/ppo/hold-credit-336to344-2026-11-22-v1"))
    args = parser.parse_args()
    root = args.run.resolve()
    report = {"scope":"Read-only independent saved-artifact verification; zero physical or optimizer steps",
              "limitations":["Short-swim RMSE uses saved native summary; gate arithmetic is independently checked",
                             "Frozen source and protected-file preservation manifests are coordinator checks"], "passed":False}
    try:
        cal_result = js(root/"calibration-result.json")
        assert cal_result["closed"]
        assert cal_result["controls"]==20960 and cal_result["substeps"]==209600
        assert cal_result["actor_steps"]==cal_result["critic_steps"]==cal_result["critic_forward_calls"]==0
        assert not cal_result["runtime_adopted"]
        training_path = root/"training-result.json"
        training = js(training_path) if training_path.exists() else None
        assert training is None or training["closed"]
        assert training is not None or cal_result["gate"] or cal_result["fault"], "Successful calibration requires closed training result"
        protocol_reader = envelope(root/"protocol.bin")
        protocol_id = fnv(protocol_reader.data)
        report["protocol_id"] = protocol_id
        baseline_summary, original = assessment(root/"baseline")
        assert baseline_summary["short_passes"] == 4 and len(original) == 28
        assert original == {(p,t) for p in [0,3] for t in range(8,61,4)}
        report["baseline"] = baseline_summary
        state = schedule()
        calibration = read_batch(root/"calibration"/"batch-336.bin",state)
        means, scales = read_baseline(root/"fixed-baseline.bin",protocol_id)
        recomputed = np.zeros(800)
        count = np.zeros(800,dtype=int)
        for i in np.flatnonzero(state["fit"]):
            recomputed[state["cell"][i]] += calibration["returns"][i]
            count[state["cell"][i]] += 1
        recomputed /= count
        assert np.max(np.abs(recomputed-means)) < 1e-10
        error = calibration["returns"]-means[state["cell"]]
        fit_rms = np.asarray([np.sqrt(np.mean(error[state["fit"] & (state["mode"]==m)]**2)) for m in [0,1]])
        assert np.max(np.abs(fit_rms-scales)) < 1e-10
        cal_signal = signal(calibration,means,scales,state)
        native_cal = js(root/"calibration-report.json")
        for key in ["hold_share","heldout_hold_quiet"]: near(cal_signal[key],native_cal[key],1e-9)
        for duration in [20,60]: near(cal_signal["by_duration"][str(duration)]["offset_energy_fraction"],native_cal[f"duration{duration}_offset_energy_fraction"],1e-9)
        report["calibration"] = {"independent_signal":cal_signal,"fit_rms":scales.tolist(),"return_recurrence_max_error":calibration["return_error"]}
        expected_cal_gate = (cal_signal["heldout_hold_quiet"]>=256 and .1<=cal_signal["hold_share"]<=.9
                             and all(v["offset_energy_fraction"]<=.25 for v in cal_signal["by_duration"].values()))
        report["calibration_signal_gate"] = expected_cal_gate
        updates = []
        actual_controls = 10720+10240
        actor_meta = []
        if training is not None:
            native_updates = [json.loads(line) for line in read(root/"updates.jsonl").decode().splitlines() if line]
            directory = root/"training-started"
            for file in sorted(directory.glob("credit-actor-update-*.bin")):
                update = int(file.stem.rsplit("-",1)[1])
                _, meta = read_actor(file,protocol_id,update)
                actor_meta.append(meta)
            assert [row["update"] for row in actor_meta] == list(range(336,337+training["PPO_updates"]))
            for file in sorted(directory.glob("batch-*.bin")):
                update = int(file.stem.rsplit("-",1)[1])
                data = read_batch(file,state)
                params, _ = read_actor(directory/f"credit-actor-update-{update}.bin",protocol_id,update)
                logp_error = old_logp(data,params)
                current_signal = signal(data,means,scales,state)
                actual_controls += 10240
                item = {"collection_update":update,"oldlogp_max_error":logp_error,"return_recurrence_max_error":data["return_error"],"signal":current_signal}
                if (directory/f"assessment-{update+1}").exists():
                    evaluated, _ = assessment(directory/f"assessment-{update+1}",original)
                    actual_controls += evaluated["controls"]
                    native = next(row for row in native_updates if row["update"]==update+1)
                    assert native["short"]==evaluated["short_passes"] and native["long"]==evaluated["long_passes"]
                    assert native["holds"]==evaluated["hold"]["passes"] and native["retained"]==evaluated["retained"]
                    assert native["controls"]==actual_controls
                    item["assessment"] = evaluated
                updates.append(item)
            assert len(native_updates)==training["PPO_updates"]
            assert len([v for v in updates if "assessment" in v])==training["PPO_updates"]
            assert not training["fault"], "Physical/data fault needs a separate partial-attempt decoder"
            assert actual_controls==training["controls"] and actual_controls*10==training["substeps"]
            assert training["actor_steps"]==actor_meta[-1]["fresh_steps"]
            assert all(40<=b["fresh_steps"]-a["fresh_steps"]<=80 for a,b in zip(actor_meta,actor_meta[1:]))
            if training["gate"]=="regressed_swimming": assert not updates[-1]["assessment"]["retained"]
            assert all(v["assessment"]["retained"] for v in updates[:-1] if "assessment" in v)
            assert training["critic_steps"]==training["critic_forward_calls"]==0 and not training["runtime_adopted"]
            assert training["PPO_updates"]<=8 and training["actor_steps"]<=640
            assert training["active_seconds"]+training["prior_active_seconds"]<=120
        else:
            assert cal_result["controls"]==actual_controls and cal_result["substeps"]==actual_controls*10
        assert actual_controls<=188640
        for path, expected in list(HASHES.items()): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
        report.update({"passed":True,"updates":updates,"actor_checkpoints":actor_meta,"counted_controls":actual_controls,
                       "counted_substeps":actual_controls*10,"input_sha256":HASHES})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["input_sha256"] = HASHES
        (root/"independent-verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        raise
    (root/"independent-verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"counted_controls":report["counted_controls"],"assessment_count":len(report["updates"]),"verified_input_files":len(HASHES)}))


if __name__=="__main__":
    main()
