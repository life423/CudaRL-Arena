"""Export a privately staged V3 preview only from qualified, verified V2 evidence.

This offline utility parses saved data. It never steps physics, trains, selects
the active GUI package or overwrites an existing package.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

from export_hold_preview import Reader, fnv, CANONICAL

ROOT = Path(__file__).resolve().parents[1]
HOLD_SHA = "C54F73B57AB4280D99802A7253C13004F95C754E39E0ACF8F6833377BBB760C3"
CURRENT_CONTRACTS = ["src/nn/motor/current_model_v2.h", "src/nn/motor/current_motor_v2.h",
                     "src/nn/motor/current_task_v2.h", "tools/current_swim_math.h"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def key(path):
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def finite(values):
    require(all(math.isfinite(x) for x in values), "Nonfinite saved parameters")


def parent_backbone(path):
    require(sha(path) == CANONICAL, "The canonical swimming parent changed")
    r = Reader(path)
    require([r.value("I"), r.value("I")] == [0x31525050, 4], "Wrong swimming checkpoint schema")
    r.value("I")
    r.value("Q")
    require(r.value("i") == 336, "Wrong swimming parent update")
    require([r.value("I") for _ in range(7)] == [6, 3, 6, 5, 40, 4, 1], "Wrong parent contract")
    r.value("f")
    require(r.value("I") == 2, "Wrong swimming observation schema")
    old = r.array("f")
    require(len(old) == 2500, "Wrong parent actor dimensions")
    require(len(r.array("f")) == 2401 and len(r.array("f")) == 4, "Wrong parent auxiliary dimensions")
    for count in (2500, 2401):
        for _ in range(5):
            r.value("f")
        r.value("i")
        require(len(r.array("f")) == count and len(r.array("f")) == count, "Wrong parent optimizer dimensions")
    require(r.at == len(r.data), "Unexpected parent data")
    migrated = []
    for row in range(32):
        migrated.extend(old[row*40:(row+1)*40])
        migrated.append(0.)
    migrated.extend(old[1280:])
    return tuple(migrated)


def holding_parent(path, backbone, parent_checksum):
    require(sha(path) == HOLD_SHA, "The qualified holding parent changed")
    r = Reader(path)
    require([r.value("I"), r.value("I")] == [0x314D4148, 1], "Wrong holding checkpoint schema")
    require(r.value("Q") != 0 and r.value("Q") == parent_checksum, "Holding/swimming parent mismatch")
    require([r.value("I") for _ in range(7)] == [6, 6, 3, 4, 41, 4, 336], "Wrong holding contract")
    require([r.value("i"), r.value("i")] == [7, 112], "Wrong holding learning history")
    require(r.array("f") == backbone, "Holding backbone differs from the canonical migration")
    head = r.array("f")
    require(len(head) == 2, "Wrong holding parameter count")
    gain, sigma = r.value("f"), r.value("f")
    r.array("B")
    require(r.at == len(r.data), "Unexpected holding data")
    finite(head + (gain, sigma))
    logistic = lambda x: 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))
    require(abs(gain-logistic(head[0])) <= 2e-7 and abs(sigma-(.05+7.95*logistic(head[1]))) <= 2e-6,
            "Holding amplitude mapping changed")
    return head, gain, sigma


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"build/ppo/current-swim-v2-2026-11-22-v1")
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"build/current-preview-v3-2026-11-22/current-flow-v2.bin")
    args = parser.parse_args()
    run = args.run.resolve()
    trained = (run/"training-result.json").is_file()
    result_path = run/("training-result.json" if trained else "baseline-result.json")
    result = json.loads(result_path.read_text())
    require(result.get("schema") == 2 and result.get("closed") is True and result.get("qualified") is True,
            "Refusing export: current V2 study is not closed and qualified")
    require(not result.get("fault") and result.get("runtime_adopted") is False and result.get("retries") == 0,
            "Refusing export: fault, adoption or retry metadata differs")
    phase = "training" if trained else "baseline"
    proof_path = (args.verification or run/f"independent-{phase}-verification.json").resolve()
    proof = json.loads(proof_path.read_text())
    require(proof.get("passed") is True and proof.get("qualified") is True and proof.get("phase") == phase,
            "Refusing export: full independent qualification is missing")
    inputs = {key(p): str(digest).upper() for p, digest in proof.get("input_sha256", {}).items()}

    def bound(path):
        path = path.resolve()
        require(inputs.get(key(path)) == sha(path), f"Verified input changed or is unbound: {path.name}")
        return path

    bound(result_path)
    parent_path = bound(run/"parent-checkpoint.bin")
    hold_path = bound(run/"holding-parent.bin")
    protocol_path = bound(run/"protocol.bin")
    protocol = fnv(Reader(protocol_path).data)
    parent_checksum, hold_checksum = fnv(Reader(parent_path).data), fnv(Reader(hold_path).data)
    backbone = parent_backbone(parent_path)
    hold_head, hold_gain, hold_sigma = holding_parent(hold_path, backbone, parent_checksum)
    updates, steps = result["PPO_updates"], result["actor_steps"]
    require(isinstance(updates, int) and 0 <= updates <= 32 and isinstance(steps, int) and 0 <= steps <= 512,
            "Invalid native optimizer counters")
    if trained:
        require(updates > 0 and steps > 0 and result.get("baseline_only") is False,
                "Training result does not identify a learned current policy")
        policy_path = run/"training-started"/f"current-policy-update-{updates}.bin"
        selection = Reader(bound(run/"selected-size.bin"))
        require([selection.value("I"), selection.value("I"), selection.value("Q")] == [0x325A5343, 2, protocol],
                "Wrong selected-size identity")
        selected_worlds = selection.value("i")
        selection.value("Q"); selection.value("Q")
        require(selection.at == len(selection.data), "Unexpected selected-size data")
        selected_binding = fnv(selection.data)
        require(selected_worlds in (2048, 4096, 8192, 16384) and result.get("selected_worlds") == selected_worlds,
                "Selected training world count changed")
    else:
        require(updates == steps == 0 and result.get("baseline_only") is True,
                "Baseline export must not imply new PPO learning")
        policy_path = run/"initial-policy/current-policy-update-0.bin"
        selected_worlds, selected_binding = 2048, 0
    bound(policy_path)
    r = Reader(policy_path)
    require([r.value("I"), r.value("I")] == [0x32575343, 2], "Wrong current checkpoint schema")
    require([r.value("Q") for _ in range(4)] == [protocol, parent_checksum, hold_checksum, selected_binding],
            "Current checkpoint source identities changed")
    require([r.value("I") for _ in range(10)] == [6, 2, 3, 5, 44, 41, 4, 16, 336, 7], "Wrong current model/task/action dimensions")
    require([r.value("i") for _ in range(3)] == [selected_worlds, updates, steps], "Current checkpoint counters differ")
    require(r.array("f") == backbone and r.array("f") == hold_head and r.value("f") == hold_gain,
            "The preserved swimming/holding policy changed")
    means = r.array("f")
    require(len(means) == 16 and r.value("f") == .5 and r.value("f") == .5, "Wrong current residual policy")
    optimizer_bytes = bytes(r.array("B"))
    require(r.at == len(r.data), "Unexpected current checkpoint data")
    finite(means)
    offset = 0
    config = struct.unpack_from("<5f", optimizer_bytes, offset); offset += 20
    require(config == tuple(f32(x) for x in (.01, .9, .999, .00001, .5)), "Wrong current Adam configuration")
    age = struct.unpack_from("<i", optimizer_bytes, offset)[0]; offset += 4
    moments = []
    for _ in range(2):
        count = struct.unpack_from("<Q", optimizer_bytes, offset)[0]; offset += 8
        require(count == 16, "Wrong current Adam moment dimensions")
        moments.append(struct.unpack_from("<16f", optimizer_bytes, offset)); offset += 64
    require(offset == len(optimizer_bytes) and age == steps, "Current Adam age/payload changed")
    finite(moments[0] + moments[1]); require(all(x >= 0 for x in moments[1]), "Negative current Adam second moment")
    if not trained:
        require(all(x == 0 for x in means + moments[0] + moments[1]), "Baseline current head is not its initial state")
    require(result.get("final_current_passes") == 16 and result.get("withheld_passes") == 16 and result.get("endurance_passes") == 2,
            "Required current case suites did not all pass")
    current_assessment = run/"training-started"/f"assessment-{updates}" if trained else run/"baseline-current"
    for directory, count in ((current_assessment, 16), (run/"reserved-withheld", 16), (run/"reserved-endurance", 2)):
        summary = rows(bound(directory/"summary.csv"))
        require(len(summary) == count and all(int(row["passed"]) == 1 for row in summary), "Current acceptance summary changed")
        if count == 2:
            for row, (pose, flow_x, flow_z) in zip(summary, ((0, -2., 0.), (3, 0., 2.))):
                require(int(row["pose"]) == pose and int(row["controls"]) == 1280 and int(row["substeps"]) == 12800,
                        "Current preview presets are not the verified 64-second cases")
                require(abs(float(row["current_x_mm_s"])-flow_x) < 1e-5 and abs(float(row["current_z_mm_s"])-flow_z) < 1e-5,
                        "Current preview preset direction/strength changed")
    calm = run/"final-calm"
    holds = rows(bound(calm/"holds.csv")); short = rows(bound(calm/"frozen-evaluation.csv"))
    long = rows(bound(calm/"long/sustained-windows.csv"))
    require(len(holds) == 6 and all(int(row["passed"]) for row in holds), "Calm holding retention failed")
    require(len(short) == 4 and all(int(row["passed"]) for row in short), "Calm short-swim retention failed")
    require(len(long) == 30 and sum(int(row["passed"]) for row in long) == 28, "Calm long-window retention changed")
    prior_long = rows(bound(run/"baseline-calm/long/sustained-windows.csv"))
    def passing_keys(table):
        return {(int(float(row["pose"])), int(float(row["end_s"]))) for row in table if int(row["passed"])}
    require(passing_keys(long) == passing_keys(prior_long), "The preserved calm passing-window keys changed")
    legacy = [line.strip() for line in (ROOT/"src/nn/motor/controller_contract.txt").read_text().splitlines() if line.strip()]
    require(len(legacy) == 12, "Legacy source contract count changed")
    contracts = []
    for relative in legacy + CURRENT_CONTRACTS:
        source = bound(ROOT/relative)
        contracts.append(hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    payload = bytearray()

    def put(kind, *values):
        payload.extend(struct.pack("<"+kind, *values))

    def array(kind, values):
        put("Q", len(values)); put(str(len(values))+kind, *values)

    put("12I", 0x31564C48, 3, 6, 6, 3, 4, 41, 4, 336, 2, 7, 1)
    put("Q", protocol); array("B", CANONICAL.encode("ascii")); put("I", len(contracts))
    for digest in contracts:
        array("B", digest.encode("ascii"))
    array("f", backbone); array("f", hold_head); put("2f", hold_gain, hold_sigma)
    put("4I", 2, 2, 5, updates); array("f", means); put("2f", .5, .5)
    array("B", sha(policy_path).encode("ascii"))
    package = struct.pack("<QQQ", 0x31504B434C524443, len(payload), fnv(payload)) + payload
    output = args.output.resolve(); sidecar = output.with_suffix(".json")
    require(not output.exists() and not sidecar.exists(), "Choose a new private package identity; no overwrite is allowed")
    metadata = {"schema":1,"scope":"Privately staged frozen research preview; no runtime training or production adoption",
        "package_version":3,"policy_method":"current_flow_residual","qualified":True,"current_qualified":True,
        "holding_updates":7,"current_policy_updates":updates,"current_Adam_steps":steps,"baseline_only":not trained,
        "checkpoint_capacity_worlds":selected_worlds,"selected_training_worlds":result.get("selected_worlds",0),
        "package_sha256":hashlib.sha256(package).hexdigest().upper(),"source_checkpoint":str(policy_path),
        "source_checkpoint_sha256":sha(policy_path),"source_result":str(result_path),"source_result_sha256":sha(result_path),
        "independent_verification":str(proof_path),"independent_verification_sha256":sha(proof_path),
        "parent_sha256":CANONICAL,"holding_parent_sha256":HOLD_SHA,"source_protocol_checksum":f"{protocol:016X}",
        "current_model_version":2,"current_task_version":2,"current_observation_version":5,"current_parameter_count":16,
        "means":means,"deterministic_coefficients":[.5*math.tanh(x) for x in means],"current_sigma":.5,"coefficient_limit":.5,
        "holding_parameters":hold_head,"deterministic_amplitude":hold_gain,"holding_sigma":hold_sigma,
        "current_cases":16,"withheld_cases":16,"endurance_cases":2,"endurance_seconds":64,
        "calm_holds":6,"calm_short_swims":4,"calm_long_windows":28,"calm_long_windows_total":30,
        "presets":[{"id":0,"label":"Headwind · 2 mm/s","pose":0,"water_current_m_s":[-.002,0,0]},
                   {"id":1,"label":"Cross-current · 2 mm/s","pose":3,"water_current_m_s":[0,0,.002]}],
        "backbone_frozen":True,"action_noise":False,"runtime_training":False,"runtime_adopted":False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(package)
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2); stream.write("\n")
    print(json.dumps({"staged":str(output),"qualified":True,"current_PPO_updates":updates,"current_Adam_steps":steps,
                      "runtime_adopted":False,"physics_controls":0,"optimizer_steps":0}))


if __name__ == "__main__":
    main()
