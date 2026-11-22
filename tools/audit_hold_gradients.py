"""Unapplied saved-data policy-gradient attribution; no simulator or optimizer.

Reconstructs the full-batch unclipped loss gradient at the collection policy.
This is not the sequence of clipped/minibatch/Adam updates actually performed.
Counterfactual baselines use only the original independent calibration fit worlds.
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
import time

import numpy as np


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def value(self, fmt):
        value = struct.unpack_from("<" + fmt, self.data, self.at)[0]
        self.at += struct.calcsize("<" + fmt)
        return value

    def array(self, dtype="<f4", count=None):
        size = self.value("Q")
        if count is not None:
            assert size == count, (size, count)
        dtype = np.dtype(dtype)
        assert self.at + size * dtype.itemsize <= len(self.data)
        array = np.frombuffer(self.data, dtype=dtype, count=size, offset=self.at)
        self.at += size * dtype.itemsize
        return array

    def finish(self):
        assert self.at == len(self.data), (self.at, len(self.data))


INPUTS = {}


def envelope(path):
    data = path.read_bytes()
    magic, size, expected = struct.unpack_from("<QQQ", data)
    assert magic == 0x31504B434C524443 and size == len(data) - 24
    payload = data[24:]
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    assert checksum == expected, path
    INPUTS[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return Reader(payload)


def batch(path):
    outer = envelope(path)
    assert outer.value("I") == 1
    inner = Reader(outer.array("u1").tobytes())
    assert [inner.value("i") for _ in range(4)] == [1, 10240, 41, 4]
    names = ["obs", "prevobs", "prevaction", "history", "action", "logp", "value", "reward", "done"]
    result = {name: inner.array().astype(np.float64) for name in names}
    inner.finish()
    for name, width in [("obs", 41), ("prevobs", 41), ("prevaction", 4), ("action", 4)]:
        result[name] = result[name].reshape(-1, width)
    assert all(np.isfinite(value).all() for value in result.values())
    assert np.all(result["value"] == 0)
    outer.array(count=10240 * 5)  # Independently checksum, but no physics ABI decoding.
    for _ in range(3):
        abi_size = outer.value("I")
        count = outer.value("Q")
        assert count == 32
        outer.at += count * abi_size
        assert outer.at <= len(outer.data)
    outer.finish()
    result["returns"] = returns(result)
    recorded_csv = path.with_name(path.name.replace("batch-", "rows-").replace(".bin", ".csv"))
    csv_bytes = recorded_csv.read_bytes()
    INPUTS[str(recorded_csv.resolve())] = hashlib.sha256(csv_bytes).hexdigest()
    native = np.asarray([float(row["return"]) for row in csv.DictReader(csv_bytes.decode("utf-8").splitlines())])
    assert native.shape == (10240,)
    result["return_recurrence_max_error"] = float(np.max(np.abs(native - result["returns"])))
    assert result["return_recurrence_max_error"] < 2e-5
    # Use the recorded native signal after checking its terminal-return recurrence.
    result["returns"] = native
    return result


def returns(data):
    rewards = data["reward"].reshape(320, 32)
    done = data["done"].reshape(320, 32)
    out = np.zeros((320, 32), dtype=np.float64)
    carry = np.zeros(32)
    gamma = float(np.float32(.99))
    for slot in range(319, -1, -1):
        carry = rewards[slot] + gamma * (1 - done[slot]) * carry
        out[slot] = carry
    return out.ravel()


def actor(path, expected_arm, expected_update):
    reader = envelope(path)
    assert reader.value("I") == 0x50505241 and reader.value("I") == 1
    reader.value("Q")
    assert reader.value("i") == expected_arm
    assert reader.value("i") == expected_update
    fresh = reader.value("i")
    params = reader.array(count=2532).astype(np.float64)
    reader.array("u1")
    reader.finish()
    return params, fresh


def pooled(path):
    reader = envelope(path)
    assert reader.value("I") == 0x50464252
    reader.value("Q")
    mean = reader.array("<f8", 1280)
    count = reader.array("<i4", 1280)
    assert np.all(count == 8)
    rms = []
    for _ in range(2):
        rms.append(reader.value("d"))
        assert reader.value("i") == 5120
    reader.finish()
    return mean, np.asarray(rms)


def layers(params):
    out, at = [], 0
    for width, height in [(41, 32), (32, 32), (32, 4)]:
        weights = params[at:at + width * height].reshape(height, width)
        at += width * height
        bias = params[at:at + height]
        at += height
        out.append((weights, bias))
    assert at == len(params)
    return out


def forward(params, values):
    activations = [values]
    for index, (weights, bias) in enumerate(layers(params)):
        values = values @ weights.T + bias
        if index != 2:
            values = np.tanh(values)
        activations.append(values)
    return values, activations


def backward(params, activations, gradient):
    output = []
    for index in range(2, -1, -1):
        weights, _ = layers(params)[index]
        if index != 2:
            gradient = gradient * (1 - activations[index + 1] ** 2)
        dw = gradient.T @ activations[index]
        db = gradient.sum(axis=0)
        output.insert(0, np.r_[dw.ravel(), db])
        gradient = gradient @ weights
    return np.concatenate(output)


def metrics(a, b):
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "hold_norm": float(an), "swim_norm": float(bn),
        "hold_swim_cosine": float(a @ b / (an * bn)),
        "hold_to_swim_norm_ratio": float(an / bn),
        "sum_norm": float(np.linalg.norm(a + b)),
        "hold_loss_derivative_along_negative_sum": float(-a @ (a + b)),
        "swim_loss_derivative_along_negative_sum": float(-b @ (a + b)),
    }


def summary(values):
    return {"mean": float(np.mean(values)), "rms": float(np.sqrt(np.mean(values ** 2))),
            "positive_fraction": float(np.mean(values > 0)), "min": float(values.min()),
            "max": float(values.max())}


def schedule(arm):
    index = np.arange(10240)
    world, slot = index % 32, index // 32
    mode, pose = (world // 2) % 2, world % 2
    if arm:
        age = np.where(slot % 80 < 20, slot % 80, slot % 80 - 20)
        length = np.where(slot % 80 < 20, 20, 60)
    else:
        age, length = slot % 20, np.full(10240, 20)
    age, length = np.where(mode == 1, age, slot), np.where(mode == 1, length, 320)
    return {"hold": mode == 1, "fit": world % 8 < 4, "mode": mode, "pose": pose,
            "slot": slot, "age": age, "length": length, "cell": (mode * 2 + pose) * 320 + slot}


def conditional(calibration, state, by_duration):
    keys = list(zip(state["mode"].tolist(), state["pose"].tolist(),
                    state["length"].tolist(), state["age"].tolist())) if by_duration else state["cell"].tolist()
    fits = {}
    for index, key in enumerate(keys):
        if state["fit"][index]:
            fits.setdefault(key, []).append(calibration["returns"][index])
    fitted = {key: np.mean(value) for key, value in fits.items()}
    mean = np.asarray([fitted[key] for key in keys])
    rms = np.asarray([np.sqrt(np.mean((calibration["returns"] - mean)[state["fit"] & (state["mode"] == mode)] ** 2)) for mode in [0, 1]])
    assert np.all(rms > 0)
    return mean, rms


def conditional_mean(params, data):
    means, cache = forward(params, data["obs"])
    previous_means, previous_cache = forward(params, data["prevobs"])
    rho = float(np.float32(.8))
    coefficient = rho * data["history"]
    mean = means + coefficient[:, None] * (data["prevaction"] - previous_means)
    variance = np.exp(-2) * (1 - rho ** 2 * data["history"])
    logp = -.5 * np.sum((data["action"] - mean) ** 2 / variance[:, None] + np.log(2 * np.pi * variance[:, None]), axis=1)
    return mean, variance, coefficient, logp, cache, previous_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("build/ppo/rest-hold-duration-pair-2026-11-21-v1"))
    parser.add_argument("--output", type=Path, default=Path("build/audit-hold-saved-2026-11-22/gradients"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    means, scales = pooled(args.source / "pooled-fixed.bin")
    result = {"scope": "Saved data and unapplied fullbatch gradients only; zero physics and zero optimizer steps",
              "objective": "Negative mean advantage times collection-policy conditional AR latent log probability",
              "limitations": ["Unclipped collection-policy fullbatch gradient is not actual minibatch/Adam update",
                              "Counterfactual fixed baselines are fitted on original independent calibration fit worlds only",
                              "Gradient conflicts and parameter deltas do not establish a causal physical fix"], "arms": {}}
    for arm, folder in enumerate(["A20", "B20_60"]):
        state = schedule(arm)
        calibration = batch(args.source / ("calibration-B" if arm else "calibration-A") / "batch-336.bin")
        slot_mean, _ = conditional(calibration, state, False)
        duration_mean, duration_scale = conditional(calibration, state, True)
        arm_result = []
        for update in [336, 337, 338]:
            directory = args.source / "phase-B-started" / folder
            params, fresh = actor(directory / f"actor-only-update-{update}.bin", arm, update)
            next_params, _ = actor(directory / f"actor-only-update-{update+1}.bin", arm, update + 1)
            data = batch(directory / f"batch-{update}.bin")
            mean, variance, coefficient, logp, cache, previous_cache = conditional_mean(params, data)
            error = float(np.max(np.abs(logp - data["logp"])))
            assert error < 1e-4, (folder, update, error)
            assert np.array_equal(data["history"], (state["age"] != 0).astype(float))
            assert np.array_equal(data["done"], (state["age"] == state["length"] - 1).astype(float))
            score = (data["action"] - mean) / variance[:, None]
            records = {}
            schemes = {
                "actual_pooled": (means[state["cell"]], scales),
                "arm_slot_fixed_scale": (slot_mean, scales),
                "duration_conditioned_fixed_scale": (duration_mean, scales),
                "duration_conditioned_task_rms": (duration_mean, duration_scale),
            }
            gradients = {}
            for scheme, (baseline, scale) in schemes.items():
                advantage = (data["returns"] - baseline) / scale[state["mode"]]
                group = []
                for mask in [state["hold"], ~state["hold"]]:
                    current_grad = -advantage[:, None] * score * mask[:, None] / 10240
                    previous_grad = -coefficient[:, None] * current_grad
                    group.append(backward(params, cache, current_grad) + backward(params, previous_cache, previous_grad))
                gh, gs = group
                gradients[scheme] = gh + gs
                values = metrics(gh, gs)
                delta = next_params - params
                values.update({"hold_advantage": summary(advantage[state["hold"]]),
                               "swim_advantage": summary(advantage[~state["hold"]]),
                               "hold_loss_derivative_along_actual_parameter_delta": float(gh @ delta),
                               "swim_loss_derivative_along_actual_parameter_delta": float(gs @ delta),
                               "negative_gradient_cosine_actual_parameter_delta": float(-(gh+gs) @ delta / (np.linalg.norm(gh+gs)*np.linalg.norm(delta))),
                               "hold_new_column40_norm": float(np.linalg.norm(gh[:1312].reshape(32,41)[:,40])),
                               "swim_new_column40_norm": float(np.linalg.norm(gs[:1312].reshape(32,41)[:,40]))})
                records[scheme] = values
                np.savez_compressed(args.output / f"{folder}-{update}-{scheme}.npz", hold_gradient=gh, swim_gradient=gs,
                                    advantage=advantage, parameter_delta=delta)
            actual = gradients["actual_pooled"]
            for scheme in records:
                records[scheme]["total_gradient_cosine_actual_pooled"] = float(gradients[scheme] @ actual / (np.linalg.norm(gradients[scheme]) * np.linalg.norm(actual)))
            # Directional finite difference against the same fixed-data objective;
            # no optimizer and no hypothetical dynamics.
            direction = np.sin(np.arange(2532) + .123)
            direction /= np.linalg.norm(direction)
            advantage = (data["returns"] - means[state["cell"]]) / scales[state["mode"]]
            epsilon = 1e-5
            plus = conditional_mean(params + epsilon * direction, data)[3]
            minus = conditional_mean(params - epsilon * direction, data)[3]
            numeric = float(-np.mean(advantage * (plus - minus)) / (2 * epsilon))
            analytic = float(actual @ direction)
            difference = abs(numeric - analytic)
            assert difference < 1e-7 * max(1, abs(numeric), abs(analytic)), (numeric, analytic)
            arm_result.append({"collection_update": update, "resulting_update": update + 1,
                               "fresh_actor_steps_before": fresh, "logp_max_error": error,
                               "native_return_recurrence_max_error": data["return_recurrence_max_error"],
                               "finite_difference": {"analytic": analytic, "numeric": numeric, "absolute_error": difference},
                               "duration_fit_rms": duration_scale.tolist(), "schemes": records})
        result["arms"][folder] = arm_result
    result["input_sha256"] = INPUTS
    result["seconds"] = time.perf_counter() - started
    (args.output / "gradient-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {name: [{"update": row["collection_update"], "schemes": {key: {metric: data[metric] for metric in ["hold_swim_cosine", "hold_to_swim_norm_ratio", "negative_gradient_cosine_actual_parameter_delta", "total_gradient_cosine_actual_pooled"]} for key, data in row["schemes"].items()}} for row in records] for name, records in result["arms"].items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
