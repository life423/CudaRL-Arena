"""Independently audit saved current-swimming evidence; never step or train.

Python/NumPy are offline evidence tools only. The executable remains native
C++/CUDA. This verifier reconstructs articulated kinematics, observations,
likelihoods, rewards and acceptance from recorded state rather than invoking
the simulator or optimizer.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

import numpy as np
import verify_hold_credit as shared

N, T, PARAMETERS = 256, 160, 16
SIGMA, COEFFICIENT_LIMIT = .5, .5
SCALE = float(np.float32(.04))
FLOW_SCALE = float(np.float32(.002))
PARENT_SHA = "810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0"
HOLD_SHA = "C54F73B57AB4280D99802A7253C13004F95C754E39E0ACF8F6833377BBB760C3"
EXPECTED_KEYS = {(pose, second) for pose in (0, 3) for second in range(8, 61, 4)}
HASHES, VALIDATED = {}, set()


def read(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def js(path):
    return json.loads(read(path))


def rows(path):
    result = list(csv.DictReader(read(path).decode("utf-8").splitlines()))
    for row in result:
        for name, value in row.items():
            if value == "":
                continue
            try:
                row[name] = float(value)
            except ValueError:
                continue
            assert np.isfinite(row[name]), (str(path), name, value)
    return result


def near(actual, expected, absolute=2e-6, relative=2e-5):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    error = np.abs(a-b)
    assert np.all(error <= absolute+relative*np.abs(b)), (float(error.max()), absolute, relative)
    return float(error.max()) if error.size else 0.


def envelope(path):
    path = Path(path).resolve()
    raw = read(path)
    magic, size, checksum = struct.unpack_from("<QQQ", raw)
    assert magic == 0x31504B434C524443 and size == len(raw)-24
    assert path in VALIDATED, f"Unchecked native envelope: {path}"
    return shared.Reader(raw[24:])


def validate_envelopes(root, helper):
    paths = sorted(root.rglob("*.bin"))
    listing = root/"independent-envelope-paths.txt"
    listing.write_text("".join(str(path.resolve())+"\n" for path in paths), encoding="utf-8")
    read(helper)
    process = subprocess.run([str(helper.resolve()), str(listing.resolve())], capture_output=True,
                             text=True, check=True,
                             creationflags=0x08000000 if os.name == "nt" else 0)
    report = json.loads(process.stdout)
    assert report["passed"] and report["files"] == len(paths)
    assert report["physical_controls"] == report["optimizer_steps"] == 0
    VALIDATED.update(path.resolve() for path in paths)
    return report


def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    av, bv, aw, bw = a[..., :3], b[..., :3], a[..., 3:4], b[..., 3:4]
    return np.concatenate((aw*bv+bw*av+np.cross(av, bv),
                           aw*bw-np.sum(av*bv, axis=-1, keepdims=True)), axis=-1)


def rotate(q, vector, inverse=False):
    q, vector = np.asarray(q, float), np.asarray(vector, float)
    axis = -q[..., :3] if inverse else q[..., :3]
    return vector+2*np.cross(axis, np.cross(axis, vector)+vector*q[..., 3:4])


def rotate_float32(q, vector, inverse=False):
    """Preserve the defined float up-vector quantization before native acosf."""
    q, vector = np.asarray(q, np.float32), np.asarray(vector, np.float32)
    axis = -q[..., :3] if inverse else q[..., :3]
    return vector+np.cross(axis, np.cross(axis, vector)+vector*q[..., 3:4])*np.float32(2)


def axis_rotation(axis, angle):
    angle = np.asarray(angle, float)
    half = angle[..., None]/2
    return np.concatenate((np.asarray(axis)*np.sin(half), np.cos(half)), axis=-1)


def motion(states):
    """COM in SI metres and velocity from recorded 21-float body coordinates."""
    states = np.asarray(states, float)
    assert states.shape[-1] >= 21
    position, orientation = states[..., :3]*SCALE, states[..., 3:7]
    angles, velocity = states[..., 7:11], states[..., 11:21]
    centers, quaternions = [position], [orientation]
    anchors, yaw_axes, pitch_axes = [], [], []
    lengths = [float(np.float32(value)) for value in (.020, .026, .026)]
    for body in (1, 2):
        parent = body-1
        anchor = centers[parent]+rotate(quaternions[parent], [-lengths[parent]/2, 0, 0])
        yaw_axis = rotate(quaternions[parent], [0, 1, 0])
        yaw = qmul(quaternions[parent], axis_rotation([0, 1, 0], angles[..., parent*2]))
        pitch_axis = rotate(yaw, [0, 0, 1])
        q = qmul(yaw, axis_rotation([0, 0, 1], angles[..., parent*2+1]))
        anchors.append(anchor)
        yaw_axes.append(yaw_axis)
        pitch_axes.append(pitch_axis)
        quaternions.append(q)
        centers.append(anchor+rotate(q, [-lengths[body]/2, 0, 0]))
    pi, radius = float(np.float32(np.pi)), float(np.float32(.0015))
    head_mass = 4*pi*(lengths[0]/2)*float(np.float32(.0064))*float(np.float32(.008))/3*1000
    rod_mass = pi*radius*radius*lengths[1]*1000
    tail_mass = rod_mass+float(np.float32(.0097))*lengths[2]*float(np.float32(.0001))*1000
    masses = [head_mass, rod_mass, tail_mass]
    weighted_position = np.zeros_like(position)
    weighted_velocity = np.zeros_like(position)
    for body in range(3):
        speed = velocity[..., :3]+np.cross(velocity[..., 3:6], centers[body]-position)
        for joint in range(body):
            arm = centers[body]-anchors[joint]
            speed = speed+np.cross(yaw_axes[joint], arm)*velocity[..., 6+joint*2, None]
            speed = speed+np.cross(pitch_axes[joint], arm)*velocity[..., 7+joint*2, None]
        weighted_position += masses[body]*centers[body]
        weighted_velocity += masses[body]*speed
    return weighted_position/sum(masses), weighted_velocity/sum(masses)


def initial_body(pose):
    pose = np.asarray(pose, dtype=int)
    state = np.zeros(pose.shape+(21,), dtype=float)
    state[..., 0] = np.float32((pose.astype(np.float32)-np.float32(1.5))*np.float32(.02))
    state[..., 1] = -2
    yaw = np.float32((pose.astype(np.float32)-np.float32(1.5))*np.float32(.05))
    state[..., 3:7] = axis_rotation([0, 1, 0], yaw)
    state[..., 7] = np.where(pose%2, float(np.float32(.02)), -float(np.float32(.02)))
    state[..., 9] = -state[..., 7]
    return state


def case_specs(withheld=False):
    result = []
    for pose in ((1, 2) if withheld else (0, 3)):
        for magnitude in ((.0015, .0025) if withheld else (.001, .002)):
            vectors = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if withheld else [(1, 0), (-1, 0), (0, 1), (0, -1)]
            component = np.float32(np.float32(magnitude)/np.float32(math.sqrt(2))) if withheld else np.float32(magnitude)
            for x, z in vectors:
                result.append((pose, float(component*x), 0., float(component*z)))
    return np.asarray(result, dtype=float)


def observations(pre, poses, world_flow, controls):
    """Reconstruct all 44 inputs from compact PRE33 state and fixed episode case."""
    pre = np.asarray(pre, float)
    assert pre.shape[-1] == 33
    orientation = pre[..., 3:7]
    center, velocity = motion(pre)
    initial = initial_body(poses)
    start, _ = motion(initial)
    heading = rotate(initial[..., 3:7], [1, 0, 0])
    heading /= np.linalg.norm(heading, axis=-1, keepdims=True)
    displacement = center-start
    lane = heading*np.sum(displacement*heading, axis=-1, keepdims=True)-displacement
    output = np.zeros(pre.shape[:-1]+(44,))
    output[..., :3] = rotate(orientation, lane, True)/float(np.float32(.002))
    output[..., 3:6] = rotate(orientation, heading*float(np.float32(.008)), True)/float(np.float32(.02))
    output[..., 6:9] = rotate(orientation, heading, True)
    output[..., 9:12] = rotate(orientation, [0, 1, 0], True)
    output[..., 12:15] = rotate(orientation, velocity, True)/float(np.float32(.02))
    output[..., 15:18] = rotate(orientation, pre[..., 14:17], True)/4
    output[..., 18:22] = pre[..., 7:11]
    output[..., 22:26] = pre[..., 17:21]/12
    output[..., 26:30] = pre[..., 25:29]
    output[..., 30:34] = pre[..., 21:25]
    output[..., 34] = np.maximum(0, 1-np.asarray(controls)/40)
    output[..., 35] = 1
    output[..., 36:40] = pre[..., 29:33]/12
    output[..., 41:44] = rotate(orientation, world_flow, True)/FLOW_SCALE
    return output


def metrics(post, poses):
    post = np.asarray(post, float)
    center, velocity = motion(post)
    initial = initial_body(poses)
    start, _ = motion(initial)
    heading = rotate(initial[..., 3:7], [1, 0, 0])
    heading /= np.linalg.norm(heading, axis=-1, keepdims=True)
    displacement = center-start
    forward = np.sum(displacement*heading, axis=-1)
    perpendicular = displacement-heading*forward[..., None]
    q = post[..., 3:7]
    up = rotate_float32(q, [0, 1, 0]).astype(float)
    local_up = rotate(q, [0, 1, 0], True)
    forward_axis = rotate(q, [1, 0, 0])
    yaw = np.degrees(np.abs(np.arctan2(np.cross(heading, forward_axis)[..., 1],
                                      np.sum(heading*forward_axis, axis=-1))))
    return {"heading": heading, "forward": forward, "lane": np.linalg.norm(perpendicular[..., [0, 2]], axis=-1),
            "depth": np.abs(displacement[..., 1]), "signed_depth": displacement[..., 1],
            "velocity": velocity, "tilt": np.degrees(np.arccos(np.clip(up[..., 1], -1, 1))),
            "head_yaw": yaw, "roll": np.degrees(np.arctan2(-local_up[..., 2], local_up[..., 1])),
            "pitch": np.degrees(np.arctan2(local_up[..., 0], np.linalg.norm(local_up[..., 1:3], axis=-1)))}


def joint_latents(latent, backbone, body_flow):
    theta = COEFFICIENT_LIMIT*np.tanh(np.asarray(latent, float)).reshape((-1, 4, 4))
    flow, base = np.asarray(body_flow, float), np.asarray(backbone, float)
    forward, side = flow[..., 0, None], flow[..., 2, None]
    return base+theta[..., 0]*forward+theta[..., 1]*side+theta[..., 2]*forward*base+theta[..., 3]*side*base


def log_probability(means, latent):
    return np.sum(-.5*((np.asarray(latent)-means)/SIGMA)**2-math.log(SIGMA)-.5*math.log(2*math.pi), axis=-1)


def leave_one_out(returns, scale):
    returns = np.asarray(returns, float)
    assert returns.shape == (N,) and math.isfinite(scale) and scale > 0
    result = np.zeros(N)
    for case in range(16):
        ids = np.arange(case, N, 16)
        assert len(ids) == 16
        result[ids] = (returns[ids]-(returns[ids].sum()-returns[ids])/15)/scale
    return result


def reward(metrics_record, mean_squared_torque, control):
    velocity = metrics_record["velocity"]
    # The original heading is supplied in the recorded task-independent case.
    target = metrics_record["heading"]*float(np.float32(.008))
    horizontal = np.sum((velocity[..., [0, 2]]-target[..., [0, 2]])**2, axis=-1)/float(np.float32(.008))**2
    horizontal = np.where(np.asarray(control) >= 40, horizontal, 0)
    recovery = np.clip(-metrics_record["signed_depth"]/2, -.001, .001)
    vertical = ((velocity[..., 1]-recovery)/float(np.float32(.008)))**2
    cost = horizontal+vertical+.05*(metrics_record["lane"]/.002)**2+.05*(metrics_record["depth"]/.001)**2
    cost += .05*(metrics_record["roll"]/10)**2+.01*(metrics_record["pitch"]/10)**2
    cost += .002*np.asarray(mean_squared_torque)
    return -float(np.float32(.05))*cost


def parents(root, protocol):
    parent_raw, hold_raw = read(root/"parent-checkpoint.bin"), read(root/"holding-parent.bin")
    assert hashlib.sha256(parent_raw).hexdigest().upper() == PARENT_SHA
    assert hashlib.sha256(hold_raw).hexdigest().upper() == HOLD_SHA
    parent_checksum = shared.fnv(parent_raw[24:])
    hold_checksum = shared.fnv(hold_raw[24:])
    hold = envelope(root/"holding-parent.bin")
    assert [hold.value("I"), hold.value("I")] == [0x314D4148, 1]
    assert hold.value("Q") != 0 and hold.value("Q") == parent_checksum
    assert [hold.value("I") for _ in range(7)] == [6, 6, 3, 4, 41, 4, 336]
    assert [hold.value("i"), hold.value("i")] == [7, 112]
    frozen, holding = hold.array(size=2532).copy(), hold.array(size=2).copy()
    gain, sigma = hold.value("f"), hold.value("f")
    expected_gain = math.exp(float(holding[0]))/(1+math.exp(float(holding[0])))
    near(gain, expected_gain, 2e-7, 0)
    near(sigma, .05+7.95/(1+math.exp(-float(holding[1]))), 2e-6, 0)
    old_adam = shared.Reader(hold.array("u1").tobytes()); hold.finish()
    old_config = [old_adam.value("f") for _ in range(5)]
    assert np.array_equal(np.float32(old_config), np.float32([.05, .9, .999, .00001, .5]))
    assert old_adam.value("i") == 112
    assert np.isfinite(old_adam.array(size=2)).all()
    assert np.all(old_adam.array(size=2) >= 0); old_adam.finish()
    migration = envelope(root/"migrated-initial.bin")
    assert [migration.value("I"), migration.value("I")] == [0x35415050, 5]
    assert migration.value("Q") == protocol and migration.value("I") == 1247037243 and migration.value("i") == 336
    assert [migration.value("I") for _ in range(9)] == [6, 6, 3, 9, 4, 41, 4, 2532, 2433]
    assert migration.value("f") == float(np.float32(.8))
    assert np.array_equal(migration.array(size=4), np.full(4, -1))
    assert np.array_equal(migration.array(size=2532), frozen)
    assert np.isfinite(migration.array(size=2433)).all()
    for count in (2532, 2433):
        assert np.isfinite([migration.value("f") for _ in range(5)]).all()
        assert migration.value("i") == 21700
        assert np.isfinite(migration.array(size=count)).all()
        assert np.all(migration.array(size=count) >= 0)
    assert [migration.value("i"), migration.value("i")] == [0, 0]; migration.finish()
    assert np.all(frozen[:1312].reshape(32, 41)[:, 40] == 0)
    return frozen, holding, gain, parent_checksum, hold_checksum


def read_policy(path, protocol, frozen, holding, gain, parent_checksum, hold_checksum):
    reader = envelope(path)
    assert [reader.value("I"), reader.value("I")] == [0x31575343, 1]
    assert [reader.value("Q") for _ in range(3)] == [protocol, parent_checksum, hold_checksum]
    assert [reader.value("I") for _ in range(10)] == [6, 1, 3, 5, 44, 41, 4, 16, 336, 7]
    update, age = reader.value("i"), reader.value("i")
    assert np.array_equal(reader.array(size=2532), frozen)
    assert np.array_equal(reader.array(size=2), holding) and reader.value("f") == gain
    means = reader.array(size=16).copy()
    assert reader.value("f") == SIGMA and reader.value("f") == COEFFICIENT_LIMIT
    adam = shared.Reader(reader.array("u1").tobytes()); reader.finish()
    config = [adam.value("f") for _ in range(5)]
    assert np.array_equal(np.float32(config), np.float32([.01, .9, .999, .00001, .5]))
    assert adam.value("i") == age
    first, second = adam.array(size=16), adam.array(size=16); adam.finish()
    assert 0 <= update <= 32 and 0 <= age <= 512 and age%4 == 0
    assert np.isfinite(means).all() and np.isfinite(first).all() and np.all(np.isfinite(second) & (second >= 0))
    if update == 0:
        assert age == 0 and np.all(means == 0) and np.all(first == 0) and np.all(second == 0)
    return means, {"update": update, "Adam_steps": age, "means": means.tolist(),
                   "deterministic_coefficients": (COEFFICIENT_LIMIT*np.tanh(means.astype(float))).tolist(),
                   "sha256": HASHES[str(path.resolve())]}


def read_batch(path, frozen, expected_means, layout):
    reader = envelope(path)
    assert [reader.value("I"), reader.value("I")] == [0x31425343, 1]
    assert [reader.value("i") for _ in range(8)] == [N, T, T, 41, 3, 4, 16, 16]
    means = reader.array(size=16)
    latent = reader.array(size=N*16).reshape(N, 16)
    old_logp, returns = reader.array(size=N), reader.array(size=N)
    assert np.array_equal(means, expected_means)
    abi = reader.value("I")
    offsets = layout["episode_counts"]
    assert abi == offsets["size"] and offsets["controls"] == 0 and offsets["substeps"] == 8
    assert offsets["reward_sum"] == 16 and offsets["failure_code"] == 24
    counts_type = np.dtype({"names": ["controls", "substeps", "reward_sum", "failure"],
                            "formats": ["<u8", "<u8", "<f8", "<i4"],
                            "offsets": [0, 8, 16, 24], "itemsize": abi})
    counts = reader.array(counts_type, N)
    obs = reader.array(size=N*T*41).reshape(T, N, 41)
    flow = reader.array(size=N*T*3).reshape(T, N, 3)
    base = reader.array(size=N*T*4).reshape(T, N, 4)
    action = reader.array(size=N*T*4).reshape(T, N, 4)
    rewards = reader.array(size=N*T).reshape(T, N)
    recorded_metrics = reader.array(size=N*T*16).reshape(T, N, 16)
    pre = reader.array(size=N*T*33).reshape(T, N, 33)
    post = reader.array(size=N*T*23).reshape(T, N, 23)
    ending = {}
    for key in ("world_size", "task_size"):
        width, count = reader.value("I"), reader.value("Q")
        assert width == layout[key] and count == N
        ending[key] = bytes(reader.data[reader.at:reader.at+width*count])
        reader.at += width*count
    reader.finish()
    for value in (means, latent, old_logp, returns, obs, flow, base, action, rewards, recorded_metrics, pre, post):
        assert np.isfinite(value).all()
    assert np.all(counts["controls"] == T) and np.all(counts["substeps"] == T*10) and np.all(counts["failure"] == 0)
    assert np.all(abs(pre[..., 7:11]) <= 1) and np.all(abs(post[..., 7:11]) <= 1)
    assert np.all(abs(pre[..., 25:29]) <= 1) and np.all((post[..., 22] >= 0) & (post[..., 22] <= 1))
    # Every row must be the next accepted physical state, never a hidden reset.
    continuity_error = near(pre[1:, :, :21], post[:-1, :, :21], 0, 0)
    specifications = case_specs()[np.arange(N)%16]
    poses, current = specifications[:, 0].astype(int), specifications[:, 1:4]
    near(pre[0, :, :21], initial_body(poses), 2e-7, 2e-6)
    predicted_obs = observations(pre, poses, current, np.arange(T)[:, None])
    observation_error = near(np.concatenate((obs, flow), axis=-1), predicted_obs, 3e-5, 3e-5)
    assert np.all(obs[..., 40] == 0)
    probability_error = near(old_logp, log_probability(means, latent), 1e-4, 0)
    predicted_action = joint_latents(latent, base, flow)
    action_error = near(action, predicted_action, 2e-5, 2e-6)
    assert layout["world_size"] == 696 and layout["task_size"] == 152 and layout["body_size"] == 92
    world_dtype = np.dtype({"names": ["body", "current", "held", "last_torque", "raw", "actual", "requested", "mean_sq",
                                     "controls", "substeps", "time", "invalid", "rates", "attempted", "failure"],
        "formats": [("<f4", 23), ("<f4", 3), ("<f4", 4), ("<f4", 4), ("<f4", 4), ("<f4", 40), ("<f4", 40), "<f4",
                    "<u8", "<u8", "<f8", "u1", ("<f4", 4), "<u8", "<i4"],
        "offsets": [0, 104, 152, 168, 184, 200, 360, 520, 528, 536, 544, 554, 616, 632, 688], "itemsize": 696})
    ending_world = np.frombuffer(ending["world_size"], dtype=world_dtype)
    assert np.all(ending_world["controls"] == T) and np.all(ending_world["substeps"] == T*10)
    assert np.all(ending_world["attempted"] == T*10) and np.all(ending_world["failure"] == 0) and np.all(ending_world["invalid"] == 0)
    near(ending_world["current"], current, 0, 0)
    near(ending_world["body"][:, :21], post[-1, :, :21], 0, 0)
    near(ending_world["body"][:, 21], post[-1, :, 21], 0, 0)
    near(ending_world["mean_sq"], post[-1, :, 22], 0, 0)
    commands = np.tanh(action.astype(float))
    near(ending_world["raw"], commands[-1], 2e-7, 2e-6)
    assert np.all(abs(ending_world["actual"]) <= 1) and np.isfinite(ending_world["requested"]).all()
    near(ending_world["mean_sq"], np.mean(ending_world["actual"].astype(float)**2, axis=1), 2e-7, 2e-6)
    # Pure target-filter algebra proves the recorded tanh commands reached the
    # actuator filter. This does not integrate any body or call a stepper.
    angle, rate = pre[..., 21:25].astype(float).copy(), pre[..., 29:33].astype(float).copy()
    h, omega = float(np.float32(.005)), float(np.float32(2*np.float32(np.pi)))
    for _ in range(10):
        acceleration = omega*omega*(commands-angle)-2*omega*rate
        rate = rate+h*acceleration
        angle = angle+h*rate
    next_angle = np.concatenate((pre[1:, :, 21:25], ending_world["held"][None, ...]), axis=0)
    next_rate = np.concatenate((pre[1:, :, 29:33], ending_world["rates"][None, ...]), axis=0)
    filter_angle_error = near(next_angle, angle, 2e-6, 2e-5)
    filter_rate_error = near(next_rate, rate, 2e-5, 2e-5)
    backbone_error = 0.
    flat_obs, flat_base = obs.reshape(-1, 41), base.reshape(-1, 4)
    for start in range(0, N*T, 4096):
        predicted = shared.forward(frozen.astype(float), flat_obs[start:start+4096].astype(float))
        backbone_error = max(backbone_error, near(flat_base[start:start+4096], predicted, 2e-5, 2e-6))
    actual_metrics = metrics(post, poses)
    values = np.stack((actual_metrics["forward"], actual_metrics["lane"], actual_metrics["depth"], actual_metrics["tilt"],
                       actual_metrics["signed_depth"], *np.moveaxis(actual_metrics["velocity"], -1, 0),
                       actual_metrics["roll"], actual_metrics["pitch"], actual_metrics["head_yaw"]), axis=-1)
    metric_errors = [near(recorded_metrics[..., col], values[..., col], .003 if col in (3, 8, 9, 10) else 2e-7, 2e-5)
                     for col in range(11)]
    near(recorded_metrics[..., 11], post[..., 22], 0, 0)
    near(recorded_metrics[..., 15], np.broadcast_to(np.arange(1, T+1)[:, None], (T, N)), 0, 0)
    predicted_rewards = reward(actual_metrics, post[..., 22], np.arange(T)[:, None])
    reward_error = near(rewards, predicted_rewards, 2e-5, 3e-5)
    sums = rewards.astype(float).sum(axis=0)
    return_error = near(counts["reward_sum"], sums, 1e-7, 1e-7)
    assert np.array_equal(returns, np.float32(sums/8))
    center, _ = motion(post)
    heading = actual_metrics["heading"]
    displacement = center[40:]-center[39]
    forward = np.sum(displacement*heading, axis=-1)
    elapsed = np.arange(1, T-40+1)[:, None]*float(np.float32(.05))
    steady_speed = forward/elapsed
    velocity_error = np.sum((actual_metrics["velocity"][40:]-heading*float(np.float32(.008)))**2, axis=-1)
    rmse = np.sqrt(np.cumsum(velocity_error, axis=0)/np.arange(1, T-40+1)[:, None])
    sideways = displacement-heading*forward[..., None]
    path = np.degrees(np.arctan2(np.linalg.norm(sideways[..., [0, 2]], axis=-1), forward))
    assert np.all(recorded_metrics[:40, :, 12:15] == 0)
    near(recorded_metrics[40:, :, 12], steady_speed, 2e-7, 2e-5)
    near(recorded_metrics[40:, :, 13], rmse, 2e-7, 2e-5)
    near(recorded_metrics[40:, :, 14], path, .003, 2e-5)
    work = post[-1, :, 21]/np.maximum(actual_metrics["forward"][-1], 1e-9)
    passed = ((steady_speed[-1] >= .0072) & (steady_speed[-1] <= .0088) & (rmse[-1] <= .0025)
              & (actual_metrics["lane"].max(axis=0) <= .002) & (actual_metrics["depth"].max(axis=0) <= .001)
              & (actual_metrics["tilt"].max(axis=0) <= 10) & (path[-1] <= 5) & (work <= .001))
    task_dtype = np.dtype({"names": ["controls", "outcome"], "formats": ["<i4", "<i4"],
                           "offsets": [92, 148], "itemsize": 152})
    ending_task = np.frombuffer(ending["task_size"], dtype=task_dtype)
    assert np.all(ending_task["controls"] == T) and np.array_equal(ending_task["outcome"] == 1, passed)
    table = rows(path.with_name(path.name.replace("-batch-", "-episodes-").replace(".bin", ".csv")))
    assert len(table) == N
    for i, row in enumerate(table):
        assert row["episode"] == i and row["case"] == i%16 and row["pose"] == poses[i]
        near([row["current_x_mm_s"], row["current_z_mm_s"]], current[i, [0, 2]]*1000, 1e-8, 1e-8)
        assert row["controls"] == T and row["substeps"] == T*10 and row["failure_code"] == 0
        near(row["old_logp"], old_logp[i], 1e-8, 1e-10)
        near(row["return"], returns[i], 1e-8, 1e-10)
        near(row["reward_sum"], sums[i], 1e-8, 1e-10)
        near([row[f"latent{k}"] for k in range(16)], latent[i], 1e-8, 1e-10)
    return returns.astype(float), {"controls": N*T, "substeps": N*T*10, "means": means.tolist(),
        "probability_error": probability_error, "observation_error": observation_error,
        "backbone_error": backbone_error, "action_error": action_error, "reward_error": reward_error,
        "return_error": return_error, "continuity_error": continuity_error,
        "filter_angle_error": filter_angle_error, "filter_rate_error": filter_rate_error,
        "sampled_complete_passes": int(passed.sum()), "metric_field_max_errors": metric_errors}


BODY_COLUMNS = (["position_x", "position_y", "position_z", "qx", "qy", "qz", "qw"]
                +[f"angle{i}" for i in range(4)]+[f"velocity{i}" for i in range(10)]+["positive_work_J"])


def body_array(table):
    values = np.asarray([[row[key] for key in BODY_COLUMNS] for row in table], dtype=float)
    assert values.shape == (len(table), 22) and np.isfinite(values).all()
    assert np.all(abs(values[:, 7:11]) <= 1) and np.all(values[:, 21] >= 0)
    return values


def swim_metrics(states, pose, start_control=40):
    count = len(states)-1
    assert count > start_control
    values = metrics(states, pose)
    center, _ = motion(states)
    heading = np.asarray(values["heading"])
    displacement = center[-1]-center[start_control]
    forward = float(np.sum(displacement*heading))
    side = displacement-heading*forward
    elapsed = (count-start_control)*float(np.float32(.05))
    velocity_error = values["velocity"][start_control+1:]-heading*float(np.float32(.008))
    result = {"steady_mm_s": forward/elapsed*1000,
              "RMSE_mm_s": float(np.sqrt(np.mean(np.sum(velocity_error**2, axis=-1))))*1000,
              "lateral_mm": float(values["lane"].max())*1000,
              "depth_mm": float(values["depth"].max())*1000,
              "tilt_deg": float(values["tilt"].max()),
              "path_deg": float(np.degrees(np.arctan2(np.linalg.norm(side[[0, 2]]), forward))),
              "work_J_per_m": float((states[-1, 21]-states[0, 21])/max(values["forward"][-1], 1e-9)),
              "forward_mm": float(values["forward"][-1])*1000}
    return result, values


def swim_gates(value):
    return {"speed": 7.2 <= value["steady_mm_s"] <= 8.8, "RMSE": value["RMSE_mm_s"] <= 2.5,
            "lane": value["lateral_mm"] <= 2, "depth": value["depth_mm"] <= 1,
            "tilt": value["tilt_deg"] <= 10, "path": value["path_deg"] <= 5,
            "work": value["work_J_per_m"] <= .001}


def current_assessment(directory, means, expected_specs, endurance=False):
    summary, configs = rows(directory/"summary.csv"), rows(directory/"cases.csv")
    controls, states = rows(directory/"controls.csv"), rows(directory/"states.csv")
    initial, windows = rows(directory/"initial.csv"), rows(directory/"windows.csv")
    # The display-pose CSV is independently bound to the richer state records.
    pose_rows = rows(directory/"poses.csv")
    assert len(summary) == len(configs) == len(initial) == len(expected_specs)
    details, counted = [], 0
    for case, specification in enumerate(expected_specs):
        pose, fx, fy, fz = specification
        config = configs[case]; count = 1280 if endurance else T
        assert config["case"] == case and config["pose"] == pose and config["maximum_controls"] == count
        near([config["current_x_m_s"], config["current_z_m_s"]], [fx, fz], 1e-11, 1e-10)
        sequence = [row for row in states if row["case"] == case]
        assert [row["control"] for row in sequence] == list(range(count+1))
        body = body_array(sequence)
        near(body[0, :21], initial_body(int(pose)), 2e-7, 2e-6)
        outcome, calculated = swim_metrics(body, int(pose))
        near([config[f"heading_{axis}"] for axis in "xyz"], calculated["heading"], 1e-7, 1e-6)
        resets = initial[case]
        for key, value in (("forward_mm", calculated["forward"][0]*1000), ("lane_mm", calculated["lane"][0]*1000),
                           ("depth_mm", calculated["depth"][0]*1000), ("tilt_deg", calculated["tilt"][0]),
                           ("speed_mm_s", np.linalg.norm(calculated["velocity"][0])*1000)):
            near(resets[key], value, .003 if key == "tilt_deg" else .0002, 2e-5)
        samples = [row for row in controls if row["case"] == case]
        assert [row["control"] for row in samples] == list(range(1, count+1))
        for key, values in (("forward_mm", calculated["forward"][1:]*1000), ("lane_mm", calculated["lane"][1:]*1000),
                            ("depth_mm", calculated["depth"][1:]*1000), ("tilt_deg", calculated["tilt"][1:])):
            near([row[key] for row in samples], values, .003 if key == "tilt_deg" else .0002, 2e-5)
        near([[row[f"velocity_{axis}_mm_s"] for axis in "xyz"] for row in samples],
             calculated["velocity"][1:]*1000, .0002, 2e-5)
        base = np.asarray([[row[f"base{joint}"] for joint in range(4)] for row in samples])
        native_latent = np.asarray([[row[f"latent{joint}"] for joint in range(4)] for row in samples])
        native_command = np.asarray([[row[f"command{joint}"] for joint in range(4)] for row in samples])
        flow = rotate(body[:-1, 3:7], [fx, fy, fz], True)/FLOW_SCALE
        near(native_latent, joint_latents(np.broadcast_to(means, (count, 16)), base, flow), 2e-5, 2e-6)
        near(native_command, np.tanh(native_latent), 2e-7, 2e-6)
        post_metrics = {key: value[1:] if key != "heading" else value for key, value in calculated.items()}
        expected_reward = reward(post_metrics, np.asarray([row["mean_squared_torque"] for row in sequence[1:]]), np.arange(count))
        native_reward = np.asarray([row["reward"] for row in samples])
        near(native_reward, expected_reward, 2e-5, 3e-5)
        native = summary[case]
        assert native["case"] == case and native["pose"] == pose
        assert native["controls"] == count and native["substeps"] == count*10
        near([native["current_x_mm_s"], native["current_z_mm_s"]], [fx*1000, fz*1000], 1e-7, 1e-7)
        for key in outcome:
            near(native[key], outcome[key], .003 if key in ("tilt_deg", "path_deg") else .0002, 2e-5)
        near(native["return"], native_reward.sum()/(count*.05), 2e-8, 2e-6)
        gates = swim_gates(outcome)
        extra = []
        case_windows = [row for row in windows if row["case"] == case]
        assert len(case_windows) == (15 if endurance else 0)
        if endurance:
            center, _ = motion(body)
            for window, end in zip(case_windows, range(160, 1281, 80)):
                assert window["end_s"] == end*.05
                start = end-80
                velocity_error = calculated["velocity"][start+1:end+1]-calculated["heading"]*.008
                mean = np.dot(center[end]-center[start], calculated["heading"])/(80*float(np.float32(.05)))*1000
                rmse = np.sqrt(np.mean(np.sum(velocity_error**2, axis=-1)))*1000
                near(window["mean_mm_s"], mean, .0002, 2e-5)
                near(window["RMSE_mm_s"], rmse, .0002, 2e-5)
                passing = bool(7.2 <= mean <= 8.8 and rmse <= 2.5)
                assert bool(window["passed"]) == passing
                extra.append({"end_s": window["end_s"], "passed": passing, "mean_mm_s": float(mean), "RMSE_mm_s": float(rmse)})
        passed = bool(all(gates.values()) and all(value["passed"] for value in extra))
        assert bool(native["passed"]) == passed
        saved_poses = [row for row in pose_rows if row["case"] == case]
        assert [row["control"] for row in saved_poses] == list(range(count+1))
        for saved, full in zip(saved_poses, body):
            near([saved[f"position_{axis}_model"] for axis in "xyz"], full[:3], 1e-10, 1e-10)
            near([saved[f"quat_{axis}"] for axis in "xyzw"], full[3:7], 1e-10, 1e-10)
            near([saved[f"angle{joint}"] for joint in range(4)], full[7:11], 1e-10, 1e-10)
        counted += count
        details.append({"case": case, "pose": int(pose), "flow_m_s": [fx, fy, fz], "passed": passed,
                        "metrics": outcome, "gates": gates, "windows": extra, "controls": count})
    assert counted == len(controls) and len(states) == counted+len(expected_specs)
    return {"passes": sum(value["passed"] for value in details), "cases": details, "controls": counted,
            "substeps": counted*10, "mean_cost": -float(np.mean([row["return"] for row in summary]))}


def calm_assessment(directory):
    recorded_states = rows(directory/"states.csv")
    short = rows(directory/"short-evaluation.csv")
    holds, hold_controls = rows(directory/"holds.csv"), rows(directory/"hold-controls.csv")
    long_rows = rows(directory/"long/sustained-windows.csv")
    assert len(short) == 4 and len(holds) == 6 and len(long_rows) == 30
    short_details, hold_details, long_details, passing_keys = [], [], [], set()
    counted = 0
    for case in range(4):
        sequence = [row for row in recorded_states if row["category"] == "short" and row["case"] == case]
        assert [row["control"] for row in sequence] == list(range(161))
        value, _ = swim_metrics(body_array(sequence), case)
        native = next(row for row in short if row["pose"] == case)
        for name in value:
            near(native[name], value[name], .003 if name in ("tilt_deg", "path_deg") else .0003, 3e-5)
        gates = swim_gates(value); passed = bool(all(gates.values()))
        assert bool(native["passed"]) == passed
        short_details.append({"pose": case, "passed": passed, "gates": gates, "metrics": value})
        counted += 160
    for case in range(6):
        pose, offset = (3 if case%2 else 0), (0 if case//2 == 0 else -.0005 if case//2 == 1 else .0005)
        sequence = [row for row in recorded_states if row["category"] == "hold" and row["case"] == case]
        assert [row["control"] for row in sequence] == list(range(1281))
        state = body_array(sequence); center, velocity = motion(state)
        values = metrics(state, pose)
        initial = initial_body(pose)
        original_center, _ = motion(initial)
        anchor = np.array([center[0, 0], original_center[1], center[0, 2]])
        anchor_error = np.linalg.norm(center-anchor, axis=-1)
        speed = np.linalg.norm(velocity, axis=-1)
        peaks = {"max_speed_mm_s": float(speed.max()*1000), "max_anchor_mm": float(anchor_error.max()*1000),
                 "max_depth_mm": float(values["depth"].max()*1000), "max_lane_mm": float(values["lane"].max()*1000),
                 "max_tilt_deg": float(values["tilt"].max()),
                 "final8_excursion_mm": float(np.max(abs(values["forward"][1120:]-values["forward"][1120]))*1000)}
        native = holds[case]
        assert native["case"] == case and native["pose"] == pose
        near(native["offset_mm"], offset*1000, 1e-7, 1e-7)
        near(abs(center[0, 1]-original_center[1]), abs(offset), 2e-8, 1e-6)
        for key in peaks:
            near(native[key], peaks[key], .003 if key == "max_tilt_deg" else .0003, 2e-5)
        gates = {key: peaks[key] <= limit for key, limit in (("max_speed_mm_s", .5), ("max_anchor_mm", 2),
            ("max_depth_mm", 1), ("max_lane_mm", 2), ("max_tilt_deg", 10), ("final8_excursion_mm", 2))}
        passed = bool(all(gates.values())); assert bool(native["passed"]) == passed
        trace = [row for row in hold_controls if row["case"] == case]
        assert [row["control"] for row in trace] == list(range(1, 1281))
        for key, value in (("forward_mm", values["forward"][1:]*1000), ("speed_mm_s", speed[1:]*1000),
                           ("anchor_mm", anchor_error[1:]*1000), ("depth_mm", values["depth"][1:]*1000),
                           ("lane_mm", values["lane"][1:]*1000), ("tilt_deg", values["tilt"][1:])):
            near([row[key] for row in trace], value, .003 if key == "tilt_deg" else .0003, 2e-5)
        hold_details.append({"case": case, "pose": pose, "passed": passed, "gates": gates, "peaks": peaks})
        counted += 1280
    for case, pose in enumerate((0, 3)):
        sequence = [row for row in recorded_states if row["category"] == "long" and row["case"] == case]
        assert [row["control"] for row in sequence] == list(range(1201))
        state = body_array(sequence); center, velocity = motion(state); values = metrics(state, pose)
        for native in (row for row in long_rows if row["pose"] == pose):
            end = int(native["end_s"]*20); start = 40 if end == 80 else end-80
            displacement = center[end]-center[start]
            forward = float(np.dot(displacement, values["heading"]))
            side = displacement-values["heading"]*forward
            velocity_error = velocity[start+1:end+1]-values["heading"]*float(np.float32(.008))
            value = {"steady_mm_s": forward/((end-start)*float(np.float32(.05)))*1000,
                "RMSE_mm_s": float(np.sqrt(np.mean(np.sum(velocity_error**2, axis=-1)))*1000),
                "lateral_mm": float(values["lane"][:end+1].max()*1000), "depth_mm": float(values["depth"][:end+1].max()*1000),
                "tilt_deg": float(values["tilt"][:end+1].max()),
                "path_deg": float(np.degrees(np.arctan2(np.linalg.norm(side[[0, 2]]), forward))),
                "work_J_per_m": float((state[end, 21]-state[start, 21])/max(forward, 1e-9))}
            for name in value:
                near(native["mean_mm_s" if name == "steady_mm_s" else name], value[name], .003 if name in ("path_deg", "tilt_deg") else .0003, 3e-5)
            gates = swim_gates(value); passed = bool(all(gates.values())); assert bool(native["passed"]) == passed
            key = (pose, int(native["end_s"]))
            if passed: passing_keys.add(key)
            long_details.append({"pose": pose, "end_s": key[1], "passed": passed, "gates": gates, "metrics": value})
        counted += 1200
    assert counted == 10720 and len(recorded_states) == counted+12
    assert passing_keys == EXPECTED_KEYS
    return {"short_passes": sum(row["passed"] for row in short_details), "hold_passes": sum(row["passed"] for row in hold_details),
            "long_passes": len(passing_keys), "long_pass_keys": sorted(passing_keys), "controls": counted,
            "short": short_details, "holds": hold_details, "long": long_details}


def verify_assembly(path):
    reader = envelope(path)
    assert reader.value("I") == 1 and reader.value("I") == 4568
    dtype = np.dtype({"names": ["body", "flow", "systems", "direct_D", "direct_P"],
                      "formats": [("<f4", 23), ("<f4", 3), ("<f4", (5,220)), "<f8", "<f8"],
                      "offsets": [0, 92, 104, 4552, 4560], "itemsize": 4568})
    groups = [reader.array(dtype, 8), reader.array(dtype, 8)]; reader.finish()
    details = []
    for name, block in zip(("CPU", "CUDA"), groups):
        assert np.isfinite(block["body"]).all() and np.isfinite(block["systems"]).all()
        forced, negative, comoving, zero, reference = np.moveaxis(block["systems"], 1, 0)
        assert np.array_equal(zero, reference)
        assert np.array_equal(forced[:, :100], negative[:, :100])
        assert np.array_equal(forced[:, 200:210], negative[:, 200:210])
        r_new, r_old = forced[:, 100:200], negative[:, 100:200]
        r_error = abs(r_new.astype(float)-r_old.astype(float))
        bound = 64*np.finfo(np.float32).eps*np.maximum(abs(r_new).max(axis=1), abs(r_old).max(axis=1))
        assert np.all(r_error <= bound[:, None])
        velocity = np.zeros((8,10)); velocity[:, :3] = block["flow"]
        galilean = negative[:, 210:220]+np.einsum("nij,nj->ni", r_old.reshape(8,10,10).astype(float), velocity)-forced[:, 210:220]
        moving = comoving[:, 210:220]-reference[:, 210:220]-np.einsum("nij,nj->ni", comoving[:, 100:200].reshape(8,10,10).astype(float), velocity)
        maximum_force = max(float(abs(galilean).max()), float(abs(moving).max()))
        assert maximum_force <= 1e-11
        assert np.max(abs(block["direct_D"])) <= 1e-16 and np.max(abs(block["direct_P"])) <= 1e-16
        details.append({"backend": name, "probes": 8, "maximum_force_error": maximum_force,
                        "maximum_R_difference": float(r_error.max()),
                        "maximum_direct_dissipation_W": float(abs(block["direct_D"]).max()),
                        "maximum_direct_supply_W": float(abs(block["direct_P"]).max())})
    return details


def verify_gate(root):
    gate = root/"physics-gate"
    assert read(gate/"approved-protocol.md") == read(root/"approved-protocol.md")
    zero = js(gate/"zero-started/result.json")
    assert zero["passed"] and zero["physical_controls"] == zero["physical_substeps"] == 0
    assembly = verify_assembly(gate/"zero-started/assembly-probes.bin")
    gate_protocol = shared.fnv(envelope(gate/"protocol.bin").data)
    marker = envelope(gate/"zero-passed.bin")
    assert marker.value("I") == 1 and marker.value("Q") == gate_protocol; marker.finish()
    report = js(gate/"run-started/result.json")
    assert report["passed"] and not report["fault"]
    assert report["PPO_updates"] == report["optimizer_steps"] == 0
    assert report["attempted_controls"] == report["completed_controls"] == 3760
    assert report["physical_substeps"] == 37600 and report["finished_cases"] == 26
    marker = envelope(gate/"gate-passed.bin")
    assert [marker.value("I") for _ in range(3)] == [0x31474343,1,1]
    assert [marker.value("Q") for _ in range(3)] == [3760,37600,gate_protocol]; marker.finish()
    seconds = float(read(gate/"run-started/active-seconds.txt"))
    near(seconds, report["active_seconds"], 1e-12, 1e-12)
    assert 0 < seconds < 900
    table = rows(gate/"run-started/states.csv")
    index = {(row["group"], int(row["case"]), int(row["backend"]), int(row["control"])): row for row in table}
    assert len(index) == len(table) == 3760+62
    groups = [("zero_swim", 4, 40, (0,1,2)), ("zero_hold", 6, 40, (0,1,2)),
              ("flow_saved_motor", 8, 80, (1,2)), ("flow_zero_target_servo", 8, 80, (1,2))]
    counted, maximum_position = 0, 0.
    for group, cases, count, backends in groups:
        for case in range(cases):
            for backend in backends:
                sequence = [index[group,case,backend,control] for control in range(count+1)]
                assert [row["substeps"] for row in sequence] == list(range(0,count*10+1,10))
                assert [row["attempted_substeps"] for row in sequence] == list(range(0,count*10+1,10))
                counted += count
                if group == "flow_zero_target_servo":
                    body = np.asarray([[row[key] for key in (["x","y","z","qx","qy","qz","qw"]
                                      +[f"j{i}" for i in range(4)]+[f"v{i}" for i in range(10)])] for row in sequence])
                    center, _ = motion(body)
                    direction = np.asarray([(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)][case%4])
                    assert float(np.dot(center[-1]-center[0], direction)) > 0
            for control in range(count+1):
                cpu, gpu = index[group,case,1,control], index[group,case,2,control]
                if group.startswith("zero_"):
                    old = index[group,case,0,control]
                    for key in cpu:
                        if key != "backend": assert old[key] == cpu[key], (group,case,control,key)
                for key in ("x","y","z"):
                    difference = abs(cpu[key]-gpu[key]); maximum_position = max(maximum_position,difference)
                    assert difference <= 2e-5
                for key in ["qx","qy","qz","qw"]+[f"j{i}" for i in range(4)]:
                    assert abs(cpu[key]-gpu[key]) <= 2e-5
                for i in range(10):
                    assert abs(cpu[f"v{i}"]-gpu[f"v{i}"]) <= (2e-7 if i < 3 else 2e-5)
                for key in ("joint_work","positive_work","relative_water_loss","external_work","numerical_loss","energy_error"):
                    assert abs(cpu[key]-gpu[key]) <= 1e-12+1e-4*max(abs(cpu[key]),abs(gpu[key]))
    assert counted == 3760
    near(maximum_position, report["maximum_position_error_model"], 1e-12, 1e-8)
    return {"passed": True, "controls": counted, "substeps": counted*10, "active_seconds": seconds,
            "assembly": assembly, "maximum_position_error_model": maximum_position}


def verify_gate_failure(root):
    gate = root/"physics-gate"
    terminal = js(root/"study-result.json")
    native = js(gate/"run-started/result.json")
    assert terminal["schema"] == 1 and terminal["study"] == "current_swim_v1"
    assert terminal["phase"] == "physics_gate_failed" and terminal["closed"] and not terminal["qualified"]
    assert not terminal["training_started"] and not terminal["current_suites_run"]
    assert terminal["PPO_updates"] == terminal["actor_steps"] == 0
    assert terminal["critic_steps"] == terminal["critic_forward_calls"] == 0
    assert all(terminal[key] is None for key in ("baseline_current_passes","final_current_passes","withheld_passes","endurance_passes"))
    assert terminal["gate_result_sha256"] == HASHES[str((gate/"run-started/result.json").resolve())]
    assert not native["passed"] and native["fault"] and native["PPO_updates"] == native["optimizer_steps"] == 0
    assert not (root/"calibration-result.json").exists() and not (root/"training-result.json").exists()
    assert not (gate/"gate-passed.bin").exists()
    assert read(root/"approved-protocol.md") == read(gate/"approved-protocol.md")
    assembly = verify_assembly(gate/"zero-started/assembly-probes.bin")
    zero = js(gate/"zero-started/result.json")
    assert zero["passed"] and zero["physical_controls"] == zero["physical_substeps"] == 0
    proof = envelope(gate/"run-started/states.bin")
    assert [proof.value("I") for _ in range(3)] == [1,696,112]
    width = 16+696+112
    count = (len(proof.data)-proof.at)//width
    assert len(proof.data)-proof.at == count*width
    dtype = np.dtype({"names": ["group","case","backend","control","body","actual","world_controls","substeps","unsafe","invalid","failure"],
                      "formats": ["<i4","<i4","<i4","<i4",("<f4",23),("<f4",40),"<u8","<u8","u1","u1","<i4"],
                      "offsets": [0,4,8,12,16,216,544,552,569,570,704], "itemsize": width})
    raw = np.frombuffer(proof.data, dtype=dtype, count=count, offset=proof.at)
    proof.at += count*width; proof.finish()
    assert np.isfinite(raw["body"]).all() and np.all(abs(raw["body"][:,7:11]) <= 1)
    assert np.all(abs(raw["actual"]) <= 1) and np.all(raw["unsafe"] == 0) and np.all(raw["invalid"] == 0)
    assert np.all(raw["failure"] == 0) and np.array_equal(raw["world_controls"], raw["control"])
    assert np.array_equal(raw["substeps"], raw["control"]*10)
    attempted = int(np.sum(raw["control"] > 0))
    assert attempted == native["attempted_controls"] == native["completed_controls"] == terminal["controls"]
    assert attempted*10 == native["physical_substeps"] == terminal["substeps"]
    assert attempted <= 3760
    assert terminal["maximum_controls"] == 1466480 and terminal["closed_unused_controls"] == 1466480-attempted
    table = rows(gate/"run-started/states.csv")
    assert len(table) == count
    index = {(row["group"],int(row["case"]),int(row["backend"]),int(row["control"])):row for row in table}
    assert len(index) == count
    names = ["zero_swim","zero_hold","flow_saved_motor","flow_zero_target_servo"]
    for saved, native_state in zip(table,raw):
        assert saved["group"] == names[int(native_state["group"])]
        assert saved["case"] == native_state["case"] and saved["backend"] == native_state["backend"] and saved["control"] == native_state["control"]
        near([saved[key] for key in ["x","y","z","qx","qy","qz","qw"]+[f"j{i}" for i in range(4)]+[f"v{i}" for i in range(10)]], native_state["body"][:21], 0, 0)
    violations, finished = [], 0
    definitions = [(0,4,40,(0,1,2)),(1,6,40,(0,1,2)),(2,8,80,(1,2)),(3,8,80,(1,2))]
    for group,cases,horizon,backends in definitions:
        for case in range(cases):
            available = [key[3] for key in index if key[:3] == (names[group],case,1)]
            if not available: continue
            last = max(available)
            for backend in backends:
                assert all((names[group],case,backend,k) in index for k in range(last+1))
            if last == horizon: finished += 1
            for control in range(last+1):
                cpu,gpu=index[names[group],case,1,control],index[names[group],case,2,control]
                if group < 2:
                    old=index[names[group],case,0,control]
                    assert all(old[key] == cpu[key] for key in cpu if key != "backend")
                for key,limit in [(axis,2e-5) for axis in ("x","y","z","qx","qy","qz","qw")]+[(f"j{i}",2e-5) for i in range(4)]+[(f"v{i}",2e-7 if i < 3 else 2e-5) for i in range(10)]:
                    error=abs(cpu[key]-gpu[key])
                    if error > limit:
                        violations.append({"group":names[group],"case":case,"control":control,"field":key,
                                           "CPU":cpu[key],"CUDA":gpu[key],"error":error,"limit":limit})
                for key in ("joint_work","positive_work","relative_water_loss","external_work","numerical_loss","energy_error"):
                    assert abs(cpu[key]-gpu[key]) <= 1e-12+1e-4*max(abs(cpu[key]),abs(gpu[key]))
    assert finished == native["finished_cases"]
    assert len(violations) == 1, violations
    violation=violations[0]
    assert violation["group"] == "flow_zero_target_servo" and violation["case"] == 0 and violation["control"] == 77 and violation["field"] == "v1"
    first=envelope(gate/"run-started/first-failure.bin")
    assert first.value("I") == 1
    assert [first.value("i") for _ in range(4)] == [3,0,0,80]
    assert first.value("f") == 0
    near([first.value("f") for _ in range(3)],[float(np.float32(.0025)),0,0],0,0)
    assert first.value("i") == 77
    # Remaining records retain complete CPU/GPU/historical Step structs.
    assert len(first.data)-first.at == 3*1000
    seconds=float(read(gate/"run-started/active-seconds.txt"))
    near(seconds,native["active_seconds"],1e-12,1e-12)
    assert 0 < seconds < 900
    return {"closed_failure_verified":True,"controls":attempted,"substeps":attempted*10,
            "PPO_updates":0,"optimizer_steps":0,"finished_cases":finished,"assembly":assembly,
            "first_parity_violation":violation,"physical_step_faults":0,"qualified":False,
            "current_behavioral_suites_executed":False,"active_seconds":seconds}


def read_protocol(root):
    reader = envelope(root/"protocol.bin")
    protocol = shared.fnv(reader.data)
    assert [reader.value("I"), reader.value("I")] == [0x31505343,1]
    assert reader.value("Q") == shared.fnv(read(root/"approved-protocol.md"))
    parent_checksum, hold_checksum = reader.value("Q"), reader.value("Q")
    assert [reader.value("i") for _ in range(20)] == [6,1,3,5,44,41,4,16,336,7,256,160,32,512,10720,2560,58000,1466480,14664800,900]
    assert np.array_equal(np.float32([reader.value("f") for _ in range(9)]), np.float32([.5,.5,.002,.0025,.01,.5,.2,.01,8]))
    assert [reader.value("I"),reader.value("I")] == [0x36B91A57,0x84E62C13]
    for specs, duration in ((case_specs(),160), (case_specs(True),160), (np.asarray([[0,-.002,0,0],[3,0,0,.002]]),1280)):
        for pose, fx, fy, fz in specs:
            assert reader.value("i") == int(pose)
            near([reader.value("f"),reader.value("f")], [fx,fz], 2e-10, 1e-7)
            assert reader.value("i") == duration
    method = reader.array("u1").tobytes().decode(); reader.finish()
    assert "nativePPOonceepisode" in method and ".5*tanh(u)" in method
    return protocol, parent_checksum, hold_checksum


def read_scale(root, protocol, returns):
    reader = envelope(root/"fixed-scale.bin")
    assert [reader.value("I"),reader.value("I")] == [0x31535343,1] and reader.value("Q") == protocol
    scale = reader.value("d")
    means = np.asarray([reader.value("d") for _ in range(16)])
    assert [reader.value("i") for _ in range(16)] == [16]*16; reader.finish()
    expected_means = np.asarray([returns[case::16].mean() for case in range(16)])
    near(means, expected_means, 1e-12, 1e-12)
    expected_scale = float(np.sqrt(np.mean((returns-means[np.arange(N)%16])**2)))
    assert math.isfinite(scale) and scale > 0
    near(scale, expected_scale, 1e-12, 1e-12)
    return scale


def self_test():
    pose = np.arange(4)
    body = initial_body(pose)
    pre = np.zeros((4, 33)); pre[:, :21] = body; pre[:, 21:25] = body[:, 7:11]
    flow = np.array([[.001, 0, 0], [-.001, 0, 0], [0, 0, .002], [0, 0, -.002]])
    out = observations(pre, pose, flow, np.zeros(4))
    assert out.shape == (4, 44) and np.isfinite(out).all()
    assert np.max(abs(out[:, :3])) == 0 and np.max(abs(out[:, 12:18])) == 0
    assert np.all(out[:, 34:36] == 1) and np.all(out[:, 40] == 0)
    near(rotate(body[:, 3:7], out[:, 41:44]*FLOW_SCALE), flow, 1e-12, 1e-10)
    assert np.all(joint_latents(np.zeros((4, 16)), np.ones((4, 4)), out[:, 41:44]) == 1)
    assert abs(log_probability(np.zeros(16), np.zeros((1,16)))[0]+16*(math.log(SIGMA)+.5*math.log(2*math.pi))) < 1e-12
    advantages = leave_one_out(np.arange(N, dtype=float), 3)
    assert all(abs(advantages[case::16].mean()) < 1e-12 for case in range(16))
    print(json.dumps({"passed": True, "scope": "Independent verifier math only", "physical_controls": 0, "optimizer_steps": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", type=Path, default=Path("build/ppo/current-swim-2026-11-22-v1"))
    parser.add_argument("--helper", type=Path, default=Path("build/learned-amplitude-preparation-2026-11-22/check-evidence.exe"))
    parser.add_argument("--gate-failure", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.gate_failure or (args.run/"study-result.json").exists():
        root = args.run.resolve()
        report = {"passed": False, "qualified": False, "phase": "closed_physics_gate_failure",
                  "scope": "Independent saved-evidence verification; zero new physical or optimizer steps"}
        try:
            read(Path(__file__)); read(Path(shared.__file__))
            report["envelopes"] = validate_envelopes(root, args.helper)
            protocol, parent_checksum, hold_checksum = read_protocol(root)
            inherited = parents(root, protocol)
            assert inherited[3:] == (parent_checksum, hold_checksum)
            means, policy = read_policy(root/"initial-policy/current-policy-update-0.bin", protocol, *inherited)
            assert np.all(means == 0) and policy["Adam_steps"] == 0
            report["initial_policy"] = policy
            report["closed_gate"] = verify_gate_failure(root)
            for path,digest in list(HASHES.items()): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            report["passed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["input_sha256"] = HASHES
            (root/"independent-verification.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"passed":report["passed"],"phase":report["phase"],"qualified":False,
                          "controls":report["closed_gate"]["controls"],"input_files":len(HASHES)}))
    else:
        raise SystemExit("No closed current-swimming v1 result found. This attempt stopped at its physical gate; no calibration/PPO artifacts exist to certify.")
