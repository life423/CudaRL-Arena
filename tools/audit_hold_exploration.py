"""Offline saved-state exploration arithmetic; no physics or optimizer execution."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from audit_hold_saved import ROOT, RUN, TASK, read_batch, read_actor, forward


def interval_probability(mean, sd, bound):
    hi, lo = (bound - mean) / (sd * math.sqrt(2)), (-bound - mean) / (sd * math.sqrt(2))
    return np.prod(np.fromiter((.5 * (math.erf(a) - math.erf(b)) for a, b in zip(hi.ravel(), lo.ravel())), float).reshape(mean.shape), axis=1)


def main():
    home = RUN / "phase-B-started/B20_60"
    data_path, actor_path, rows_path = home / "batch-338.bin", home / "actor-only-update-338.bin", home / "rows-338.csv"
    paths = [data_path, actor_path, rows_path, Path(__file__)]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data = read_batch(data_path)
    params, _ = read_actor(actor_path)
    mu, previous_mu = forward(params, data["obs"]), forward(params, data["prevobs"])
    reset = (TASK == 1) & (data["history"] == 0)
    hold = TASK == 1
    threshold = .03
    bound = np.arctanh(threshold)
    sigmas = []
    for sigma in [math.exp(-1), .5, .75, 1., 1.5, 2., 3.]:
        probability = interval_probability(mu[reset], sigma, bound)
        sigmas.append({"sigma": sigma, "all4_within_0p03_at_saved_resets_mean_probability": float(probability.mean()),
                       "all4_within_0p03_at_saved_resets_max_probability": float(probability.max())})
    conditional = mu + np.float32(.8) * data["history"][:, None] * (data["prevaction"] - previous_mu)
    std = np.exp(np.float32(-1)) * np.sqrt(1 - np.float32(.8) ** 2 * data["history"])
    # Integrate the conditional four-dimensional probability over log-uniform gain.
    # Recorded state/history are held fixed algebraically; no future trajectory is inferred.
    grid = np.linspace(math.log(.01), 0, 257)
    probabilities = []
    for log_gain in grid:
        gain = math.exp(log_gain)
        probabilities.append(interval_probability(conditional[hold] * gain, std[hold, None] * gain, bound))
    integrated = np.trapezoid(np.array(probabilities), x=grid, axis=0) / math.log(100)
    fixed_gains = []
    rows = list(csv.DictReader(rows_path.open(newline="")))
    hold_worlds = [world for world in range(32) if (world // 2) % 2 == 1]
    for gain in [.01, .03, .1, 1.]:
        quiet = np.max(abs(np.tanh(gain * data["action"])), axis=1) <= threshold
        successful_segments, segments, quiet_rows, longest = 0, 0, 0, 0
        for world in hold_worlds:
            segment, run = [], 0
            for t in range(320):
                row = t * 32 + world
                segment.append(bool(quiet[row]))
                if quiet[row]: run += 1; quiet_rows += 1; longest = max(longest, run)
                else: run = 0
                if float(rows[row]["done"]):
                    segments += 1; successful_segments += all(segment)
                    segment = []; run = 0
            assert not segment
        fixed_gains.append({"gain": gain, "all4_within_0p03_rows": quiet_rows,
                            "complete_recorded_segments_within_0p03": successful_segments,
                            "segments": segments, "longest_controls": longest})
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert after == before
    result = {"scope": "Saved weights, network inference, algebra and one-dimensional probability quadrature only; no physics/optimizer",
              "source": str(home.relative_to(ROOT)), "threshold": threshold,
              "hold_reset_mean_abs_latent_by_axis": np.mean(abs(mu[reset]), axis=0).tolist(),
              "hold_reset_mean_latent_by_axis": np.mean(mu[reset], axis=0).tolist(),
              "hold_reset_max_abs_latent": float(np.max(abs(mu[reset]))),
              "std_only_at_saved_reset_observations": sigmas,
              "gain_mixture": {"law": "LogUniform[0.01,1], fixed for a true holding segment",
                               "quadrature_points": len(grid), "saved_conditional_expected_quiet_rows": float(integrated.sum()),
                               "saved_conditional_mean_probability": float(integrated.mean()),
                               "gain_probability_at_most_0p03": math.log(3) / math.log(100),
                               "mean_gain_squared": (1 - .01 ** 2) / (2 * math.log(100))},
              "algebraic_rescaling_of_recorded_latents_not_policy_replay": fixed_gains,
              "caveats": ["Any changed sampler changes subsequent states; this analysis gives no physical success prediction.",
                          "Instantaneous interval probability is not probability of a sustained successful future sequence.",
                          "Low-gain stochastic success is insufficient; deterministic evaluation must use gain1 and learned means.",
                          "Keeping pretrained weights fixed and learning four conditional gains restricts capacity; it is not general braking or recovery."],
              "input_hashes": before, "inputs_unchanged": True}
    out = ROOT / "build/audit-hold-exploration-2026-11-22"
    out.mkdir(exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
