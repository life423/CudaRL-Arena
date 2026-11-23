"""Read-only design audit of saved pond data; no simulation or learning.

The arithmetic examples are objective counterexamples, not simulated creatures.
The proposed fixed-horizon utility is an offline specification, not an adopted
reward, policy, protocol, or authorization for another experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import verify_continuous_pond as saved


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded_motion_utility(forward_speed, side_speed=0., depth_error=0.,
                           roll_degrees=0., pitch_degrees=0., positive_power=0.):
    """Proposed utility in [0,1]; moving forward is essential to earn it."""
    quality_cost = (.25 * (side_speed / .008) ** 2
                    + .05 * (depth_error / .001) ** 2
                    + .05 * (roll_degrees / 10.) ** 2
                    + .01 * (pitch_degrees / 10.) ** 2
                    + 6250. * max(0., positive_power))
    return float(np.clip(forward_speed / .008, 0., 1.)) / (1. + quality_cost)


def fixed_horizon_return(utility_integral, elapsed, failed, horizon=60.):
    assert 0 <= elapsed <= horizon and 0 <= utility_integral <= elapsed
    # Account for unexecuted failure time in the objective only. Do not advance
    # or invent a body state, physical control, action, or measurement.
    return (utility_integral - ((horizon - elapsed) + 4. if failed else 0.)) / horizon


def objective_checks():
    tested = 0
    for fail_at in (0., .005, 4., 10., 30., 59.95):
        for prefix_rate in (0., .1, .5, 1.):
            prefix = fail_at * prefix_rate
            early = fixed_horizon_return(prefix, fail_at, True)
            for later in (fail_at, (fail_at + 60.) / 2., 60.):
                for later_rate in (0., .1, .5, 1.):
                    total = prefix + (later - fail_at) * later_rate
                    assert fixed_horizon_return(total, later, True) >= early - 1e-12
                    assert fixed_horizon_return(total, 60., False) > early
                    tested += 2
    speeds = np.linspace(-.008, .016, 101)
    utilities = [bounded_motion_utility(x) for x in speeds]
    assert all(0 <= u <= 1 for u in utilities)
    assert all(a <= b for a, b in zip(utilities, utilities[1:]))
    assert bounded_motion_utility(0.) == 0
    assert bounded_motion_utility(.008) == 1
    assert bounded_motion_utility(.008, depth_error=.002) < 1
    return {'same_prefix_return_inequalities_checked': tested,
            'bounded_speed_samples': len(speeds), 'passed': True}


def audit(root, output):
    run = root / 'build/ppo/continuous-pond-2026-11-23-v1'
    # Validate the exact envelopes we read. Do not mark an unchecked native
    # payload as verified, or execute the simulator to recreate it.
    for path in [run / 'protocol.bin', *sorted((run / 'baseline-free').glob('*.bin'))]:
        raw = path.read_bytes()
        magic, size, checksum = struct.unpack_from('<QQQ', raw)
        assert magic == 0x31504b434c524443 and size == len(raw) - 24
        assert saved.reference.shared.fnv(raw[24:]) == checksum
        saved.VALIDATED.add(path.resolve())
    protocol, _, _ = saved.read_protocol(run)
    layout = saved.js(run / 'layout.json')
    batch = saved.open_batch(run / 'baseline-free', protocol, layout)
    assert np.all(batch['means'] == 0), 'Authority statistics require exact zero-head baseline'
    active = batch['steps'] > 0
    base = np.asarray(batch['parts']['joint-latent'])[active]
    low, high = np.tanh(base - .75), np.tanh(base + .75)
    authority = []
    for j in range(4):
        authority.append({'axis': j, 'recorded_active_controls': len(base),
                          'base_latent_min': float(base[:, j].min()),
                          'base_latent_max': float(base[:, j].max()),
                          'target_sign_unreachable_fraction': float(np.mean(abs(base[:, j]) > .75)),
                          'median_normalized_target_interval_width': float(np.median(high[:, j] - low[:, j]))})
    paths = [Path(__file__), root / 'src/nn/motor/turning_policy_v2.h',
             root / 'src/nn/motor/turning_task_v2.h', run / 'layout.json',
             run / 'protocol.bin', run / 'baseline-free/meta.bin',
             run / 'baseline-free/joint-latent.bin', run / 'baseline-free/row-substeps.bin']
    report = {'schema': 1, 'scope': 'Offline design audit of saved data and arithmetic only',
              'physical_controls': 0, 'optimizer_steps': 0,
              'inputs_sha256': {str(p.relative_to(root)): sha(p) for p in paths},
              'v2_recommendation': 'Do not execute the prepared proposal; repair the objective and turn lesson first',
              'authority_on_saved_zero_head_local_forward_baseline': authority,
              'authority_limit': 'A +/-.75 latent residual cannot cross zero when abs(base) > .75. This is not proof that turning is impossible.',
              'objective_counterexample': {
                  'assumptions': 'Algebra only: same prefix, constant valid rate -2, equal -4 failure cost, zero effort; not a physical rollout',
                  'fail_now_unnormalized_return': -4.,
                  'continue_four_seconds_then_fail_unnormalized_return': -12.,
                  'v2_prefers_early_failure': True,
                  'cause': 'Unexecuted episode time contributes zero while live utility can be negative'},
              'proposed_objective': {
                  'status': 'Offline specification only; not wired into a trainer',
                  'live_utility_bounds': [0, 1], 'unexecuted_failure_rate': -1,
                  'horizon_seconds': 60, 'failure_cost': 4,
                  'same_prefix_property': 'Any legal continuation improves over failing now; this does not prove learnability or qualification',
                  'checks': objective_checks()},
              'interpretation_limits': [
                  'A frozen old-mode bypass proves those old routes are retained, not that free-swim retains their stability.',
                  'The local-forward observer removes fixed heading and horizontal lane restoration signals.',
                  'No saved trace establishes a clean repeated 90-degree turn within the final speed/depth/tilt limits.',
                  'This audit does not authorize physical probes, training, adoption, a budget extension, or a simulated navigation script.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'output': str(output), 'physical_controls': 0,
                      'optimizer_steps': 0, 'recommendation': report['v2_recommendation']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(Path(__file__).resolve().parents[1], args.output)
