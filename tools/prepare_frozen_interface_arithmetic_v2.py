"""Export immutable old PRE states for zero-time arithmetic; never simulate."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import verify_frozen_requested_turn_result as old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root, output = args.old_run.resolve(), args.output.resolve()
    assert not output.exists()
    rows, _ = old.records(root, json.loads((root/'layout.json').read_bytes()))
    source = json.loads((root/'result.json').read_bytes())
    assert source['controls'] == 214 and source['cases_started'] == 0
    parts = []
    for backend in (0, 1):
        samples = rows[(rows['group'] == 0) & (rows['case'] == 0) & (rows['backend'] == backend)]
        assert len(samples) == 108 and np.array_equal(samples['tick'], np.arange(108))
        for tick in range(1, 108):
            pre, post = samples[tick-1], samples[tick]
            for density in (975., 1000., 1025.):
                values = np.concatenate([pre['world']['body'], pre['world']['held'], pre['world']['rates'], post['command'], [density, .001, 9.81]]).astype('<f4')
                assert len(values) == 38 and np.isfinite(values).all()
                parts.append(values.tobytes() + struct.pack('<ii', tick, backend))
    assert len(parts) == 642 and all(len(part) == 160 for part in parts)
    payload = struct.pack('<IIIIQ', 0x32414946, 2, len(parts), 160, len(parts)) + b''.join(parts)
    encoded = struct.pack('<QQQ', 0x31504b434c524443, len(payload), old.shared.fnv(payload)) + payload
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded)
    report = {'schema': 1, 'rows': len(parts), 'densities': [975, 1000, 1025],
              'source_states_sha256': hashlib.sha256((root/'probe-started/states.bin').read_bytes()).hexdigest(),
              'source_layout_sha256': hashlib.sha256((root/'layout.json').read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(encoded).hexdigest(),
              'input_order': 'backend, control1..107, density975/1000/1025; both evaluators receive exactly these same PRE inputs',
              'physical_controls': 0, 'physical_substeps': 0, 'GPU_kernels': 0, 'optimizer_steps': 0}
    with output.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
