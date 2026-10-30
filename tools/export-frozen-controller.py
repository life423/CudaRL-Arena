"""Export only the verified winning actor; no simulation or training."""
from pathlib import Path
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/straight-motor/2026-10-30-task4-steady-reward-v2"
EXPECTED = "31C9982778F32A5E0F421CDBA88AF4431ED27431A355C66D9E290D60EDC97092"
CONTRACTS = ["src/motor/movement_model.h", "src/motor/stroke_motor.h", "src/motor/straight_task.h", "src/motor/forward_task.h", "src/physics/dynamics.h", "src/physics/contact.h", "src/simulation/world.h", "src/nn/mlp_cpu.h", "src/nn/mlp_spec.h"]

def checksum(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value

def payload(path):
    raw = path.read_bytes()
    magic, count, expected = struct.unpack_from("<QQQ", raw)
    result = raw[24:]
    assert magic == 0x31504B434C524443 and count == len(result) and checksum(result) == expected
    return result

raw = (SOURCE / "update-25.bin").read_bytes()
assert hashlib.sha256(raw).hexdigest().upper() == EXPECTED
data = payload(SOURCE / "update-25.bin")
magic, schema, world_size, task_size, update = struct.unpack_from("<5I", data)
assert magic == 0x46575331 and schema == 1 and update == 25
pos = 20
size = struct.unpack_from("<Q", data, pos)[0]
pos += 8
binding = data[pos:pos + size]
pos += size
assert binding == payload(SOURCE / "protocol.bin")
count = struct.unpack_from("<Q", data, pos)[0]
pos += 8
params = data[pos:pos + count * 4]
assert count == 2498 and checksum(params) == 0x1699F42D0090C042
portable = struct.pack("<10I", 0x36434D46, 1, 6, 4, 2, 42, 2, 25, world_size, task_size)
portable += struct.pack("<fQQI", .008, checksum(binding), checksum(params), len(CONTRACTS))
hashes = {}
for name in CONTRACTS:
    current = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    assert current == (SOURCE / "source" / name).read_bytes().replace(b"\r\n", b"\n"), name
    digest = hashlib.sha256(current).hexdigest()
    hashes[name] = digest
    portable += struct.pack("<Q", 64) + digest.encode("ascii")
portable += struct.pack("<Q", count) + params
asset = ROOT / "live/assets/frozen-controller.bin"
asset.parent.mkdir(parents=True, exist_ok=True)
asset.write_bytes(struct.pack("<QQQ", 0x31504B434C524443, len(portable), checksum(portable)) + portable)
metadata = dict(model=6, task=4, reward_version=2, frozen=True, training=False,
    source_checkpoint=str(SOURCE.relative_to(ROOT) / "update-25.bin"), source_checkpoint_sha256=EXPECTED,
    actor_parameters=count, actor_checksum_fnv64=f"{checksum(params):016x}",
    source_protocol_checksum_fnv64=f"{checksum(binding):016x}",
    source_contract_sha256_normalized_lf=hashes, asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
    reset="Fresh frozen evaluation fixtures, not saved training-world continuation")
(asset.parent / "controller-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Exported verified actor: {asset.stat().st_size} bytes; no physical advances")
