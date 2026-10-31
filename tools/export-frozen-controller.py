"""Export actor parameters only into the current strict native contract. No stepping."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]

def checksum(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value

def payload(path):
    raw = path.read_bytes()
    magic, count, expected = struct.unpack_from("<QQQ", raw)
    data = raw[24:]
    if magic != 0x31504B434C524443 or count != len(data) or checksum(data) != expected:
        raise ValueError("Invalid source checkpoint checksum")
    return data

def export(args):
    source = Path(args.source).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest().lower() != args.sha256.lower():
        raise ValueError("Source identity differs from explicitly selected checkpoint")
    data = payload(source)
    magic, schema = struct.unpack_from("<2I", data)
    if magic == 0x46575331 and schema == 2:
        update = struct.unpack_from("<I", data, 16)[0]
        count = struct.unpack_from("<Q", data, 20)[0]
        binding = data[28:28 + count]
        pos = 28 + count
        mapping, protocol_checksum = 1, checksum(binding)
    elif magic == 0x32524E54 and schema == 1:
        _, _, mapping, protocol_checksum, update = struct.unpack_from("<IIIQI", data)
        if mapping != 2:
            raise ValueError("Wrong signed-cadence source mapping")
        pos = 24
    else:
        raise ValueError("Unknown research snapshot schema")
    count = struct.unpack_from("<Q", data, pos)[0]
    params = data[pos + 8:pos + 8 + count * 4]
    if count != 2498 or len(params) != 9992 or not (.004 <= args.minimum <= args.maximum <= .008):
        raise ValueError("Wrong actor shape or supported range")
    if not all(math.isfinite(value) for value in struct.unpack("<2498f", params)):
        raise ValueError("Nonfinite actor parameters")
    acceptance = None
    if mapping == 2:
        if not args.acceptance:
            raise ValueError("Signed adoption requires explicit acceptance provenance")
        acceptance = json.loads(Path(args.acceptance).read_text())
        if not acceptance.get("accepted") or acceptance.get("source_checkpoint_sha256", "").lower() != args.sha256.lower():
            raise ValueError("Acceptance does not identify this exact checkpoint")
    names = (ROOT / "src/nn/motor/controller_contract.txt").read_text().splitlines()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in names}
    portable = struct.pack("<9I2f2QI", 0x36434D46, 3, 6, 5, 2, 42, 2, update, mapping,
                           args.minimum, args.maximum, protocol_checksum, checksum(params), len(names))
    for digest in hashes.values():
        portable += struct.pack("<Q", 64) + digest.encode("ascii")
    portable += struct.pack("<Q", count) + params
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(struct.pack("<QQQ", 0x31504B434C524443, len(portable), checksum(portable)) + portable)
    metadata = dict(model=6, task=5, reward_version=2, frozen=True, training=False, source_update=update, action_mapping=mapping,
                    source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=args.sha256.upper(),
                    actor_parameters=count, actor_checksum_fnv64=f"{checksum(params):016x}",
                    source_protocol_checksum_fnv64=f"{protocol_checksum:016x}",
                    source_contract_sha256_normalized_lf=hashes, asset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    minimum_speed_m_s=args.minimum, maximum_speed_m_s=args.maximum,
                    viewer_arena_m=dict(width=1.2,depth=.8,height=.16), benchmark_arena_m=dict(width=.36,depth=.28,height=.16),
                    reset="Actor only; fresh physical state, no optimizer or saved world continuation",
                    speed_selection=("Persistent request preserves physical/filter state and running/paused status; 4-to8-to4 validated in heldout poses" if mapping == 2 else "Forward-only actor: changed requests reset to rest"),
                    continuous_viewing="Unvalidated beyond four seconds; unchanged inference with time feature held at zero; explicit reset at 20mm pond clearance, never safety-failure recovery")
    if acceptance is not None:
        metadata["acceptance_provenance"] = str(Path(args.acceptance).resolve().relative_to(ROOT))
        metadata["acceptance_sha256"] = hashlib.sha256(Path(args.acceptance).read_bytes()).hexdigest()
        metadata["original_automatic_selection"] = acceptance["original_automatic_selection"]
        metadata["user_accepted_work8_increase_percent"] = acceptance["work8_increase_percent"]
    target.with_name("controller-provenance.json").write_bytes((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(dict(exported_bytes=target.stat().st_size, actor_checksum=metadata["actor_checksum_fnv64"], physical_advances=0)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--minimum", type=float, required=True)
    parser.add_argument("--maximum", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--acceptance")
    export(parser.parse_args())
