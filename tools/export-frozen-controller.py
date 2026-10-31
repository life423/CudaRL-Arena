"""Export actor parameters only into the current strict native contract. No stepping."""
import argparse
import hashlib
import json
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
    magic, schema, _, _, update = struct.unpack_from("<5I", data)
    if magic != 0x46575331 or schema != 2:
        raise ValueError("Unknown research snapshot schema")
    count = struct.unpack_from("<Q", data, 20)[0]
    binding = data[28:28 + count]
    pos = 28 + count
    count = struct.unpack_from("<Q", data, pos)[0]
    params = data[pos + 8:pos + 8 + count * 4]
    if count != 2498 or len(params) != 9992 or not (.004 <= args.minimum <= args.maximum <= .008):
        raise ValueError("Wrong actor shape or supported range")
    names = (ROOT / "src/motor/controller_contract.txt").read_text().splitlines()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in names}
    portable = struct.pack("<8I2f2QI", 0x36434D46, 2, 6, 5, 2, 42, 2, update,
                           args.minimum, args.maximum, checksum(binding), checksum(params), len(names))
    for digest in hashes.values():
        portable += struct.pack("<Q", 64) + digest.encode("ascii")
    portable += struct.pack("<Q", count) + params
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(struct.pack("<QQQ", 0x31504B434C524443, len(portable), checksum(portable)) + portable)
    metadata = dict(model=6, task=5, reward_version=2, frozen=True, training=False,
                    source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=args.sha256.upper(),
                    actor_parameters=count, actor_checksum_fnv64=f"{checksum(params):016x}",
                    source_protocol_checksum_fnv64=f"{checksum(binding):016x}",
                    source_contract_sha256_normalized_lf=hashes, asset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    minimum_speed_m_s=args.minimum, maximum_speed_m_s=args.maximum,
                    reset="Actor only; fresh physical state, no optimizer or saved world continuation")
    target.with_name("controller-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(exported_bytes=target.stat().st_size, actor_checksum=metadata["actor_checksum_fnv64"], physical_advances=0)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--minimum", type=float, required=True)
    parser.add_argument("--maximum", type=float, required=True)
    parser.add_argument("--output", required=True)
    export(parser.parse_args())
