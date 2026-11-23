"""Write the version-two grant only AFTER an explicit fresh user approval.

This data-only utility never starts physics/training. The execution owner must
provide the actual approval response in a UTF-8 file; this utility must never
be invoked with generated, inferred, historical, or placeholder approval text.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):value.update(part)
    return value.hexdigest()


def fnv(raw):
    value=14695981039346656037
    for byte in raw:value=((value^byte)*1099511628211)&0xffffffffffffffff
    return value


def payload(path):
    raw=Path(path).read_bytes();magic,size,checksum=struct.unpack_from('<QQQ',raw)
    assert magic==0x31504b434c524443 and size==len(raw)-24 and fnv(raw[24:])==checksum
    return raw[24:]


def require_not_held(root):
    if (root/'design-hold.json').exists():
        raise RuntimeError('This prepared protocol is on design hold and cannot be authorized')
    for name in ('prepared.json','protocol-status.json'):
        path=root/name
        if not path.exists():continue
        state=json.loads(path.read_bytes())
        held=state.get('held') is True or state.get('design_hold') is True
        status=str(state.get('protocol_status',state.get('status',''))).strip().lower()
        if held or status in ('held','design_hold','design-hold'):
            raise RuntimeError('This prepared protocol is marked held and cannot be authorized')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--approval-message-file',type=Path,required=True)
    parser.add_argument('--confirm-explicit-user-grant',action='store_true',required=True)
    args=parser.parse_args();root=args.run.resolve();message_path=args.approval_message_file.resolve()
    assert args.confirm_explicit_user_grant and root.is_dir() and message_path.is_file()
    require_not_held(root)
    target=root/'authorization.bin';receipt=root/'authorization.json'
    assert not target.exists() and not receipt.exists(),'Never overwrite an earlier authorization identity'
    for name in ('gate-started','gate-result.json','baseline-started','calibration-started','training-started'):
        assert not (root/name).exists(),'The fresh grant must precede every physical phase'
    message=message_path.read_bytes();assert 0<len(message)<=65536 and message.decode('utf-8').strip()
    protocol_bytes=payload(root/'protocol.bin');protocol=fnv(protocol_bytes)
    assert struct.unpack_from('<II',protocol_bytes)==(0x32525054,2)
    assert struct.unpack_from('<Q',protocol_bytes,8)[0]==fnv((root/'approved-protocol.md').read_bytes())
    integers=struct.unpack_from('<26i',protocol_bytes,32)
    assert integers==tuple([6,2,3,4,2,41,20,4,336,80,2048,1200,24,384,4,512,2480,10720,46240,16,1200,8,36000,62286640,622866400,900])
    preparation_path=root/'independent-preparation-verification.json'
    prepared=json.loads(preparation_path.read_bytes())
    assert prepared['schema']==2 and prepared['phase']=='preparation' and prepared['passed'] and not prepared['qualified']
    assert prepared['physical_run_approved'] is False
    mutable=(Path(__file__).resolve().parent/'verify_continuous_pond_v2.py').resolve()
    for source in prepared['auditor_sources']:assert sha(source['snapshot'])==source['sha256']
    for path,digest in prepared['input_sha256'].items():
        if Path(path).resolve()!=mutable:assert sha(path)==digest,path
    freeze_path=root/'frozen-inputs.json';frozen=json.loads(freeze_path.read_bytes())
    entries=frozen['files'] if isinstance(frozen,dict) else frozen
    assert entries
    for entry in entries:
        assert sha(entry['path'])==entry['sha256'].lower(),entry['path']
        if 'snapshot' in entry:assert sha(entry['snapshot'])==entry['sha256'].lower()
    body=struct.pack('<IIQQ',0x32554150,1,protocol,len(message))+message
    artifact=struct.pack('<QQQ',0x31504b434c524443,len(body),fnv(body))+body
    record={'schema':1,'study':'continuous_pond_v2','explicit_user_grant_recorded':True,
        'protocol_checksum':f'{protocol:016X}','protocol_sha256':sha(root/'protocol.bin'),
        'approval_message_file':str(message_path),'approval_message_sha256':sha(message_path),
        'prepared_verification_sha256':sha(preparation_path),'frozen_inputs_manifest_sha256':sha(freeze_path),
        'authorization_sha256':hashlib.sha256(artifact).hexdigest(),'writer_sha256':sha(__file__),
        'physical_controls_executed':0,'optimizer_steps_executed':0}
    with target.open('xb') as stream:stream.write(artifact);stream.flush();os.fsync(stream.fileno())
    with receipt.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps({'authorization_written':True,'protocol_checksum':record['protocol_checksum'],
        'authorization_sha256':record['authorization_sha256'],'physical_controls':0,'optimizer_steps':0}))


if __name__=='__main__':main()
