"""Export an exact frozen research actor for explicit native GUI inspection.

Parses existing native checkpoints only. No training or physical integration.
The original runtime policy package and personal saves are never written.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = '810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0'

def fnv(data):
    value=14695981039346656037
    for byte in data:value=((value^byte)*1099511628211)&0xffffffffffffffff
    return value

class Reader:
    def __init__(self,path):
        data=path.read_bytes();magic,size,checksum=struct.unpack_from('<QQQ',data)
        assert magic==0x31504b434c524443 and size==len(data)-24 and fnv(data[24:])==checksum
        self.data=data[24:];self.at=0
    def value(self,kind):
        size=struct.calcsize('<'+kind);v=struct.unpack_from('<'+kind,self.data,self.at)
        self.at+=size;return v[0]
    def array(self,kind):
        count=self.value('Q');size=struct.calcsize('<'+kind)
        assert count<=1000000 and self.at+count*size<=len(self.data)
        result=struct.unpack_from('<'+str(count)+kind,self.data,self.at);self.at+=count*size
        return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['auto','four-axis-gains','learned-amplitude'],default='auto')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    default_run='hold-exploration-2026-11-22-v1' if args.method=='four-axis-gains' else 'learned-amplitude-2026-11-22-v1'
    run=(args.run or ROOT/'build/ppo'/default_run).resolve()
    result=json.loads((run/'training-result.json').read_text())
    assert result['closed'] and not result['fault']
    assert hashlib.sha256((run/'parent-checkpoint.bin').read_bytes()).hexdigest().upper()==CANONICAL
    verification=json.loads((run/'independent-verification.json').read_text())
    assert verification['passed'] and verification['phase']=='closed_attempt'
    updates=result['PPO_updates'];assert 0<updates<=32
    amplitude_policy=run/'training-started'/f'amplitude-policy-update-{updates}.bin'
    learned=amplitude_policy.is_file() if args.method=='auto' else args.method=='learned-amplitude'
    policy=amplitude_policy if learned else run/'training-started'/f'exploration-policy-update-{336+updates}.bin'
    r=Reader(policy)
    assert r.value('I')==(0x314d4148 if learned else 0x32454841) and r.value('I')==1
    protocol=r.value('Q');parent_checksum=r.value('Q')
    assert protocol==fnv(Reader(run/'protocol.bin').data)
    assert parent_checksum==fnv(Reader(run/'parent-checkpoint.bin').data)
    if learned:
        assert tuple(r.value('I') for _ in range(7))==(6,6,3,4,41,4,336)
        update=r.value('i');steps=r.value('i')
        parameters=r.array('f');head=r.array('f');gain=r.value('f');sigma=r.value('f');r.array('B')
        assert len(head)==2 and update==updates
        expected_gain=1/(1+math.exp(-head[0])) if head[0]>=0 else math.exp(head[0])/(1+math.exp(head[0]))
        variance_sigmoid=1/(1+math.exp(-head[1])) if head[1]>=0 else math.exp(head[1])/(1+math.exp(head[1]))
        assert abs(gain-expected_gain)<2e-7 and abs(sigma-(.05+7.95*variance_sigmoid))<2e-6
        assert 0<=gain<=1 and .05<=sigma<=8
        gains=(gain,)*4
    else:
        update=r.value('i');steps=r.value('i');assert r.value('I')==41 and r.value('I')==4
        parameters=r.array('f');gains=r.array('f');r.array('B')
        assert len(gains)==4 and update==336+updates
    assert r.at==len(r.data) and len(parameters)==2532
    assert all(math.isfinite(x) for x in parameters+gains)
    assert steps==result['actor_steps']
    policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    inputs=verification.get('input_sha256',{})
    expected_sha={key.casefold():value for key,value in inputs.items()}.get(str(policy).casefold())
    if expected_sha is not None:assert expected_sha.lower()==policy_sha
    original_holds=withheld_holds=endurance_holds=short_swims=retained_long_windows=0
    if learned:
        assert expected_sha is not None,'Independent evidence must bind the exact learned checkpoint'
        final=verification['updates'][-1]['assessment']
        original_holds=final['hold']['passes'];short_swims=final['short_passes'];retained_long_windows=final['long_passes']
        withheld_holds=verification.get('reserved_withheld',{}).get('passes',0)
        endurance_holds=verification.get('reserved_endurance',{}).get('passes',0)
        if result['qualified']:
            assert result['original_success'] and original_holds==6 and withheld_holds==6 and endurance_holds==2
            assert short_swims==4 and retained_long_windows==28 and final['retained']
    payload=bytearray()
    def put(kind,*values):payload.extend(struct.pack('<'+kind,*values))
    def array(kind,values):put('Q',len(values));put(str(len(values))+kind,*values)
    if learned:put('12I',0x31564c48,2,6,6,3,4,41,4,336,1,updates,int(result['qualified']))
    else:put('11I',0x31564c48,1,6,6,3,4,41,4,336,updates,int(result['qualified']))
    put('Q',protocol);array('B',CANONICAL.encode('ascii'))
    contracts=(ROOT/'src/nn/motor/controller_contract.txt').read_text().splitlines()
    contracts=[s.strip() for s in contracts if s.strip()]
    put('I',len(contracts))
    for relative in contracts:
        digest=hashlib.sha256((ROOT/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        array('B',digest.encode('ascii'))
    array('f',parameters)
    if learned:array('f',head);put('2f',gain,sigma)
    else:array('f',gains)
    default_output='learned-amplitude-v1.bin' if learned else 'latest-policy.bin'
    output=(args.output or ROOT/'live/research_data'/default_output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise SystemExit('Research package already exists; choose a new output explicitly')
    output.write_bytes(struct.pack('<QQQ',0x31504b434c524443,len(payload),fnv(payload))+payload)
    provenance={'scope':'Frozen research inspection; no runtime training or production adoption',
                'parent_sha256':CANONICAL,'source_checkpoint':str(policy),
                'source_checkpoint_sha256':policy_sha.upper(),
                'package_sha256':hashlib.sha256(output.read_bytes()).hexdigest().upper(),
                'holding_updates':updates,'coefficient_Adam_steps':steps,'PPO_Adam_steps':steps,
                'qualified':result['qualified'],'gains':gains,'backbone_frozen':True,
                'family':'learned-amplitude' if learned else 'four-axis-gains',
                'policy_method':'learned_persistent_amplitude' if learned else 'four_axis_hold_gains',
                'original_holds':original_holds,'withheld_holds':withheld_holds,'endurance_holds':endurance_holds,
                'endurance_seconds':300 if learned and endurance_holds else 0,
                'short_swims':short_swims,'retained_long_windows':retained_long_windows,
                'holding_parameter_count':2 if learned else 4,
                'deterministic_amplitude':gain if learned else 1,'action_noise':False,
                'runtime_training':False,'runtime_adopted':False}
    if learned:provenance.update(mu=head[0],variance_coordinate=head[1],sigma=sigma)
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'exported':str(output),'holding_updates':result['PPO_updates'],'qualified':result['qualified']}))

if __name__=='__main__':main()
