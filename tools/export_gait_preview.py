"""Stage a V3 gait package only after closed, independent full qualification.

Reads saved native data only. Never integrates physics, trains, selects a live
package, overwrites a package, or changes the active Godot project.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from export_hold_preview import fnv, CANONICAL
from export_current_preview import ROOT, HOLD_SHA, require, sha, key, f32, finite, parent_backbone, holding_parent, rows

GAIT_CONTRACTS = ["src/nn/motor/gait_policy_v1.h", "src/nn/motor/gait_task_v1.h"]
QUALIFICATION = {"short_movement_passes":4,"short_gait_passes":4,
    "long60_retained_windows":28,"long60_gait_windows":28,"recovery_passes":3,
    "long300_retained_windows":148,"long300_gait_windows":148,
    "original_hold_passes":6,"withheld_hold_passes":6,"hold300_passes":2}


class Reader:
    def __init__(self,path):
        raw=path.read_bytes()
        require(len(raw)>=24,"Truncated checkpoint envelope")
        magic,size,checksum=struct.unpack_from("<QQQ",raw)
        require(magic==0x31504B434C524443 and size==len(raw)-24 and fnv(raw[24:])==checksum,
                "Checkpoint envelope integrity differs")
        self.data=raw[24:];self.at=0

    def value(self,kind):
        size=struct.calcsize("<"+kind)
        require(self.at+size<=len(self.data),"Truncated checkpoint value")
        result=struct.unpack_from("<"+kind,self.data,self.at)[0];self.at+=size
        return result

    def array(self,kind):
        count=self.value("Q");size=struct.calcsize("<"+kind)
        require(count<=1000000 and self.at+count*size<=len(self.data),"Invalid checkpoint array")
        result=struct.unpack_from("<"+str(count)+kind,self.data,self.at);self.at+=count*size
        return result


def validate_gait(row):
    values=[float(row[name]) for name in ("mean_pitch1_deg","mean_pitch3_deg","mean_tail_vertical_bend_deg","centered_joint_RMS_deg")]
    finite(values)
    require(all(abs(x)<=10 for x in values[:3]) and values[3]>=5 and int(row["gait_passed"])==1,
            "Balanced moving gait acceptance changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--verification",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"build/pond-world-gait-2026-11-22/private/gait-trim-v1.bin")
    args=parser.parse_args();run=args.run.resolve()
    result_path=run/"training-result.json"
    native=json.loads(result_path.read_text())
    require(native.get("schema")==1 and native.get("study")=="gait_pond_v1" and
            native.get("closed") is True and native.get("qualified") is True,
            "Refusing export: gait study is not closed and qualified")
    require(not native.get("fault") and native.get("gate")=="qualified_balanced_gait_no_adoption" and
            native.get("runtime_adopted") is False and native.get("retries")==0,
            "Refusing export: gait fault, qualification or adoption metadata differs")
    proof_path=(args.verification or run/"independent-verification.json").resolve()
    proof=json.loads(proof_path.read_text())
    require(proof.get("passed") is True and proof.get("qualified") is True and proof.get("phase")=="training",
            "Refusing export: full independent gait qualification is missing")
    require(all(proof.get("qualification",{}).get(name)==wanted for name,wanted in QUALIFICATION.items()),
            "Independent retained-skill qualification differs")
    inputs={key(path):str(digest).upper() for path,digest in proof.get("input_sha256",{}).items()}

    def bound(path):
        path=path.resolve()
        require(inputs.get(key(path))==sha(path),f"Verified input changed or is unbound: {path.name}")
        return path

    bound(result_path)
    updates,steps=native["PPO_updates"],native["actor_steps"]
    require(type(updates) is int and 1<=updates<=64 and type(steps) is int and updates*4<=steps<=updates*16,
            "Invalid native gait optimizer counters")
    require(native.get("worlds")==8192 and native.get("critic_steps")==native.get("critic_forward_calls")==0,
            "Native gait capacity or critic counters changed")
    require(type(native.get("controls")) is int and 0<native["controls"]<=85295280 and
            native.get("substeps")==native["controls"]*10,"Native gait physical counters changed")
    for name,wanted in (("short_movement_passes",4),("short_gait_passes",4),("recovery_passes",3),
                        ("original_hold_passes",6),("withheld_hold_passes",6),("hold300_passes",2)):
        require(native.get(name)==wanted,f"Required retained-skill count differs: {name}")
    require(28<=native.get("long60_passing_windows",-1)<=30 and 148<=native.get("long300_passing_windows",-1)<=150,
            "Native long-swim retention counts differ")
    parent_path=bound(run/"parent-checkpoint.bin");hold_path=bound(run/"holding-parent.bin")
    protocol_path=bound(run/"protocol.bin")
    protocol=fnv(Reader(protocol_path).data)
    parent_checksum,hold_checksum=fnv(Reader(parent_path).data),fnv(Reader(hold_path).data)
    backbone=parent_backbone(parent_path)
    hold_head,hold_gain,hold_sigma=holding_parent(hold_path,backbone,parent_checksum)
    policy_path=bound(run/"training-started"/f"gait-policy-update-{updates}.bin")
    require(str(proof.get("final_policy_sha256","")).upper()==sha(policy_path),"Final candidate SHA differs from the independent proof")
    r=Reader(policy_path)
    require([r.value("I"),r.value("I")]==[0x31504747,1],"Wrong gait checkpoint schema")
    require([r.value("Q") for _ in range(3)]==[protocol,parent_checksum,hold_checksum],"Gait checkpoint parents/protocol differ")
    require([r.value("I") for _ in range(8)]==[6,1,3,4,41,4,336,8],"Wrong gait model/task/action dimensions")
    require([r.value("i") for _ in range(3)]==[8192,updates,steps],"Gait checkpoint counters differ")
    require(r.array("f")==backbone and r.array("f")==hold_head and r.value("f")==hold_gain,
            "The preserved swimming/holding policy changed")
    means=r.array("f")
    require(len(means)==8 and [r.value("f") for _ in range(3)]==[f32(.35),.75,2.],"Wrong gait distribution/mapping")
    finite(means)
    adam=bytes(r.array("B"));require(r.at==len(r.data),"Unexpected gait checkpoint data")
    require(len(adam)==104,"Wrong gait Adam payload size")
    require(struct.unpack_from("<5f",adam)==tuple(f32(x) for x in (.01,.9,.999,.00001,.5)),"Wrong gait Adam configuration")
    require(struct.unpack_from("<i",adam,20)[0]==steps,"Gait Adam age differs")
    moments=[];offset=24
    for _ in range(2):
        require(struct.unpack_from("<Q",adam,offset)[0]==8,"Wrong gait Adam moment dimensions");offset+=8
        moments.append(struct.unpack_from("<8f",adam,offset));offset+=32
    finite(moments[0]+moments[1]);require(all(x>=0 for x in moments[1]),"Negative gait Adam second moment")

    short=rows(bound(run/"training-started"/f"assessment-{updates}"/"summary.csv"))
    require(len(short)==4 and {int(row["pose"]) for row in short}=={0,1,2,3},"Wrong final short-swim cases")
    for row in short:
        require(int(row["controls"])==160 and int(row["movement_passed"])==1,"Original short-swim acceptance changed")
        validate_gait(row)
    for seconds in (60,300):
        windows=rows(bound(run/f"final-skills/long{seconds}/windows.csv"))
        require(len(windows)==seconds//4*2,"Wrong long-swim window count")
        keys={(int(row["pose"]),int(float(row["end_s"]))):row for row in windows}
        require(len(keys)==len(windows),"Repeated long-swim window")
        for pose in (0,3):
            for end in range(8,seconds+1,4):
                row=keys[(pose,end)]
                require(int(row["movement_passed"])==int(row["gait_scored"])==1,"A retained long-swim window was lost")
                validate_gait(row)
    recovery=rows(bound(run/"final-skills/recovery64/summary.csv"))
    require(len(recovery)==3 and all(int(row["controls"])==1280 and int(row["movement_passed"])==int(row["recovery_passed"])==1 for row in recovery),
            "Fixed-depth recovery retention changed")
    for name,count,controls in (("original-holds",6,1280),("withheld-holds",6,1280),("hold300",2,6000)):
        table=rows(bound(run/"final-skills"/name/"summary.csv"))
        require(len(table)==count and all(int(row["passed"])==1 and int(row["controls"])==controls for row in table),
                f"Retained holding acceptance changed: {name}")
    legacy=[line.strip() for line in (ROOT/"src/nn/motor/controller_contract.txt").read_text().splitlines() if line.strip()]
    require(len(legacy)==12,"Legacy contract count changed")
    contracts=[]
    for relative in legacy+GAIT_CONTRACTS:
        path=bound(ROOT/relative)
        contracts.append(hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest())
    payload=bytearray()

    def put(kind,*values): payload.extend(struct.pack("<"+kind,*values))
    def array(kind,values): put("Q",len(values));put(str(len(values))+kind,*values)

    put("12I",0x31564C48,3,6,6,3,4,41,4,336,3,7,1)
    put("Q",protocol);array("B",CANONICAL.encode("ascii"));put("I",len(contracts))
    for digest in contracts: array("B",digest.encode("ascii"))
    array("f",backbone);array("f",hold_head);put("2f",hold_gain,hold_sigma)
    put("4I",1,1,4,updates);array("f",means);put("3f",.35,.75,2.)
    array("B",sha(policy_path).encode("ascii"))
    package=struct.pack("<QQQ",0x31504B434C524443,len(payload),fnv(payload))+payload
    output=args.output.resolve();sidecar=output.with_suffix(".json")
    require(not output.exists() and not sidecar.exists(),"Choose a new private package identity; no overwrite is allowed")
    metadata={"schema":1,"scope":"Privately staged qualified frozen gait preview; runtime adoption is separate",
        "policy":{"family":"Learned swimming gait","holding_updates":7,"gait_updates":updates,
                  "package_version":3,"policy_method":"gait_trim_gain","qualified":True,
                  "package_sha256":hashlib.sha256(package).hexdigest().upper(),"source_checkpoint":str(policy_path)},
        "result":{"outcome":"Balanced swimming gait verified","hold_passes":6,"hold_total":6,
                  "short_passes":4,"long_passes":native["long60_passing_windows"],"long_retained":28,
                  "training_stopped":True,"source_run":str(run),
                  "method":"Native GPU PPO learned eight swimming trim/gain parameters; inherited body network and learned holding parameters remain frozen.",
                  "validation_note":"Balanced moving gait passed all four short swims and retained long swimming, fixed-depth recovery and original/withheld/endurance holding. Calm water only."},
        "gait":{"means":means,"biases":[.75*math.tanh(x) for x in means[:4]],
                "gains":[math.exp(math.log(2)*math.tanh(x)) for x in means[4:]],"sigma":f32(.35),
                "PPO_updates":updates,"actor_steps":steps,"qualification":proof["qualification"]},
        "source_checkpoint_sha256":sha(policy_path),"source_result":str(result_path),"source_result_sha256":sha(result_path),
        "independent_verification":str(proof_path),"independent_verification_sha256":sha(proof_path),
        "parent_sha256":CANONICAL,"holding_parent_sha256":HOLD_SHA,"source_protocol_checksum":f"{protocol:016X}",
        "runtime_adopted":False,"runtime_training":False,"backbone_frozen":True,"action_noise":False,
        "history":[],"sources":[{"path":str(result_path),"sha256":sha(result_path)},
                                  {"path":str(proof_path),"sha256":sha(proof_path)}]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as stream: stream.write(package)
    with sidecar.open("x",encoding="utf-8") as stream: json.dump(metadata,stream,indent=2);stream.write("\n")
    print(json.dumps({"staged":str(output),"qualified":True,"gait_PPO_updates":updates,"gait_Adam_steps":steps,
                      "runtime_adopted":False,"physics_controls":0,"optimizer_steps":0}))


if __name__=="__main__": main()
