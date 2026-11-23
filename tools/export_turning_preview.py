"""Stage a qualified food-free continuous swimmer for separate native review.

Saved-data packaging only. Never steps physics, applies PPO/Adam, replaces a live
policy, or writes the active Godot project. Exact assessment/proof copies are
staged beside the package so later adoption can use portable resource paths.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from export_gait_preview import Reader
from export_hold_preview import fnv, CANONICAL
from export_current_preview import ROOT, HOLD_SHA, require, sha, key, f32, finite, parent_backbone, holding_parent, rows

CONTRACTS=["src/nn/motor/turning_policy_v1.h","src/nn/motor/turning_task_v1.h"]
QUALIFICATION={"selection_passes":16,"withheld_continuous_passes":8,"withheld_seconds":1800,
    "short_swim_passes":4,"long60_retained_windows":28,"recovery_passes":3,
    "long300_retained_windows":148,"original_hold_passes":6,"withheld_hold_passes":6,"hold300_passes":2}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--verification",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"build/continuous-pond-preview-2026-11-23/continuous-pond-v1.bin")
    args=parser.parse_args();run=args.run.resolve()
    result_path=run/"training-result.json";native=json.loads(result_path.read_text())
    require(native.get("closed") is True and native.get("qualified") is True,
            "Refusing export: continuous swimming is not closed and qualified")
    require(not native.get("fault") and native.get("runtime_adopted") is False and native.get("retries")==0,
            "Refusing export: continuous-study fault or adoption metadata differs")
    proof_path=(args.verification or run/"independent-verification.json").resolve()
    proof=json.loads(proof_path.read_text())
    require(proof.get("passed") is True and proof.get("qualified") is True and proof.get("phase")=="training",
            "Refusing export: full independent continuous qualification is missing")
    require(all(proof.get("qualification",{}).get(name)==wanted for name,wanted in QUALIFICATION.items()),
            "Required continuous/retained-skill qualification differs")
    inputs={key(path):str(digest).upper() for path,digest in proof.get("input_sha256",{}).items()}

    def bound(path):
        path=path.resolve()
        require(inputs.get(key(path))==sha(path),f"Verified input changed or is unbound: {path.name}")
        return path

    bound(result_path)
    require(str(proof.get("assessment_sha256","")).upper()==sha(result_path),"Independent assessment SHA differs")
    for name,wanted in QUALIFICATION.items(): require(native.get(name)==wanted,f"Native qualification differs: {name}")
    updates,steps=native["PPO_updates"],native["actor_steps"]
    require(type(updates) is int and 0<=updates<=64 and type(steps) is int and updates*4<=steps<=updates*16,
            "Invalid turning optimizer counters")
    inherited=updates==0
    require(native.get("inherited_only") is inherited,"Native inherited/learned classification differs")
    require(native.get("final_policy_update")==updates,"Selected continuous candidate update differs")
    require(native.get("critic_steps")==native.get("critic_forward_calls")==0,"Unexpected continuous critic work")
    controls,substeps=native["controls"],native["substeps"]
    require(type(controls) is int and type(substeps) is int and 0<controls<=86811440 and controls<=substeps<=controls*10,
            "Continuous physical counters exceed the frozen budget")
    parent_path=bound(run/"parent-checkpoint.bin");hold_path=bound(run/"holding-parent.bin");protocol_path=bound(run/"protocol.bin")
    protocol=fnv(Reader(protocol_path).data)
    parent_checksum,hold_checksum=fnv(Reader(parent_path).data),fnv(Reader(hold_path).data)
    backbone=parent_backbone(parent_path);hold_head,hold_gain,hold_sigma=holding_parent(hold_path,backbone,parent_checksum)
    policy_path=bound(run/"training-started"/f"turning-policy-update-{updates}.bin")
    require(str(proof.get("final_policy_sha256","")).upper()==sha(policy_path),"Final continuous checkpoint SHA differs")
    r=Reader(policy_path)
    require([r.value("I"),r.value("I")]==[0x31505054,1],"Wrong turning checkpoint schema")
    require([r.value("Q") for _ in range(3)]==[protocol,parent_checksum,hold_checksum],"Turning checkpoint protocol/parents differ")
    require([r.value("I") for _ in range(10)]==[6,1,3,4,1,41,8,4,336,16],"Wrong turning model/task/sensor dimensions")
    require([r.value("i") for _ in range(3)]==[4096,updates,steps],"Turning checkpoint capacity/counters differ")
    require(r.array("f")==backbone and r.array("f")==hold_head and r.value("f")==hold_gain,
            "The preserved body/holding policy changed")
    means=r.array("f")
    require(len(means)==16 and r.value("f")==f32(.20) and r.value("f")==.75,"Wrong turning distribution/mapping")
    finite(means);adam=bytes(r.array("B"));require(r.at==len(r.data),"Unexpected turning checkpoint bytes")
    require(len(adam)==168 and struct.unpack_from("<5f",adam)==tuple(f32(x) for x in (.005,.9,.999,.00001,.5)),"Wrong turning Adam configuration")
    require(struct.unpack_from("<i",adam,20)[0]==steps,"Turning Adam age differs")
    offset=24;moments=[]
    for _ in range(2):
        require(struct.unpack_from("<Q",adam,offset)[0]==16,"Wrong turning Adam moment dimensions");offset+=8
        moments.append(struct.unpack_from("<16f",adam,offset));offset+=64
    finite(moments[0]+moments[1]);require(all(x>=0 for x in moments[1]),"Negative turning Adam second moment")
    if inherited:
        require(all(value==0 for value in means+moments[0]+moments[1]),"Inherited transfer must keep the exact initial zero head/optimizer")

    selected_directory=run/"baseline-free" if inherited else run/"training-started"/f"assessment-{updates}"
    for directory,count,horizon in ((selected_directory,16,1200),(run/"withheld-validation",8,36000)):
        summary=rows(bound(directory/"summary.csv"))
        require(len(summary)==count,"Wrong continuous validation case count")
        for row in summary:
            require(int(row["passed"])==int(row["survived"])==1 and int(row["actual_controls"])==horizon,
                    "A required continuous validation case did not survive/pass")
            require(int(row["attempted_substeps"])==int(row["accepted_substeps"])==horizon*10,
                    "Continuous validation has a rejected/partial interval")
            require(int(row["task_done"])==0 and int(row["external_truncated"])==1,
                    "A bounded validation cutoff was misrepresented as an episode success/reset")
    retention_path=bound(run/"final-retention/retention.json");retention=json.loads(retention_path.read_text())
    require(retention.get("full") is True,"Full old-skill retention was not assessed")
    for name,wanted in {"short_passes":4,"original_holds":6,"long60_retained":28,"recovery_passes":3,
                        "long300_retained":148,"withheld_holds":6,"endurance_holds":2}.items():
        require(retention.get(name)==wanted,f"Retained skill differs: {name}")
    require(28<=retention.get("long60_windows",-1)<=30 and 148<=retention.get("long300_windows",-1)<=150,
            "Actual long-swim totals differ")
    ablation=rows(bound(run/"constant-only-ablation/summary.csv"))
    require(len(ablation)==16,"State-feedback ablation evidence is incomplete")
    legacy=[line.strip() for line in (ROOT/"src/nn/motor/controller_contract.txt").read_text().splitlines() if line.strip()]
    require(len(legacy)==12,"Legacy source contract count changed")
    contracts=[]
    for relative in legacy+CONTRACTS:
        path=bound(ROOT/relative)
        contracts.append(hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest())
    payload=bytearray()

    def put(kind,*values): payload.extend(struct.pack("<"+kind,*values))
    def array(kind,values): put("Q",len(values));put(str(len(values))+kind,*values)

    put("12I",0x31564C48,4,6,6,3,4,41,4,336,4,7,1)
    put("Q",protocol);array("B",CANONICAL.encode("ascii"));put("I",len(contracts))
    for digest in contracts: array("B",digest.encode("ascii"))
    array("f",backbone);array("f",hold_head);put("2f",hold_gain,hold_sigma)
    put("5I",1,1,4,1,updates);array("f",means);put("6f",.20,.75,.20,.12,.195,.115)
    array("B",sha(policy_path).encode("ascii"))
    package=struct.pack("<QQQ",0x31504B434C524443,len(payload),fnv(payload))+payload
    output=args.output.resolve();sidecar=output.with_suffix(".json")
    assessment_copy=output.with_name(output.stem+"-assessment.json")
    verification_copy=output.with_name(output.stem+"-verification.json")
    for path in (output,sidecar,assessment_copy,verification_copy): require(not path.exists(),"Choose a fresh private package identity; overwrite is not allowed")
    method=("The inherited PPO motor transferred to the food-free task. No new turning parameters were learned; its existing body and holding weights are unchanged." if inherited else
            "Native GPU PPO learned sixteen boundary-aware yaw parameters in the same four-output motor. Inherited swimming and holding weights remain frozen.")
    continuous_note=("Inherited motor transfer; zero new PPO updates. " if inherited else "The restricted sixteen-parameter PPO steering head was trained. ")+"All sixteen one-minute selection starts and eight withheld thirty-minute starts passed. No food, scripted route, deadline or automatic world reset is used during playback."
    metadata={"schema":1,"scope":"Privately staged qualified frozen continuous swimmer; adoption is separate",
        "policy":{"family":"Transferred PPO swimmer" if inherited else "Learned bank turning","holding_updates":7,"turning_updates":updates,"free_swim":True,
                  "package_version":4,"policy_method":"learned_bank_turning_v1","qualified":True,
                  "package_sha256":hashlib.sha256(package).hexdigest().upper(),"source_checkpoint":str(policy_path),
                  "source_checkpoint_sha256":sha(policy_path)},
        "result":{"qualified":True,"outcome":"Continuous pond swimming verified","hold_passes":6,"hold_total":6,
                  "short_passes":4,"long_passes":retention["long60_windows"],"long_retained":28,"training_stopped":True,
                  "source_run":str(run),"method":method,"inherited_only":inherited,"new_learning":not inherited,
                  "validation_note":"Original swimming, fixed-depth recovery and original/withheld/endurance holding were retained. Calm water only.",
                  "continuous_validation_note":continuous_note},
        "assessment":{"path":"res://research_data/"+assessment_copy.name,"sha256":sha(result_path)},
        "verification":{"path":"res://research_data/"+verification_copy.name,"sha256":sha(proof_path)},
        "continuous":{"means":means,"PPO_updates":updates,"actor_steps":steps,"qualification":proof["qualification"],
                      "sigma":f32(.20),"steering_limit":.75,"active_radii_m":[.20,.12],"inset_radii_m":[.195,.115],
                      "runtime_deadline":False,"runtime_world_resets":False,"runtime_training":False,"inherited_only":inherited,"new_learning":not inherited,
                      "ablation_passes":sum(int(row["passed"]) for row in ablation)},
        "parent_sha256":CANONICAL,"holding_parent_sha256":HOLD_SHA,"source_protocol_checksum":f"{protocol:016X}",
        "runtime_adopted":False,"backbone_frozen":True,"action_noise":False,"history":[],
        "sources":[{"path":str(result_path),"sha256":sha(result_path)},{"path":str(proof_path),"sha256":sha(proof_path)}]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as stream: stream.write(package)
    with assessment_copy.open("xb") as stream: stream.write(result_path.read_bytes())
    with verification_copy.open("xb") as stream: stream.write(proof_path.read_bytes())
    with sidecar.open("x",encoding="utf-8") as stream: json.dump(metadata,stream,indent=2);stream.write("\n")
    print(json.dumps({"staged":str(output),"qualified":True,"turning_PPO_updates":updates,"turning_Adam_steps":steps,
                      "runtime_adopted":False,"physics_controls":0,"optimizer_steps":0}))


if __name__=="__main__": main()
