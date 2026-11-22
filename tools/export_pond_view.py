"""Export small, evidence-backed GUI metadata. No simulation or learning."""
import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

def rows(path):
    return list(csv.DictReader(path.open(newline="",encoding="utf-8")))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--assessment",type=Path,required=True)
    parser.add_argument("--policy-metadata",type=Path,default=Path("live/research_data/learned-amplitude-v1.json"))
    parser.add_argument("--method",required=True)
    parser.add_argument("--outcome",required=True)
    parser.add_argument("--family",required=True)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--output",type=Path,default=Path("live/research_data/viewer.json"))
    args=parser.parse_args()
    result=json.loads((args.run/"training-result.json").read_text())
    assert result["closed"] and not result.get("fault"),"Only closed, nonfaulted study results are exported"
    policy=json.loads(args.policy_metadata.read_text())
    policy_file=args.policy_metadata.with_suffix('.bin')
    policy_bytes=policy_file.read_bytes()
    assert hashlib.sha256(policy_bytes).hexdigest().lower()==policy['package_sha256'].lower()
    package_version=struct.unpack_from('<I',policy_bytes,28)[0]
    holds=rows(args.assessment/"rest"/"summary.csv")
    shorts=rows(args.assessment/"frozen-evaluation.csv")
    windows=rows(args.assessment/"long"/"sustained-windows.csv")
    assert len(holds)==6 and len(shorts)==4 and len(windows)==30
    passed={(int(float(row["pose"])),int(float(row["end_s"]))) for row in windows if int(row["passed"])}
    original={(pose,second) for pose in [0,3] for second in range(8,61,4)}
    history=[]
    if args.previous and args.previous.exists():
        prior=json.loads(args.previous.read_text())
        history=prior.get("history",[])
        previous_result=prior.get("result",{})
        if previous_result.get("source_run")!=str(args.run.resolve()) and previous_result:
            history.append({"name":prior.get("policy",{}).get("family","Earlier study"),"outcome":previous_result.get("outcome","See evidence"),"source_run":previous_result.get("source_run")})
    payload={"schema":1,"policy":{"family":args.family,"holding_updates":policy["holding_updates"],"package_version":package_version,"policy_method":policy.get("policy_method","four_axis_hold_gains"),
             "package_sha256":policy["package_sha256"],"qualified":policy["qualified"],"source_checkpoint":policy["source_checkpoint"]},
             "result":{"outcome":args.outcome,"hold_passes":sum(int(row["passed"]) for row in holds),"hold_total":6,
                       "short_passes":sum(int(row["passed"]) for row in shorts),"long_passes":len(passed),"long_retained":len(passed&original),
                       "method":args.method,"source_run":str(args.run.resolve()),"training_stopped":True},
             "history":history[-6:],"sources":[]}
    if all(key in policy for key in ["original_holds","withheld_holds","endurance_holds","endurance_seconds"]):
        payload["result"]["validation_note"]=(f"{policy['original_holds']}/6 original and {policy['withheld_holds']}/6 withheld 64-second holds passed, plus {policy['endurance_holds']}/2 holds lasting {policy['endurance_seconds']} seconds. Calm water only; braking and currents remain separate.")
    for path in [args.run/"training-result.json",args.policy_metadata,args.assessment/"rest"/"summary.csv",args.assessment/"frozen-evaluation.csv",args.assessment/"long"/"sustained-windows.csv"]:
        payload["sources"].append({"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"exported":str(args.output),"holding":payload["result"]["hold_passes"],"swimming":payload["result"]["short_passes"],"long_retained":payload["result"]["long_passes"]}))

if __name__=="__main__": main()
