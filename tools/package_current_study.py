"""Package exact closed-study evidence for portable, read-only Godot disclosure.

Copies saved JSON bytes; never simulates, trains, rewrites source evidence,
changes a policy, or claims the rejected current policy qualified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA="120F9B485A25373CB42CEF6CC3BD8659325A04A4A0547C3639369B586F8DC982"
PROOF_SHA="6F0C0D04C093B607D31D9AD0CF09C31C44B8180E8B4F27950F896555E32B8B70"


def require(value,message):
    if not value: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest().upper()


def identity(value):
    return posixpath.normpath(value.replace("\\","/")).casefold()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=ROOT/"assets/ppo-four-axis-checkpoints/current-swim-v2-2026-11-22-v1-rejected")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"live/research_data")
    parser.add_argument("--check",action="store_true",help="Verify packaged bytes without writing")
    args=parser.parse_args()
    source_bytes=(args.archive/"study-result.json").read_bytes()
    proof_bytes=(args.archive/"independent-training-verification.json").read_bytes()
    require(digest(source_bytes)==SOURCE_SHA and digest(proof_bytes)==PROOF_SHA,"Archived source/proof SHA differs")
    source=json.loads(source_bytes);proof=json.loads(proof_bytes)
    require(source.get("study")=="current_swim_v2" and source.get("closed") is True and source.get("qualified") is False,
            "Expected the actual closed, unqualified current V2 study")
    require(source.get("PPO_updates")==32 and source.get("baseline_current_passes")==4 and source.get("final_current_passes")==8,
            "Actual current study counters differ")
    require(proof.get("passed") is True and proof.get("qualified") is False and proof.get("phase")=="training",
            "Actual independent closed-training audit is missing")
    inputs=proof.get("input_sha256")
    require(isinstance(inputs,dict),"Independent input bindings are missing")
    bindings=[path for path,sha in inputs.items() if isinstance(path,str) and isinstance(sha,str) and sha.upper()==SOURCE_SHA]
    require(len(bindings)==1 and bindings[0].strip(),"Independent proof must bind the exact source once")
    audit_identity=bindings[0]
    require(sum(identity(path)==identity(audit_identity) for path in inputs)==1,"Ambiguous independent source identity")
    wrapper={"schema":1,"source":{"path":"res://research_data/current-study-result.json",
        "sha256":SOURCE_SHA.lower(),"verification_input":audit_identity},
        "verification":{"path":"res://research_data/current-study-verification.json","sha256":PROOF_SHA.lower()}}
    wrapper_bytes=(json.dumps(wrapper,indent=2)+"\n").encode("utf-8")
    files={"current-study-result.json":source_bytes,"current-study-verification.json":proof_bytes,"current-study.json":wrapper_bytes}
    if args.check:
        for name,wanted in files.items(): require((args.output_dir/name).read_bytes()==wanted,f"Packaged bytes differ: {name}")
    else:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for name,wanted in files.items():
            path=args.output_dir/name
            if name!="current-study.json" and path.exists(): require(path.read_bytes()==wanted,f"Refusing to replace different evidence: {name}")
        for name,wanted in files.items():
            path=args.output_dir/name
            if not path.exists() or path.read_bytes()!=wanted: path.write_bytes(wanted)
    print(json.dumps({"passed":True,"checked_only":args.check,"source_sha256":SOURCE_SHA,"proof_sha256":PROOF_SHA,
        "qualified":False,"PPO_updates":32,"current_cases_before":4,"current_cases_after":8,
        "physics_controls":0,"physics_substeps":0,"optimizer_steps":0}))


if __name__=="__main__": main()
