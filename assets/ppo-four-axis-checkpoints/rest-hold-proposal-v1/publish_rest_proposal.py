from pathlib import Path
import json,hashlib,shutil,csv,subprocess
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation');local=Path(__file__).parent
src=r/'build/analysis/rest-hold-proposal-v1';dest=r/'assets/ppo-four-axis-checkpoints/rest-hold-proposal-v1'
assert not dest.exists(),'Proposal identity exists'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
result=json.loads((src/'results/result.json').read_text());assert result['physical_controls']==0 and result['optimizer_steps']==0 and result['passed']
prior=json.loads((r/'build/review-preservation/2026-11-04-before-frozen-brake/prior-evidence-hashes.json').read_text())
for h in prior:assert sha(r/h['path'])==h['sha256']
brake=r/'assets/ppo-four-axis-checkpoints/frozen-brake-B344'
brakefiles=json.loads((brake/'manifest.json').read_text())
for h in brakefiles:assert sha(brake/h['path'])==h['sha256']
unrelated=json.loads((r/'build/review-preservation/2026-11-04-before-frozen-brake/unrelated-hashes.json').read_text())
for h in unrelated:assert sha(r/h['path'])==h['sha256']
frozen=json.loads((r/'build/diagnostics/frozen-brake-B344-v1/frozen-hashes.json').read_text())
for h in frozen:assert sha(Path(h['path']))==h['sha256']
dest.mkdir()
shutil.copy2(local/'REST_HOLD_PROPOSAL.md',dest/'PROTOCOL.md')
for name in ['rest-starts.csv','instantaneous-reward-grid.csv','result.json']:shutil.copy2(src/'results'/name,dest/name)
shutil.copy2(src/'execution.json',dest/'static-probe-execution.json')
shutil.copy2(local/'rest_scale_probe.cpp',dest/'rest_scale_probe.cpp');shutil.copy2(local/'build_rest_scale.ps1',dest/'build_rest_scale.ps1');shutil.copy2(local/'publish_rest_proposal.py',dest/'publish_rest_proposal.py')
prov={'status':'PROPOSED ONLY — requires user approval before any physical rollout or learning','branch':'codex/minimal-foundation',
 'current_physical_controls':0,'current_optimizer_steps':0,'runtime_changed':False,'current_source_changes':False,
 'starting_checkpoint':'update-336-episode64/ppo-update-336.bin','starting_SHA256':'810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0',
 'measured_static_states':6,'proposed_training_horizon_controls':320,'proposed_hold_training_starts':'Centered at rest and hold point, zero goal from first control; no depth offset in hold training',
 'proposed_rehearsal_fraction':.5,'proposed_hold_reward_gain':1/1024,'proposed_critic_only_steps':80,'proposed_PPO_updates':8,
 'proposed_actor_Adam_steps_max':640,'proposed_critic_Adam_steps_max_including_calibration':720,
 'proposed_controls_max':113664,'proposed_substeps_max':1136640,'proposed_native_process_seconds_max':120,
 'pretraining_gates':'Return bounds, held-out critic fit/retention and first-PPO advantage allocation; fail closed without retuning',
 'required_physical_passes':'6/6 new64srestholds,4/4shortswims,retain28passinglongwindows; no grace period for rest holding',
 'prior_evidence_files_verified':len(prior),'frozen_brake_archive_files_verified':len(brakefiles),'unrelated_files_verified':len(unrelated),'last_frozen_input_files_verified':len(frozen),
 'limitations':'Static constant-cost arithmetic is not a rollout; actual returns and training success unmeasured. Critic fitting is proposed learning, not already performed.'}
(dest/'provenance.json').write_text(json.dumps(prov,indent=2))
manifest=[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file()]
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'protocol':str(dest/'PROTOCOL.md'),'provenance':prov},indent=2))
