import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {parseArgs} from 'node:util';
const opts=parseArgs({options:{archive:{type:'string'},raw:{type:'string'},baseline:{type:'string'},selected:{type:'string'},actor:{type:'string'}}}).values;
const join=path.join,read=file=>fs.readFileSync(file,'utf8').replace(/^\uFEFF/,'').trim();
const csv=file=>{
    const [header,...rows]=read(file).split(/\r?\n/);
    const keys=header.split(',');
    return rows.map(row=>Object.fromEntries(row.split(',').map((value,i)=>[keys[i],value])));
};
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex').toUpperCase();
const selected=JSON.parse(read(opts.selected));
assert.equal(hash(opts.actor),selected.actor_sha256);
const rows=[];
let work=0,process_s=0,matchingHashes=0;
const speed=a=>[6,8,12][Math.floor(a/5)];
const rate=a=>(a%5-2)*4;
const same=(a,b,tol)=>assert(Math.abs(Number(a)-Number(b))<=tol,`mismatch ${a} vs ${b}`);
for(const depth of [72,75,80]) {
    const receipt=JSON.parse(read(join(opts.raw,`receipt-${depth}.json`)));
    assert.equal(receipt.exit,0);assert.equal(receipt.timeout,false);process_s+=receipt.wall_s;
    const candidates=csv(join(opts.raw,`depth-${depth}`,'candidate.csv'));
    const base=csv(join(opts.baseline,`depth-${depth}`,'branches.csv'));
    const cases=csv(join(opts.baseline,`depth-${depth}`,'cases.csv'));
    assert.equal(candidates.length,8);
    for(const c of candidates) {
        const match=r=>r.case===c.case&&r.upstream_mm===c.upstream_mm;
        const previous=base.filter(match);
        const initial=cases.find(match);
        assert.equal(previous.length,15);assert.equal(initial.depth_mm,c.depth_mm);
        for(const [key,oldkey] of [['spawn_gap_mm','spawn_gap_mm'],['cue_gap_mm','cue_gap_mm'],['cue_speed_mm_s','cue_actual_speed_mm_s']])
            same(c[key],initial[oldkey],1e-6);
        if(c.cue_hash===previous[0].cue_hash)++matchingHashes;
        const r={depth_mm:depth,case:c.case,upstream_mm:Number(c.upstream_mm),cue_speed_mm_s:c.cue_speed_mm_s,
            candidate_first_speed:speed(+c.first_action),candidate_first_rate:rate(+c.first_action),
            candidate_qualified:+c.qualified180s,candidate_survival_s:c.survival_s,candidate_min_gap_mm:c.min_gap_mm,
            candidate_stall_s:+c.longest_stall_controls/50,candidate_mean_speed_mm_s:c.mean_speed_mm_s,candidate_escape20:+c.escaped20};
        work+=+c.attempted_substeps;
        assert(+c.attempted_substeps<=144000);
        assert.equal(Number(!+c.fault && +c.survival_s===180 && +c.min_gap_mm>0 && +c.mean_speed_mm_s>=6 &&
            +c.tilt_deg<=10 && +c.depth_error_mm<=1 && +c.longest_stall_controls<=407),+c.qualified180s);
        for(const id of [104,112,128]) {
            const a=previous.find(x=>+x.actor===id && +x.forced_action===-1);
            assert.equal(a.first_actual_action,c[`first_${id}`]);
            Object.assign(r,{[`qualified_${id}`]:+a.recovered180,[`survival_s_${id}`]:a.survival_s,
                [`min_gap_mm_${id}`]:a.min_clearance_mm,[`first_speed_${id}`]:speed(+a.first_actual_action),
                [`first_rate_${id}`]:rate(+a.first_actual_action),[`first_difference_s_${id}`]:c[`different_s_${id}`],
                [`chosen_action_at_difference_${id}`]:c[`candidate_when_different_${id}`],[`old_action_at_difference_${id}`]:c[`reference_when_different_${id}`]});
        }
        const witness=previous.filter(x=>+x.recovered180).sort((a,b)=>+b.min_clearance_mm-+a.min_clearance_mm)[0];
        Object.assign(r,{prior_witness_actor:witness?.actor??'',prior_witness_forced_action:witness?.forced_action??'',
            prior_witness_held_decisions:witness?.hold_decisions??'',prior_witness_min_gap_mm:witness?.min_clearance_mm??''});
        rows.push(r);
    }
}
assert.equal(rows.length,24);
assert.equal(new Set(rows.map(r=>[r.depth_mm,r.case,r.upstream_mm].join('/'))).size,24);
assert(work<=3456000);
const keys=Object.keys(rows[0]);
fs.writeFileSync(join(opts.archive,'comparison.csv'),[keys,...rows.map(row=>keys.map(k=>row[k]))].map(row=>row.join(',')).join('\n')+'\n');
const grouped=rows=>Object.fromEntries(['candidate_qualified','qualified_104','qualified_112','qualified_128','previous_witness'].map(k=>[k,rows.filter(r=>k==='previous_witness'?r.prior_witness_actor!=='':r[k]===1).length]));
const summary={update:selected.update,actor_sha256:selected.actor_sha256,physical_substeps:work,process_wall_s:process_s,
    full_raw_cue_hashes_matching:matchingHashes,case_depth_cue_gap_speed_and_all_three_archived_initial_choices_matched:rows.length,
    cases_per_shift:12,upstream_60:grouped(rows.filter(r=>r.upstream_mm===60)),upstream_120:grouped(rows.filter(r=>r.upstream_mm===120)),
    at_depth:Object.fromEntries([72,75,80].map(d=>[d,grouped(rows.filter(r=>r.depth_mm===d))])),optimizer_updates:0};
fs.writeFileSync(join(opts.archive,'summary.json'),JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(summary));
