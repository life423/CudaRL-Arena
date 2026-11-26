import assert from 'node:assert/strict';
import path from 'node:path';
import {writeFileSync} from 'node:fs';
import {csv} from './publication.mjs';

const [reference, candidate, output] = process.argv.slice(2);
assert(reference && candidate && output,'Expected +104 selection directory, final candidate directory, and output CSV');
const result = [];
for(const depth of [72,75,80]) {
    for(const name of ['moving_a','moving_b','moving_c','moving_d']) {
        const before = csv(path.join(reference,'depth-'+depth,'assessment','traces',name+'.csv'));
        const after = csv(path.join(candidate,'original-moving-'+depth,'assessment','traces',name+'.csv'));
        const previous = before.filter(row=>row.event==='decision');
        const next = after.filter(row=>row.event==='decision');
        assert.equal(next[0].time_s, '0.8');
        let oldChoice, newChoice;
        for(let i=0;i<Math.min(previous.length,next.length);i++) {
            const a=previous[i], b=next[i];
            assert.equal(a.control,b.control);
            for(const key of ['x_m','y_m','z_m','actual_COM_speed_mm_s','clearance_bound_mm'])
                assert(Math.abs(Number(a[key])-Number(b[key]))<1e-5, 'Inconsistent common physical prefix in '+name);
            if(a.action!==b.action){oldChoice=a;newChoice=b;break;}
        }
        if(!oldChoice) {
            assert.equal(previous.length,next.length);
            assert.equal(before.at(-1).time_s,after.at(-1).time_s);
        }
        const firstTurn=next.find(row=>Number(row.requested_course_rate_deg_s)!==0);
        result.push({depth_mm:depth,case:name,first_new_turn_s:firstTurn?.time_s??'',
            first_difference_s:oldChoice?.time_s??'',com_speed_at_difference_mm_s:oldChoice?.actual_COM_speed_mm_s??'',
            terrain_gap_at_difference_mm:oldChoice?.clearance_bound_mm??'',
            previous_request_mm_s:oldChoice?.requested_speed_mm_s??'',previous_rate_deg_s:oldChoice?.requested_course_rate_deg_s??'',
            selected_request_mm_s:newChoice?.requested_speed_mm_s??'',selected_rate_deg_s:newChoice?.requested_course_rate_deg_s??'',
            previous_survival_s:before.at(-1).time_s,selected_survival_s:after.at(-1).time_s,
            previous_fault:before.at(-1).fault,selected_fault:after.at(-1).fault});
    }
}
writeFileSync(output,[Object.keys(result[0]).join(','),...result.map(r=>Object.values(r).join(','))].join('\n')+'\n');
