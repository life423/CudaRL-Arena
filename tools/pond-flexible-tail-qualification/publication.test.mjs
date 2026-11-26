import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { csv, stallPass, continuousPass, coverageLosses } from './publication.mjs';
import { scoreOriginalMoving } from './moving-publication.mjs';

const evidence = new URL('../../assets/research-evidence/',import.meta.url);
const rows = relative => csv(fileURLToPath(new URL(relative,evidence)));
test('the twelve moving approaches keep their own fixture identity and full physical horizon', () => {
    for(const depth of [72,75,80]) {
        const saved=rows('pond-flexible-tail-open-water-chooser-v1/results/selection/update-104/depth-'+depth+'/cases.csv');
        assert.equal(scoreOriginalMoving(saved,depth).filter(r=>r.passed).length,0);
        assert.throws(()=>scoreOriginalMoving(saved.slice(1),depth));
        const trial=saved.map(r=>({...r,pass:'1',fault:'0',min_clearance_mm:'1',accepted_substeps:'144000',max_stall_controls:'407'}));
        assert.equal(scoreOriginalMoving(trial,depth).filter(r=>r.passed).length,4);
        trial[0].accepted_substeps='143999';
        trial[1].max_stall_controls='408';
        trial[2].min_clearance_mm='0';
        assert.equal(scoreOriginalMoving(trial,depth).filter(r=>r.passed).length,1);
    }
});
test('every swimmer must retain its own update-128 coverage', () => {
    const parent=rows('pond-flexible-tail-three-depth-exploration-v1/qualification/candidate/continuous/assessment/cases.csv');
    assert.deepEqual(coverageLosses(parent,parent),[]);
    const trial=parent.map(row=>({...row}));
    trial[0].unique_cells='184';
    trial[1].unique_cells='210';
    assert.deepEqual(coverageLosses(trial,parent),['endurance_godot_rest covered 184 cells; update 128 covered 185.']);
    assert.throws(()=>coverageLosses(trial.reverse(),parent));
});
test('inclusive cutoff on the native per-swimmer control counter', () => {
    assert.equal(stallPass({max_stall_controls:'406'}),true);
    assert.equal(stallPass({max_stall_controls:'407'}),true);
    assert.equal(stallPass({max_stall_s:'8.14',max_stall_controls:'408'}),false);
    assert.throws(()=>stallPass({max_stall_s:'8.14'}));
    assert.throws(()=>stallPass({max_stall_controls:'407.1'}));
});
test('regional update eight is rejected on its 30-minute Godot-start swim', () => {
    const original = rows('pond-flexible-tail-regional-exploration-v1/data/continuous/update-008/assessment/cases.csv');
    const replay = rows('pond-flexible-tail-three-depth-exploration-v1/qualification/regression/continuous/assessment/cases.csv');
    assert.deepEqual(replay.map(({max_stall_controls,...oldColumns})=>oldColumns),original);
    const swimmer=replay.find(r=>r.case==='endurance_godot_rest');
    assert.equal(swimmer.physical_pass,'1');
    assert.equal(swimmer.simulated_s,'1800');
    assert.equal(swimmer.max_stall_controls,'2071');
    assert.equal(continuousPass(swimmer),false);
    assert.equal(continuousPass({...swimmer,max_stall_controls:'407'}),true);
});
test('the installed southwest swimmer passes exactly at the boundary', () => {
    const installed=rows('pond-flexible-tail-three-depth-exploration-v1/qualification/installed/continuous/assessment/cases.csv');
    const swimmer=installed.find(r=>r.case==='endurance_southwest');
    assert.equal(swimmer.max_stall_controls,'407');
    assert.equal(continuousPass(swimmer),true);
});
