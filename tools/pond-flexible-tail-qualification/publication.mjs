import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { shorelineArcDegrees } from '../pond-flexible-tail-chooser/shoreline-arc.mjs';

const read = file => readFileSync(file, 'utf8').replace(/^\uFEFF/, '');
const json = file => JSON.parse(read(file));
const sha = file => createHash('sha256').update(readFileSync(file)).digest('hex').toUpperCase();
const live = ['endurance_godot_rest','endurance_northeast','endurance_west','endurance_southwest'];
const floors = [151,147,148,160];
const motor = '01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E';
const installedHash = '3B0B68C1CD89F6FE8CBA2DFCDDE591CDE5A8C145856D36368ECC2E9DA8594323';
const parentHash = '7A16B85CDF19E98D150510F0D24713B592A2EB7DC5ADB124880B0E4EB30473A4';
const parentDir = fileURLToPath(new URL('../../assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/qualification/candidate/', import.meta.url));
const gait = '2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6';
const fixtures = {
    72: '4E5B86C513DEF4B5AEF955E2896CC87C940DA6970D49EE8B53A4471951687689',
    75: '31ECBEDFB9B4F6752044171EADF59AB934DD8EFB407140170C6A961EB15355C4',
    80: '4A081CCC3DEA8C7310191902CC094E50FA6C0B341606DC9DBC4E63DD581FFF39'
};
const shoreNames = [];
for (const pose of [8,9]) for (const bank of [0,1]) for (const sign of [-1,1])
    shoreNames.push('phase'+pose+'-bank'+bank+'-direction'+sign);

// Native evidence contains only unquoted identifiers and numerical fields.
export function csv(file) {
    const [header,...lines] = read(file).trimEnd().split(/\r?\n/);
    const keys = header.split(',');
    assert.equal(new Set(keys).size, keys.length);
    return lines.map(line => {
        const fields = line.split(',');
        assert.equal(fields.length, keys.length);
        return Object.fromEntries(keys.map((key,i) => [key, fields[i]]));
    });
}
function number(row,key) {
    const value = row[key];
    assert(value !== undefined && value.trim() !== '' && Number.isFinite(Number(value)), 'Missing or invalid '+key);
    return Number(value);
}
export function stallPass(row) {
    assert(/^(0|[1-9]\d*)$/.test(row.max_stall_controls), 'Native stall counter required');
    const ticks = Number(row.max_stall_controls);
    assert(Number.isSafeInteger(ticks));
    return ticks <= 407;
}
function continuousPhysicalPass(row) {
    return row.physical_pass === '1' && row.fault === '0' &&
        number(row,'accepted_substeps') === 1440000 && number(row,'attempted_substeps') === 1440000 &&
        number(row,'min_clearance_mm') > 0 && number(row,'simulated_s') === 1800;
}
export function continuousPass(row) {
    return stallPass(row) && continuousPhysicalPass(row);
}
export function coverageLosses(candidate, parent) {
    assert.deepEqual(parent.map(row => row.case), live);
    assert.deepEqual(candidate.map(row => row.case), live);
    return candidate.flatMap((row, i) => {
        const cells = number(row, 'unique_cells');
        const required = number(parent[i], 'unique_cells');
        return cells < required ? [row.case+' covered '+cells+' cells; update 128 covered '+required+'.'] : [];
    });
}
function assessment(dir, name, actorHash, fixtureHash) {
    const where = path.join(dir,name);
    const receipt = json(path.join(where,name+'.receipt.json'));
    const bindings = json(path.join(dir,'bindings.json'));
    function bound(file) {
        const absolute = file.replaceAll('\\','/').toLowerCase();
        const hits = bindings.filter(b => absolute.endsWith('/'+b.path.replaceAll('\\','/').toLowerCase()));
        assert.equal(hits.length,1,'Unbound assessment input '+file);
        assert.match(hits[0].sha256,/^[0-9A-F]{64}$/);
        return hits[0].sha256;
    }
    assert.equal(receipt.timeout,false);
    assert([0,1].includes(receipt.exit));
    assert.equal(bound(receipt.arguments[0]),gait);
    assert.equal(bound(receipt.arguments[1]),motor);
    assert.equal(bound(receipt.arguments[2]),actorHash);
    assert.equal(bound(receipt.arguments[3]),fixtureHash);
    bound(receipt.executable);
    assert.equal(receipt.arguments[5],'0');
    assert.deepEqual(receipt.arguments.slice(6), name==='navigation' ? [] :
        [name==='continuous' ? '--continuous-coverage' : '--close-shore']);
    const root = path.join(where,'assessment');
    const summary = json(path.join(root,'summary.json'));
    assert.equal(summary.optimizer_updates,0);
    assert.equal(summary.pass,receipt.exit===0);
    return {root,summary,rows:csv(path.join(root,'cases.csv'))};
}
export function evaluate(root, actorFile) {
    const failures = [];
    const check = (ok, reason) => { if (!ok) failures.push(reason); };
    const actorHash = sha(actorFile);
    const candidate = path.join(root,'candidate');
    const installed = path.join(root,'installed');
    const nav = assessment(candidate,'navigation',actorHash,fixtures[80]);
    check(nav.summary.pass && nav.summary.resets===0 && nav.summary.diagnostic_passes===12 &&
        nav.summary.endurance_passes===4 && nav.summary.clearance_stops===0 && nav.summary.other_native_faults===0,
        'Original navigation requires 12 ordinary and four five-minute passes with zero stops or resets.');
    for(const row of nav.rows) {
        check(row.pass==='1' && stallPass(row), row.case+' failed navigation or its individual stall limit.');
    }
    for(const depth of [72,75,80]) {
        const suite = 'shore-'+depth;
        const a = assessment(installed,suite,installedHash,fixtures[depth]);
        const b = assessment(candidate,suite,actorHash,fixtures[depth]);
        assert.equal(sha(path.join(a.root,'starts.csv')),sha(path.join(b.root,'starts.csv')));
        for(const side of [a,b]) assert.deepEqual(side.rows.map(r=>r.case),shoreNames);
        check(b.summary.resets===0, depth+' mm shoreline reset.');
        for(const row of b.rows) {
            check(row.pass==='1' && row.fault==='0' && number(row,'accepted_substeps')===144000 &&
                number(row,'attempted_substeps')===144000 && number(row,'min_clearance_mm')>0,
                depth+' mm '+row.case+' failed the physical shoreline test at '+row.simulated_s+' s, fault '+row.fault+'.');
            check(stallPass(row), depth+' mm '+row.case+' stalled for '+row.max_stall_controls+' controls.');
        }
        const sustained=b.rows.filter(r=>r.pass==='1' && number(r,'close_time_s')>=15).length;
        const mean=b.rows.reduce((s,r)=>s+number(r,'close_time_s'),0)/8;
        check(sustained>=4, depth+' mm had '+sustained+'/8 sustained near-bank swims; four required.');
        check(mean>=10.96, depth+' mm mean near-bank time '+mean.toFixed(2)+' s; 10.96 s required.');
    }
    const a=assessment(installed,'continuous',installedHash,fixtures[80]);
    const b=assessment(candidate,'continuous',actorHash,fixtures[80]);
    for(const side of [a,b]) {
        assert.deepEqual(side.rows.map(r=>r.case),live);
        assert.equal(side.summary.reference_cells,345);
    }
    assert.deepEqual(a.rows.map(r=>number(r,'unique_cells')),floors);
    const parent = assessment(parentDir,'continuous',parentHash,fixtures[80]);
    assert(parent.rows.every(continuousPass) && parent.summary.resets===0);
    assert.deepEqual(parent.rows.map(r=>number(r,'unique_cells')), [185,195,218,201]);
    failures.push(...coverageLosses(b.rows, parent.rows));
    check(b.summary.resets===0 && b.summary.faults===0,'Continuous trial recorded a reset or native stop.');
    let total=0;
    for(const [i,row] of b.rows.entries()) {
        check(stallPass(row), row.case+' exceeded 407 native stall controls ('+row.max_stall_controls+').');
        check(continuousPhysicalPass(row), row.case+' failed physical completion at '+row.simulated_s+' s, fault '+row.fault+'.');
        const cells=number(row,'unique_cells');
        check(cells>=floors[i], row.case+' fell below installed coverage.');
        total+=cells;
    }
    check(total>floors.reduce((sum,n)=>sum+n,0),'Mean coverage did not improve.');
    let shoreArc = null;
    const traceDir = path.join(candidate,'shore-80','assessment','traces');
    if (existsSync(traceDir) || failures.length===0) {
        const shoreRows = csv(path.join(candidate,'shore-80','assessment','cases.csv'));
        const arcs = shoreRows.map(row => {
            const points = csv(path.join(traceDir,row.case+'.csv')).filter(r=>r.event==='start' || r.event==='macro');
            assert(points.length>0 && points[0].event==='start');
            assert.equal(points.length,1+Math.floor(number(row,'accepted_substeps')/64));
            for (const [i, r] of points.entries()) {
                assert.equal(number(r,'control'),4*i);
                assert(Math.abs(number(r,'time_s')-.08*i)<1e-5);
                for (const key of ['x_m','z_m','clearance_bound_mm']) number(r,key);
            }
            return shorelineArcDegrees(points);
        });
        shoreArc = Math.max(...arcs);
        check(shoreArc>=11.81,'Maximum uninterrupted 80 mm shoreline arc '+shoreArc.toFixed(2)+'° fell below 11.81°.');
    }
    return {actor_sha256:actorHash,motor_sha256:motor,stall_limit_controls:407,control_ms:20,
        longest_80mm_shore_arc_deg:shoreArc,
        eligible_for_staging:failures.length===0,failures};
}
if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
    const [root,actor,option]=process.argv.slice(2);
    assert(root && actor && [undefined,'--report-only','--installed-stalls'].includes(option),
        'Expected qualification directory and actor');
    if(option==='--installed-stalls') {
        assert.equal(sha(actor),installedHash);
        const installed=assessment(path.join(root,'installed'),'continuous',installedHash,fixtures[80]);
        assert.deepEqual(installed.rows.map(row=>row.case),live);
        assert.equal(installed.summary.resets,0);
        assert(installed.rows.every(continuousPass),'Installed swimmer exceeded its native qualification');
    } else {
        const decision=evaluate(root,actor);
        console.log(JSON.stringify(decision,null,2));
        if(!decision.eligible_for_staging && !option) process.exitCode=1;
    }
}
