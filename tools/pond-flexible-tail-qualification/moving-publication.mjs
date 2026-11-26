import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {csv, evaluate, stallPass} from './publication.mjs';

const reference = fileURLToPath(new URL('../../assets/research-evidence/pond-flexible-tail-open-water-chooser-v1/results/selection/update-104/', import.meta.url));
const depths = [72,75,80];
const names = ['moving_a','moving_b','moving_c','moving_d'];
const hash = file => createHash('sha256').update(readFileSync(file)).digest('hex').toUpperCase();
const json = file => JSON.parse(readFileSync(file,'utf8').replace(/^\uFEFF/,''));

function frozenInput(receipt, bindings, index, expectedHash) {
    const original = receipt.arguments[index].replaceAll('\\','/').toLowerCase();
    const matches = bindings.filter(b => original.endsWith('/'+b.path.replaceAll('\\','/').toLowerCase()));
    assert.equal(matches.length,1,'Unbound final assessment input '+original);
    assert.equal(matches[0].sha256,expectedHash);
}

export function scoreOriginalMoving(rows, depth) {
    const moving = rows.filter(r => r.suite === 'moving');
    assert.deepEqual(moving.map(r => r.case), names);
    return moving.map(row => {
        assert.equal(Number(row.depth_mm), depth);
        assert.match(row.accepted_substeps,/^\d+$/);
        const accepted = Number(row.accepted_substeps);
        assert(accepted > 0 && accepted <= 144000);
        assert.match(row.pass,/^[01]$/);
        assert(row.min_clearance_mm.trim() !== '' && Number.isFinite(Number(row.min_clearance_mm)));
        return {
            depth_mm:depth, case:row.case,
            passed:row.pass === '1' && row.fault === '0' && accepted === 144000 && Number(row.min_clearance_mm) > 0 && stallPass(row),
            fault:Number(row.fault), seconds:accepted/800, min_clearance_mm:Number(row.min_clearance_mm),
            max_stall_controls:Number(row.max_stall_controls)
        };
    });
}

export function movingPublication(qualification, actor, selectedFile) {
    const selected = json(selectedFile);
    assert.equal(selected.actor_sha256,hash(actor),'Final actor differs from frozen selection');
    const base = evaluate(qualification, actor);
    const candidate = path.join(qualification,'candidate');
    const bindings = json(path.join(candidate,'bindings.json'));
    const moving = [];
    for(const depth of depths) {
        const suite = 'original-moving-'+depth;
        const root = path.join(candidate,suite);
        const receipt = json(path.join(root,suite+'.receipt.json'));
        const old = path.join(reference,'depth-'+depth);
        assert.equal(receipt.timeout,false);
        assert.equal(receipt.exit,0);
        assert.deepEqual(receipt.arguments.slice(5),['0','--open-validation']);
        const regular = json(path.join(candidate,'shore-'+depth,'shore-'+depth+'.receipt.json'));
        assert.equal(receipt.executable,regular.executable,'Original moving test ran a different evaluator');
        for(const i of [0,1,3]) assert.equal(receipt.arguments[i].replaceAll('\\','/').toLowerCase(),regular.arguments[i].replaceAll('\\','/').toLowerCase());
        frozenInput(receipt, bindings, 2, selected.actor_sha256);
        assert.equal(hash(path.join(root,'assessment','starts.csv')),hash(path.join(old,'starts.csv')));
        moving.push(...scoreOriginalMoving(csv(path.join(root,'assessment','cases.csv')), depth));
    }
    const failures = [...base.failures,...moving.filter(r=>!r.passed).map(r=>r.depth_mm+' mm '+r.case+' failed the original moving approach at '+r.seconds+' s (fault '+r.fault+').')];
    return {...base, selected_update:selected.update, moving, eligible_for_staging:failures.length===0, failures};
}

if(process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const [qualification, actor, selected, option] = process.argv.slice(2);
    assert(qualification && actor && selected && [undefined,'--report-only'].includes(option));
    const report = movingPublication(qualification,actor,selected);
    console.log(JSON.stringify(report,null,2));
    if(!report.eligible_for_staging && !option) process.exitCode=1;
}
