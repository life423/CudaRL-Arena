// Offline reporting for the native continuous-coverage and close-shore evaluators.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { shorelineArcDegrees as arc } from './shoreline-arc.mjs';

const [longDir, shoreDir, out] = process.argv.slice(2);
if (!out) throw Error('Usage: node analyze-coverage.mjs LONG_ASSESSMENT SHORE_ASSESSMENT AUDIT_OUTPUT');
function readCsv(file) {
    const [header, ...lines] = fs.readFileSync(file, 'utf8').trimEnd().split(/\r?\n/);
    const names = header.split(',');
    return lines.map(line => {
        const fields = line.split(',');
        assert.equal(fields.length, names.length, file);
        return Object.fromEntries(names.map((name,i) => [name,fields[i]]));
    });
}
const key = (x,z) => x + ',' + z;
const cell = x => Math.floor(Math.fround(Math.fround(Math.fround(x)/Math.fround(.05)) + .5));
function samples(file) {
    const points = readCsv(file).filter(r => ['start','macro'].includes(r.event));
    points.forEach((r,i) => {
        assert.equal(+r.control, i*4, file);
        assert(Math.abs(+r.time_s - i*.08) < 1e-5, file);
        assert([r.x_m,r.z_m,r.clearance_bound_mm].every(v => v !== '' && Number.isFinite(+v)), file);
    });
    return points;
}

// All interior cells crossed by the straight segment between successive samples,
// including a short diagonal corner crossing which has no recorded point inside.
function crossedCells(ax,az,bx,bz, crossed) {
    const cuts = [0,1];
    for (const [a,b] of [[ax,bx],[az,bz]]) {
        for (let i=cell(Math.min(a,b)); i<cell(Math.max(a,b)); ++i) {
            const t = ((i+.5)*.05-a)/(b-a);
            if (t>0 && t<1) cuts.push(t);
        }
    }
    cuts.sort((a,b)=>a-b);
    for (let i=1; i<cuts.length; ++i) {
        const t=(cuts[i]+cuts[i-1])/2;
        crossed.add(key(cell(ax+(bx-ax)*t),cell(az+(bz-az)*t)));
    }
}


const original = readCsv(path.join(longDir,'coverage.csv'));
const geometry = readCsv(path.join(out,'geometry.csv'));
const byCase = readCsv(path.join(longDir,'cases.csv'));
const results=[], maps=[], byCell = ['case,x_cell,z_cell,scored,clear_headings,min_gap_mm,max_gap_mm,entered_sample,entered_segment,credited'];
for (const summary of byCase) {
    const id=summary.case;
    const scored=new Set(original.map(r=>key(cell(+r.x_m),cell(+r.z_m))));
    const credited=new Set(original.filter(r=>+r[id]===1).map(r=>key(cell(+r.x_m),cell(+r.z_m))));
    const shape=geometry.filter(r=>r.case===id);
    assert.deepEqual(new Set(shape.filter(r=>+r.reference_cell).map(r=>key(r.x_cell,r.z_cell))),scored);
    assert.equal(credited.size,+summary.unique_cells);
    const points=samples(path.join(longDir,'traces',id+'.csv'));
    assert.equal(+points.at(-1).time_s,1800);
    assert.equal(+points.at(-1).unique_cells,credited.size);
    const entered=new Set(), crossed=new Set();
    let near=0;
    for (let i=0;i<points.length;++i) {
        const r=points[i];
        entered.add(key(cell(+r.x_m),cell(+r.z_m)));
        if (i) {
            const p=points[i-1];
            crossedCells(+p.x_m,+p.z_m,+r.x_m,+r.z_m,crossed);
            if (+r.clearance_bound_mm>0 && +r.clearance_bound_mm<=20) near+=.08;
        }
    }
    const touched=new Set([...entered,...crossed]);
    for (const k of credited) assert(touched.has(k),id+' credited cell missing from track');
    const inScore=[...touched].filter(k=>scored.has(k)).length;
    const group = rows => {
        const keys=rows.map(r=>key(r.x_cell,r.z_cell));
        return {cells:keys.length, entered:keys.filter(k=>touched.has(k)).length, credited:keys.filter(k=>credited.has(k)).length};
    };
    for (const r of shape) {
        const k=key(r.x_cell,r.z_cell);
        byCell.push([id,r.x_cell,r.z_cell,r.reference_cell,r.clear_headings,r.min_gap_mm,r.max_gap_mm,
            Number(entered.has(k)),Number(touched.has(k)),Number(credited.has(k))].join(','));
    }
    const depths=new Set(shape.map(r=>+r.depth_mm));
    assert.equal(depths.size,1);
    const interior=shape.filter(r=>+r.reference_cell && +r.min_gap_mm>20);
    results.push({case:id,depth_mm:[...depths][0],scored:scored.size,credited:credited.size,
        crossed_uncredited:inScore-credited.size,unentered:scored.size-inScore,
        interpolated_only:[...touched].filter(k=>scored.has(k)&&!entered.has(k)).length,
        entered_outside_reference:touched.size-inScore,
        clear_at_own_depth:group(shape.filter(r=>+r.clear_headings>0)),
        scored_without_center_certificate_at_own_depth:shape.filter(r=>+r.reference_cell && +r.clear_headings===0).length,
        scored_interior:group(interior),
        credited_at_5_15_30_min:[300,900,1800].map(t=>+points[Math.round(t/.08)].unique_cells),
        near_s:near, near_arc_deg:arc(points)});
    maps.push({id,shape,points,credited,touched});
}
const shore=readCsv(path.join(shoreDir,'cases.csv')).map(r=>({
    case:r.case,pass:+r.pass,near_arc_deg:arc(samples(path.join(shoreDir,'traces',r.case+'.csv')))
}));
fs.writeFileSync(path.join(out,'coverage-results.json'),JSON.stringify({swims:results,shore},null,2)+'\n');
fs.writeFileSync(path.join(out,'cell-audit.csv'),byCell.join('\n')+'\n');

// Draw at metric scale. Decimation here affects the visible polyline only.
const colors={credited:'#5eae9c', crossed:'#e4ad55', missing:'#e0e4df', extra:'#faf9f4', ink:'#153b37'};
let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="842" viewBox="0 0 1040 842"><rect width="1040" height="842" fill="#faf9f4"/><g font-family="Segoe UI, sans-serif" fill="#163832">';
const label=(x,y,t,size,weight=400) => '<text x="'+x+'" y="'+y+'" font-size="'+size+'" font-weight="'+weight+'">'+t+'</text>';
svg+=label(32,47,'Where each tadpole swam',34,650)+label(33,77,'Four 30-minute swims with the installed Godot policy · top view · each square is 50 mm',18);
let lx=33;
for (const [text,color,width] of [['Credited',colors.credited,130],['Crossed without credit',colors.crossed,245],['Unentered',colors.missing,148]]) {
    svg+='<rect x="'+lx+'" y="101" width="19" height="19" fill="'+color+'" stroke="#667770"/>'+label(lx+27,117,text,16);
    lx+=width;
}
svg+='<path d="M560 111h36" stroke="'+colors.ink+'" stroke-width="2"/>'+label(605,117,'Actual path',16);
svg+='<path d="M743 111h34" stroke="#506a64" stroke-width="2" stroke-dasharray="4 3"/>'+label(785,117,'Open-water boundary¹',16);
const names=['Godot start','Northeast start','West start','Southwest start'];
for (let i=0;i<maps.length;++i) {
    const {shape,points,credited,touched}=maps[i], r=results[i];
    const x0=28+(i%2)*504, y0=155+Math.floor(i/2)*308;
    svg+=label(x0+4,y0+10,names[i],22,650);
    svg+=label(x0+4,y0+35,r.credited+' credited   ·   '+r.crossed_uncredited+' crossed without credit   ·   '+r.unentered+' unentered',15);
    const pitch=13, cx=x0+234, cy=y0+173;
    const interior=new Set(shape.filter(c=>+c.reference_cell && +c.min_gap_mm>20).map(c=>key(c.x_cell,c.z_cell)));
    for (const c of shape) {
        if (!+c.reference_cell) continue;
        const x=+c.x_cell, z=+c.z_cell, k=key(x,z);
        const left=cx+(x-.5)*pitch, top=cy+(z-.5)*pitch;
        const fill=credited.has(k)?colors.credited:(touched.has(k)?colors.crossed:colors.missing);
        svg+='<rect x="'+left+'" y="'+top+'" width="'+pitch+'" height="'+pitch+'" fill="'+fill+'" stroke="#faf9f4" stroke-width=".6"/>';
    }
    for (const k of interior) {
        const [x,z]=k.split(',').map(Number);
        const left=cx+(x-.5)*pitch, top=cy+(z-.5)*pitch;
        for (const [dx,dz,ax,az,bx,bz] of [[-1,0,0,0,0,1],[1,0,1,0,1,1],[0,-1,0,0,1,0],[0,1,0,1,1,1]]) {
            if (!interior.has(key(x+dx,z+dz))) svg+='<path d="M'+(left+ax*pitch)+' '+(top+az*pitch)+'L'+(left+bx*pitch)+' '+(top+bz*pitch)+'" fill="none" stroke="#506a64" stroke-dasharray="3 2" stroke-width="1.4"/>';
        }
    }
    const coords=points.filter((_,j)=>j%5===0).map(p=>(cx+260*p.x_m).toFixed(2)+','+(cy+260*p.z_m).toFixed(2)).join(' ');
    svg+='<polyline points="'+coords+'" fill="none" stroke="'+colors.ink+'" stroke-opacity=".73" stroke-width=".8"/>';
}
svg+=label(32,793,'¹ Inside the dashed boundary, all eight tested headings had more than 20 mm of whole-body clearance.',16);
svg+=label(32,819,'Cell colors use the same 345-cell reference area as the published score. Gray cells may lie inside or outside that boundary.',15);
fs.writeFileSync(path.join(out,'coverage.svg'),svg+'</g></svg>\n');
