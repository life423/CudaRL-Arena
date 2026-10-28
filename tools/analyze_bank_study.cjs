// Offline report analysis only; not a dependency of native C++/CUDA PPO.
const fs=require('node:fs');
const root=process.argv[2]||'build/pond-bank-study';
function csv(path) {
  const lines=fs.readFileSync(path,'utf8').trim().split(/\r?\n/);const header=lines.shift().split(',');
  return lines.map(line=>Object.fromEntries(line.split(',').map((value,i)=>[header[i],value])));
}
const key=r=>`${r.seed}:${r.environment}`;
const average=values=>values.reduce((a,b)=>a+b,0)/values.length;
const hit=(row,seconds)=>+row.first_food_s>0 && +row.first_food_s<=seconds;
const modes=['original','pilot','new'];const banks={};
for(const mode of modes) banks[mode]=['0xBAA7E001','0xBAA7E002'].flatMap(seed=>csv(`${root}/holdout-${seed}-${mode}/cases.csv`));
const bank={};
for(const mode of modes) {
  const rows=banks[mode];bank[mode]={trials:rows.length,recovered20:rows.filter(r=>hit(r,20)).length,recovered60:rows.filter(r=>hit(r,60)).length,
    mean_foods:average(rows.map(r=>+r.foods)),low_speed_over_20s:rows.filter(r=>+r.max_low_speed_s>20).length};
}
const bankPairs={};
for(const mode of ['original','pilot']) {
  const old=new Map(banks[mode].map(r=>[key(r),r]));let gained=0,lost=0;
  for(const row of banks.new) {const before=old.get(key(row));if(!before) throw Error('Unpaired bank identity');gained+=!hit(before,20)&&hit(row,20);lost+=hit(before,20)&&!hit(row,20);}
  bankPairs[mode]={new_20s_successes:gained,lost_20s_successes:lost,net_percentage_points:100*(gained-lost)/banks.new.length};
}
function normal(kind) {
  const rows=csv(`${root}/fresh-${kind}/cases.csv`),prior=csv(`${root}/prior-fresh-${kind}/cases.csv`);
  const priorByKey=new Map(prior.map(r=>[`${r.condition}:${key(r)}`,r]));let mismatches=0;
  const groups=new Map();
  for(const row of rows) {
    const old=priorByKey.get(`${row.condition}:${key(row)}`);if(!old) throw Error('Unpaired normal identity');
    if(+row.original_foods!==+old.original_foods || +row.original_gap_seconds!==+old.original_gap_seconds) ++mismatches;
    const group=kind==='long'?row.condition:(row.condition==='calm'?'calm':row.condition.startsWith('cm05')?'0.5_cm_s':'1.0_cm_s');
    if(!groups.has(group)) groups.set(group,[]);groups.get(group).push({row,old});
  }
  const result={};
  for(const [name,entries] of groups) {
    const factor=60/+entries[0].row.seconds;
    const summarize=(food,gap)=>({food_per_minute:average(entries.map(e=>food(e)*factor)),gaps_over_20s:entries.filter(e=>gap(e)>20).length,max_gap_seconds:Math.max(...entries.map(gap))});
    const original=summarize(e=>+e.row.original_foods,e=>+e.row.original_gap_seconds);
    const pilot=summarize(e=>+e.old.candidate_foods,e=>+e.old.candidate_gap_seconds);
    const candidate=summarize(e=>+e.row.candidate_foods,e=>+e.row.candidate_gap_seconds);
    const comparisons={};
    for(const mode of ['original','pilot']) {
      const clusters=new Map();
      for(const e of entries) {const id=key(e.row);const oldFood=mode==='original'?+e.row.original_foods:+e.old.candidate_foods;
        if(!clusters.has(id)) clusters.set(id,[]);clusters.get(id).push((+e.row.candidate_foods-oldFood)*factor);}
      const differences=[...clusters.values()].map(average),mean=average(differences);
      const variance=differences.reduce((sum,x)=>sum+(x-mean)**2,0)/(differences.length-1),ci=1.96*Math.sqrt(variance/differences.length);
      comparisons[mode]={paired_delta_fpm:mean,approximate_ci95:[mean-ci,mean+ci],layout_clusters:differences.length};
    }
    result[name]={trials:entries.length,original,pilot,candidate,comparisons};
  }
  return {groups:result,repeated_original_reference_mismatches:mismatches};
}
const result={bank,bank_pairs:bankPairs,normal_minute:normal('minute'),normal_long:normal('long'),
  note:'Intervals cluster repeated directions within each layout and are conditional on these selected policies and source seeds; they do not measure training-seed variance. No multiple-testing correction.'};
fs.writeFileSync(`${root}/aggregate.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
