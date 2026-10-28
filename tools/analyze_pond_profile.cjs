// Optional profiling analysis with Node's built-in SQLite reader; no dependency
// in the C++/CUDA training loop. Input is an Nsight Systems SQLite export.
const fs = require('node:fs');
const { DatabaseSync } = require('node:sqlite');
const input = process.argv[2];
if (!input) throw new Error('Usage: node analyze_pond_profile.cjs TRACE.sqlite [OUTPUT.json]');
const db = new DatabaseSync(input, { readOnly: true });
const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all().map(x => x.name);
const relevant = tables.filter(x => /CUPTI_ACTIVITY_KIND_(KERNEL|RUNTIME|MEMCPY|MEMSET)|NVTX_EVENTS|StringIds/.test(x));
if (process.argv.includes('--schema')) {
  for (const name of relevant) console.log(name, db.prepare(`PRAGMA table_info("${name}")`).all().map(x => x.name));
  console.log(db.prepare('SELECT * FROM NVTX_EVENTS LIMIT 3').all());
  process.exit(0);
}
const ranges = db.prepare("SELECT start,end,text FROM NVTX_EVENTS WHERE text IN ('measured_training','update','rollout','bootstrap_gae','ppo_optimization') AND end IS NOT NULL ORDER BY start").all();
const capture = ranges.find(x => x.text === 'measured_training');
if (!capture) throw new Error('Missing measured_training NVTX range');
const kernels = db.prepare('SELECT k.start,k.end,k.correlationId,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').all();
const runtime = db.prepare('SELECT r.start,r.end,r.correlationId,s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId ORDER BY r.start').all();
const transfers = [];
for (const name of ['CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']) {
  if (tables.includes(name)) transfers.push(...db.prepare(`SELECT start,end FROM ${name}`).all());
}
function unionTime(events, begin, end) {
  const intervals = events.map(e => [Math.max(begin, e.start), Math.min(end, e.end)]).filter(e => e[1] > e[0]).sort((a,b) => a[0]-b[0]);
  let total=0, left=0, right=0;
  for (const [a,b] of intervals) {
    if (a>right) { total+=right-left; left=a; right=b; }
    else right=Math.max(right,b);
  }
  return total+right-left;
}
const span = capture.end-capture.start;
const busy=unionTime([...kernels,...transfers],capture.start,capture.end);
const kernelTotal=kernels.reduce((sum,e)=>sum+e.end-e.start,0);
const physics=kernels.filter(e=>e.name.includes('progress_selector') && e.name.includes('advance_kernel'));
const physicsTotal=physics.reduce((sum,e)=>sum+e.end-e.start,0);
const launches=runtime.filter(e=>e.name.startsWith('cudaLaunchKernel'));
if (!launches.length) throw new Error('No CUDA launches found; check the trace API naming/schema');
const durations=launches.map(e=>e.end-e.start).sort((a,b)=>a-b);
const byId=new Map(runtime.map(e=>[e.correlationId,e]));
const queued=kernels.map(e=>({kernel:e,api:byId.get(e.correlationId)})).filter(e=>e.api);
const delays=queued.map(e=>Math.max(0,e.kernel.start-e.api.end)).sort((a,b)=>a-b);
const phaseTimes={};
for (const entry of queued) {
  const phase=ranges.find(r=>['rollout','bootstrap_gae','ppo_optimization'].includes(r.text) && entry.api.start>=r.start && entry.api.start<r.end);
  if (phase) phaseTimes[phase.text]=(phaseTimes[phase.text]??0)+(entry.kernel.end-entry.kernel.start)/1e6;
}
const quantile=(values,q)=>values[Math.min(values.length-1,Math.floor(q*values.length))] ?? 0;
const result={
  source:input,capture_ms:span/1e6,gpu_busy_union_ms:busy/1e6,gpu_busy_fraction:busy/span,
  gpu_idle_fraction:1-busy/span,kernel_count:kernels.length,kernel_time_sum_ms:kernelTotal/1e6,
  physics_kernel_count:physics.length,physics_time_ms:physicsTotal/1e6,
  physics_fraction_of_kernel_time:physicsTotal/kernelTotal,physics_fraction_of_capture:physicsTotal/span,
  memory_operation_time_ms:transfers.reduce((sum,e)=>sum+e.end-e.start,0)/1e6,
  cuda_launch_calls:launches.length,launch_api_median_us:quantile(durations,.5)/1e3,
  launch_api_p95_us:quantile(durations,.95)/1e3,launch_api_max_ms:quantile(durations,1)/1e6,
  queued_kernel_median_ms:quantile(delays,.5)/1e6,queued_kernel_p95_ms:quantile(delays,.95)/1e6,
  kernel_time_by_submission_phase_ms:phaseTimes,
  note:'API call time overlaps GPU work and can include launch-queue backpressure; it is not additive pure CPU overhead. CUDA event/phase spans include idle gaps. No hardware utilization metric implies peak arithmetic efficiency.'
};
if (process.argv[3]) fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
