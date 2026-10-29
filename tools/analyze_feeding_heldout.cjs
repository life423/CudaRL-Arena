#!/usr/bin/env node
'use strict';

// Read-only analysis of frozen-policy cases. No simulation, training or promotion.
const fs = require('node:fs');
const path = require('node:path');
const GROUPS = ['retention', 'calm', 'steady', 'fluctuating'];
const POLICIES = ['before', 'after'];
const FIELDS = ['policy', 'seed', 'world', 'group', 'foods', 'food_per_minute',
  'first_food_seconds', 'first_food_censored', 'max_gap_seconds',
  'max_available_gap_seconds', 'immobility_events', 'max_immobility_seconds',
  'terminal_immobile', 'empty_seconds', 'final_gap_seconds', 'final_food_count',
  'created_foods', 'cap_seconds', 'invalid_body_ticks', 'invalid_food_ticks'];
const BOOTSTRAP_DRAWS = 10000;
const BOOTSTRAP_SEED = 0xF00D501;

function requireThat(ok, message) { if (!ok) throw Error(message); }
function csvRows(text) {
  const rows = []; let row = [], value = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { value += '"'; i++; }
      else if (ch === '"') quoted = false;
      else value += ch;
    } else if (ch === '"') {
      requireThat(value === '', 'Malformed CSV quoted field'); quoted = true;
    } else if (ch === ',') { row.push(value); value = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(value); if (row.some(v => v !== '')) rows.push(row);
      row = []; value = '';
    } else value += ch;
  }
  requireThat(!quoted, 'Unterminated CSV quoted field');
  if (value !== '' || row.length) { row.push(value); rows.push(row); }
  return rows;
}
function mean(values) { return values.reduce((sum, n) => sum + n, 0) / values.length; }
function sum(values) { return values.reduce((total, n) => total + n, 0); }
function quantile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), at = (sorted.length - 1) * fraction;
  const low = Math.floor(at), high = Math.ceil(at);
  return sorted[low] + (sorted[high] - sorted[low]) * (at - low);
}
function distribution(values) {
  return { mean: values.length ? mean(values) : null, median: quantile(values, .5),
    p95: quantile(values, .95), max: values.length ? Math.max(...values) : null };
}
function parseCases(directory, metadata) {
  requireThat(metadata.protocol === 'autonomous-heldout-v1', 'Unsupported study protocol');
  requireThat(metadata.seconds === 120 && metadata.worlds === 16, 'Expected fixed 120-second, 16-world protocol');
  requireThat(Array.isArray(metadata.seeds) && metadata.seeds.length === 8, 'Expected eight declared seeds');
  const seeds = metadata.seeds.map(Number);
  requireThat(seeds.every(n => Number.isInteger(n) && n >= 0 && n <= 0xffffffff) && new Set(seeds).size === 8,
    'Declared seeds must be eight unique uint32 values');
  for (const flag of ['self_check', 'actor_weights_unchanged', 'source_reference_unchanged'])
    requireThat(metadata[flag] === true, 'Provenance verification failed/missing: ' + flag);
  for (const policy of POLICIES) {
    const actor = metadata[policy];
    requireThat(actor && Number.isInteger(actor.learner_update) && Number.isInteger(actor.validated_update) &&
      actor.validated_update >= 0 && actor.validated_update <= actor.learner_update && typeof actor.actor_fnv64 === 'string',
      'Missing or inconsistent saved actor metadata: ' + policy);
  }
  const parsed = csvRows(fs.readFileSync(path.join(directory, 'cases.csv'), 'utf8'));
  requireThat(parsed.length > 1, 'No evaluation cases');
  const header = parsed.shift();
  requireThat(header.length === new Set(header).size, 'Duplicate CSV columns');
  requireThat(FIELDS.every(name => header.includes(name)), 'CSV is missing required columns');
  const indices = Object.fromEntries(FIELDS.map(name => [name, header.indexOf(name)]));
  const rows = [], keys = new Map();
  const integers = ['seed', 'world', 'foods', 'first_food_censored', 'immobility_events',
    'terminal_immobile', 'final_food_count', 'created_foods', 'invalid_body_ticks', 'invalid_food_ticks'];
  for (let line = 0; line < parsed.length; line++) {
    const cells = parsed[line], row = {};
    requireThat(cells.length === header.length, 'CSV column count mismatch at row ' + (line + 2));
    for (const name of FIELDS) {
      const raw = cells[indices[name]];
      if (name === 'policy' || name === 'group') row[name] = raw;
      else {
        requireThat(raw.trim() !== '', 'Missing ' + name + ' at row ' + (line + 2));
        row[name] = Number(raw);
        requireThat(Number.isFinite(row[name]), 'Nonfinite ' + name + ' at row ' + (line + 2));
      }
    }
    requireThat(POLICIES.includes(row.policy) && GROUPS.includes(row.group), 'Unknown policy or group');
    requireThat(integers.every(name => Number.isInteger(row[name]) && row[name] >= 0), 'Invalid integer case metric');
    requireThat(seeds.includes(row.seed) && row.world < metadata.worlds, 'Undeclared seed/world');
    requireThat(row.group === GROUPS[row.world % 4], 'World/group assignment does not match the four-group protocol');
    requireThat([0, 1].includes(row.first_food_censored) && [0, 1].includes(row.terminal_immobile), 'Invalid boolean case metric');
    requireThat((row.first_food_censored === 1 && row.first_food_seconds === -1 && row.foods === 0) ||
      (row.first_food_censored === 0 && row.first_food_seconds >= 0 && row.first_food_seconds <= metadata.seconds + .051 && row.foods > 0),
    'First-food censoring/latency/count is inconsistent');
    requireThat(Math.abs(row.food_per_minute - row.foods * 60 / metadata.seconds) <= 1e-6, 'Food rate/count mismatch');
    for (const name of ['max_gap_seconds', 'max_available_gap_seconds', 'max_immobility_seconds', 'empty_seconds', 'final_gap_seconds', 'cap_seconds'])
      requireThat(row[name] >= 0 && row[name] <= metadata.seconds + .051, 'Out-of-range duration: ' + name);
    requireThat(row.foods + row.final_food_count === row.created_foods, 'Food conservation mismatch');
    const key = row.policy + '/' + row.seed + '/' + row.world;
    requireThat(!keys.has(key), 'Duplicate case: ' + key); keys.set(key, row); rows.push(row);
  }
  requireThat(rows.length === 256, 'Expected exactly 256 rows; found ' + rows.length);
  const pairs = [];
  for (const seed of seeds) for (let world = 0; world < metadata.worlds; world++) {
    const before = keys.get('before/' + seed + '/' + world), after = keys.get('after/' + seed + '/' + world);
    requireThat(before && after && before.group === after.group, 'Missing/mismatched paired case');
    pairs.push({ seed, world, group: before.group, before, after, delta: after.food_per_minute - before.food_per_minute });
  }
  for (const group of GROUPS) {
    const selected = pairs.filter(pair => pair.group === group);
    requireThat(selected.length === 32, 'Expected 32 paired cases in ' + group);
    for (const seed of seeds) requireThat(selected.filter(pair => pair.seed === seed).length === 4, 'Unbalanced seed/group cases');
  }
  return { seeds, pairs };
}
function policySummary(rows) {
  const latency = rows.filter(row => !row.first_food_censored).map(row => row.first_food_seconds);
  const values = name => rows.map(row => row[name]);
  return {
    cases: rows.length, foods: sum(values('foods')), food_per_minute: mean(values('food_per_minute')),
    first_food: { censored_cases: rows.length - latency.length, observed_cases: latency.length, seconds_among_observed: distribution(latency) },
    max_gap_seconds: distribution(values('max_gap_seconds')),
    max_available_gap_seconds: distribution(values('max_available_gap_seconds')),
    immobility: { confirmed_cases: rows.filter(row => row.immobility_events > 0).length,
      confirmed_events: sum(values('immobility_events')), max_seconds: Math.max(...values('max_immobility_seconds')),
      terminal_immobile_cases: sum(values('terminal_immobile')) },
    invalid: { cases: rows.filter(row => row.invalid_body_ticks > 0 || row.invalid_food_ticks > 0).length,
      body_ticks: sum(values('invalid_body_ticks')), food_ticks: sum(values('invalid_food_ticks')) },
    empty_seconds: distribution(values('empty_seconds')), final_gap_seconds: distribution(values('final_gap_seconds')),
    accepted_food_creations: { total_including_initial: sum(values('created_foods')), mean_including_initial: mean(values('created_foods')) },
    final_food_count: distribution(values('final_food_count')),
    capacity: { cases_at_cap: rows.filter(row => row.cap_seconds > 0).length, seconds: distribution(values('cap_seconds')) }
  };
}
function analyze(directory) {
  const metadata = JSON.parse(fs.readFileSync(path.join(directory, 'metadata.json'), 'utf8').replace(/^\uFEFF/, ''));
  const { seeds, pairs } = parseCases(directory, metadata);
  const names = [...GROUPS, 'overall'];
  const selected = Object.fromEntries(names.map(group => [group, group === 'overall' ? pairs : pairs.filter(pair => pair.group === group)]));
  const clusters = Object.fromEntries(names.map(group => [group, seeds.map(seed => mean(selected[group].filter(pair => pair.seed === seed).map(pair => pair.delta)))]));
  const draws = Object.fromEntries(names.map(group => [group, []])); let randomState = BOOTSTRAP_SEED;
  const next = () => { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 4294967296; };
  for (let draw = 0; draw < BOOTSTRAP_DRAWS; draw++) {
    const sampled = Array.from({ length: seeds.length }, () => Math.floor(next() * seeds.length));
    for (const group of names) draws[group].push(mean(sampled.map(index => clusters[group][index])));
  }
  const results = {};
  for (const group of names) {
    const cases = selected[group], before = policySummary(cases.map(pair => pair.before)), after = policySummary(cases.map(pair => pair.after));
    const delta = mean(cases.map(pair => pair.delta)), low = quantile(draws[group], .025), high = quantile(draws[group], .975);
    results[group] = { pairs: cases.length, before, after,
      paired: { delta_food_per_minute: delta, change_percent: before.food_per_minute === 0 ? null : delta * 100 / before.food_per_minute,
        ci95: [low, high], interval_method: 'paired cluster bootstrap of seed-group mean deltas; percentile 95%',
        observed_direction: delta > 0 ? 'increase' : delta < 0 ? 'decrease' : 'tie',
        evidence: low > 0 ? 'increase' : high < 0 ? 'decrease' : 'inconclusive',
        wins: cases.filter(pair => pair.delta > 0).length, ties: cases.filter(pair => pair.delta === 0).length,
        losses: cases.filter(pair => pair.delta < 0).length,
        seed_mean_deltas: seeds.map((seed, index) => ({ seed, delta: clusters[group][index] })) } };
    const margin = .05 * before.food_per_minute;
    results[group].provisional_interpretation = {
      five_percent_margin_food_per_minute: margin,
      material_mean_regression: delta < -margin,
      lower_ci_above_negative_five_percent_margin: low > -margin,
      retention_evidence: low > -margin ? 'bounded retention supported' : high < -margin ? 'material decrease supported' : 'inconclusive',
      after_reliability_gap: after.immobility.confirmed_cases > 0 || after.immobility.terminal_immobile_cases > 0 || after.invalid.cases > 0
    };
  }
  const materialRegressions = GROUPS.filter(group => results[group].provisional_interpretation.material_mean_regression);
  const reliabilityGap = results.overall.provisional_interpretation.after_reliability_gap;
  const supportsOverallGain = results.overall.paired.ci95[0] > 0 && materialRegressions.length === 0;
  const interpretation = {
    thresholds_are_provisional: true,
    any_after_reliability_gap: reliabilityGap,
    material_subgroup_mean_regressions: materialRegressions,
    bounded_overall_food_rate_improvement_supported: supportsOverallGain,
    next_step_signal: reliabilityGap ? 'Investigate reliability before anatomy' :
      materialRegressions.length ? 'Address material subgroup regression before anatomy' :
        supportsOverallGain ? 'Bounded food-rate progress supported; broader validation remains necessary' :
          'Overall progress inconclusive; do not claim a general food-rate improvement',
    rules: [
      'Any after-policy confirmed food-present five-second immobility or invalid case is a reliability gap to address before anatomy.',
      'A subgroup mean food-rate loss exceeding 5% is a material regression.',
      'A subgroup lower confidence bound above minus 5% of its before mean provides stronger bounded retention support.',
      'An overall food-rate improvement claim requires a positive overall lower confidence bound and no material subgroup mean loss.'
    ]
  };
  const warnings = [
    'This is progress between saved validated actors, not causal evidence that the new curriculum beats equally long old-curriculum training.',
    'Aggregate improvement does not establish improvement in every current subgroup. Confidence intervals are descriptive; no multiple-comparison adjustment is applied.',
    'Eight seed clusters give limited uncertainty resolution; an interval containing zero is inconclusive, not proof of equivalence.',
    'First-food latency summaries exclude censored cases; compare censoring counts alongside latency.',
    'Timed food can create legitimate empty-pond waiting. Available-food gaps/immobility are reported separately from raw food gaps.',
    'Matched nominal arrivals can be suppressed differently at the occupancy cap. Accepted food creations include initial pellets.'
  ];
  const summary = { protocol: metadata.protocol, provenance: metadata,
    uncertainty: { draws: BOOTSTRAP_DRAWS, seed: BOOTSTRAP_SEED, clusters: seeds.length, resampling_unit: 'seed; same sampled seed IDs across all groups and both policies' },
    groups: Object.fromEntries(GROUPS.map(group => [group, results[group]])), overall: results.overall,
    provisional_decision: interpretation, limitations: warnings };
  fs.writeFileSync(path.join(directory, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  const number = value => value === null ? 'n/a' : value.toFixed(3);
  const signed = value => (value > 0 ? '+' : '') + number(value);
  const lines = ['# Frozen feeding-policy held-out comparison', '',
    'Eight held-out seeds; 16 worlds per seed; 120 seconds per actor. Each subgroup contains 32 matched pairs. Policies use frozen mean actions with no learning or deployment.', '',
    '| Group | Before food/min | After food/min | Paired change | Change % | Seed-cluster 95% interval | Food-rate evidence | Wins/ties/losses |',
    '| --- | ---: | ---: | ---: | ---: | --- | --- | --- |'];
  for (const group of names) {
    const row = results[group], p = row.paired;
    lines.push('| ' + group + ' | ' + number(row.before.food_per_minute) + ' | ' + number(row.after.food_per_minute) +
      ' | ' + signed(p.delta_food_per_minute) + ' | ' + number(p.change_percent) + ' | [' + signed(p.ci95[0]) + ', ' + signed(p.ci95[1]) +
      '] | ' + p.evidence + ' | ' + p.wins + '/' + p.ties + '/' + p.losses + ' |');
  }
  lines.push('', '## Reliability and timing', '',
    '| Group / policy | Available-food gap median / max (s) | Confirmed immobility cases / events | Terminal immobile | Invalid cases | First-food censored / observed | Observed first-food median (s) | Mean accepted creations | Mean cap time (s) |',
    '| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |');
  for (const group of names) for (const policy of POLICIES) {
    const row = results[group][policy];
    lines.push('| ' + group + ' / ' + policy + ' | ' + number(row.max_available_gap_seconds.median) + ' / ' + number(row.max_available_gap_seconds.max) +
      ' | ' + row.immobility.confirmed_cases + ' / ' + row.immobility.confirmed_events + ' | ' + row.immobility.terminal_immobile_cases +
      ' | ' + row.invalid.cases + ' | ' + row.first_food.censored_cases + ' / ' + row.first_food.observed_cases +
      ' | ' + number(row.first_food.seconds_among_observed.median) + ' | ' + number(row.accepted_food_creations.mean_including_initial) +
      ' | ' + number(row.capacity.seconds.mean) + ' |');
  }
  lines.push('', '## Prospective interpretation', '', interpretation.next_step_signal + '.', '',
    'These are provisional decision thresholds declared before the held-out run, not universal guarantees or simultaneous subgroup confidence claims.', '',
    ...interpretation.rules.map(rule => '- ' + rule), '',
    'Material subgroup mean regressions: ' + (materialRegressions.length ? materialRegressions.join(', ') : 'none') + '.',
    'After-policy reliability gap: ' + (reliabilityGap ? 'yes' : 'none observed in this bounded suite') + '.', '',
    '| Group | Mean loss exceeds 5% | Retention evidence against −5% margin |',
    '| --- | --- | --- |');
  for (const group of GROUPS) {
    const decision = results[group].provisional_interpretation;
    lines.push('| ' + group + ' | ' + (decision.material_mean_regression ? 'yes' : 'no') + ' | ' + decision.retention_evidence + ' |');
  }
  lines.push('', '## Interpretation limits', '', ...warnings.map(warning => '- ' + warning), '',
    'Provenance and all duration/censoring/cap metrics are preserved in summary.json. Analysis does not train, alter policies, choose another candidate or deploy a winner.', '');
  fs.writeFileSync(path.join(directory, 'summary.md'), lines.join('\n'));
  console.log('HELDOUT_ANALYSIS pairs=128 groups=4 seed_clusters=8 bootstrap_draws=' + BOOTSTRAP_DRAWS);
  for (const group of names) {
    const row = results[group], p = row.paired;
    console.log(group + ': ' + number(row.before.food_per_minute) + ' -> ' + number(row.after.food_per_minute) +
      ' food/min; delta ' + signed(p.delta_food_per_minute) + ' [' + signed(p.ci95[0]) + ', ' + signed(p.ci95[1]) + ']; ' + p.evidence +
      '; immobile cases ' + row.before.immobility.confirmed_cases + ' -> ' + row.after.immobility.confirmed_cases);
  }
  return summary;
}

if (require.main === module) {
  try {
    requireThat(process.argv.length === 3, 'Usage: node tools/analyze_feeding_heldout.cjs STUDY_OUTPUT_DIRECTORY');
    analyze(path.resolve(process.argv[2]));
  } catch (error) { console.error('HELDOUT_ANALYSIS FAIL: ' + error.message); process.exitCode = 1; }
}
module.exports = { analyze };
