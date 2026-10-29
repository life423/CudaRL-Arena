# Frozen networks at 0.75 cm/s — October 28, 2026

The latest saved swimmer passes the [predeclared suite](PROTOCOL.md): **100 CPU/CUDA sessions / 700 simulated minutes**, including calm and 0.5 cm/s references. Zero confirmed five-second immobility, invalid states, invalid food or skipped simulation steps. The largest food gap is **17.00 s**. There is no PPO training in these trials, no change to either network and no replacement of the saved actor.

## Conditions and results

Each backend uses eight fresh layout seeds `0xC0DECD01`–`0xC0DECD08` for five minutes per condition. The first two repeat for fifteen minutes. Conditions are calm, steady 0.5 cm/s, steady 0.75 cm/s, and 0.75 cm/s turning in either direction once per 120 simulated seconds. Food drifts with the imposed horizontal flow and retains its existing underwater clearance constraints. The same vector drives articulated-body water-relative drag. No reset, forced target switch or scripted propulsion is used.

| Backend | Minutes/case | Cases/condition | Calm | Steady 0.5 | Steady 0.75 | Turning toward +Z | Turning toward −Z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 5 | 8 | 18.13 | 17.70 | 17.55 | 17.25 | 17.45 |
| CPU | 15 | 2 | 17.80 | 17.67 | 17.70 | 17.83 | 17.87 |
| CUDA | 5 | 8 | 17.78 | 17.80 | 17.23 | 17.40 | 17.10 |
| CUDA | 15 | 2 | 17.77 | 17.80 | 17.47 | 17.77 | 17.17 |

Values are mean food/minute. Every stronger-current condition retains at least the declared 90% of its matched steady-0.5 mean; observed ratios are **96.07%–101.13%**. The CUDA conditions show a modest descriptive decrease, especially turning toward −Z. This is not a statistical ranking of controllers. Matched layouts may diverge after collection and respawns; CPU and CUDA may diverge over long trajectories. The [paired cases](paired.csv), [all cases](cases.csv), [aggregates](aggregates.csv) and [summary](summary.json) preserve those differences.

The stronger-flow conditions contain **705 sampled bank visits with at least one cumulative second of contact**. Of those, 702 clear the proxy for one continuous second before trial end. Three remain open at the cutoff (CUDA negative five-minute seed 1 at 293 s, steady fifteen-minute seed 2 at 896.55 s, negative fifteen-minute seed 1 at 898.50 s); they are censored, not counted as cleared. All 705 have a food collection within twenty seconds of bank entry, reported separately from clearance. Maximum cumulative contact is 3.85 s; maximum observed time to clear is 11.50 s. See [bank events](bank-events.csv). Ordinary contacts are not a deliberate difficult-state recovery benchmark, and earlier retained hard bank failures are not resolved by this result.

## Preserved controllers and evidence

Candidate: `gentle-currents/bank-recovery/selected-policy.bin`, selector update 320, swimming actor 64,350, SHA-256 `4186F47471EDE2D0034EB566EFA1EC959B03D0A015E686FFA5F4262FCEB4A563`. Both decision networks are frozen. The original bundled selector, intermediate policies and full learner saves remain separate and unchanged; [before](preserved-before.json) and [after](preserved-after.json) verify eleven protected references.

The existing `pond_reliability` runner and CUDA module are reused without another physical or inference implementation. Its 10,856 sampled current vectors agree with the intended simulation-time schedule within `1e-8 m/s` (maximum error `5.45e-9 m/s`). Raw mode metadata and sampled state/action/food traces are retained by backend/condition. No stall snapshots were triggered. Per-run `passes_provisional_gate` covers structural/immobility checks; the root summary also applies this protocol's twenty-second food-gap and matched-collection gates.

Reproduce with the existing release runner, a new output folder, all declared seeds and both durations/backends:

```powershell
& ./build/Release/pond_reliability.exe `
  --policy docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin `
  --progress --backend cuda `
  --module godot-project/addons/cudarl/bin/cudarl_pond_cuda.dll `
  --drifting-food --current-cm-s 0.75 --rotation-seconds 120 `
  --minutes 5 --seed 0xC0DECD01 --output build/my-current-075-trial
```

Period `0` is steady; `-120` reverses the turning direction. Reference speeds are `0` and `0.5`, with period zero. Repeat `--seed` for multiple layouts; CPU needs no optional module. Strict CUDA rejects fallback. These are frozen-agent research runs, not a learning-speed benchmark or post-training retention test.

## Visible pond and future disturbances

The user-facing menu is now focused on **Tadpole Pond**, with Open pond, Experiments and Help & training. CartPole/Racing code, policies, scenes, developer routing and training remain intact, without entries in the pond menu. The new preview is a render of the actual deep pond. Pond return controls say Pond menu. Project `config/name` remains `CudaRL Arena` so existing Godot user-data/training-save locations do not move. This is a UI focus change, not a project/save migration.

The live pond retains its existing **steady 0.5 cm/s switch**. The stronger/turning conditions are research tooling, not newly exposed player controls; no new UI strength slider or whirlpool is claimed. The tested schedule rotates the direction of a spatially uniform vector. A real whirlpool has different directions/speeds at different positions and needs a spatial flow model.

A rock entering the pond would need a local, time-varying disturbance and obstacle/contact model that affects both the creature and suspended food. Rendering extra ripples does not implement those forces. The next staged work should validate such a physical model, expose bounded interactions, and evaluate held-out disturbances. If the frozen agent fails, diagnose whether target choice or physical control is limiting, then train a copy with varied conditions and retain old skills before adoption. Keep a dependable saved creature in charge while practice is validated. No controller can be promised to handle arbitrary unseen physics, and sufficiently strong flow can make an upstream target physically unreachable. Rock interaction, spatial flow, new training and anatomy growth are not implemented here.

## UI verification

Rendered CUDA pond navigation passes **141 checks**, and headless CPU navigation passes **125**, across 1280×720, 768×1024, 390×844 and 844×390. Menu, help, experiments and the live pond remain reachable with usable controls and keyboard focus. The retained-demo regression passes **99 checks**, including direct developer routes, policy playback, training and session cleanup. Training exercised by that regression uses isolated test saves; it is separate from the frozen-agent research above. Escape cleanup now marks input handled before leaving the viewport, avoiding a stale-scene access. Final logs contain no script errors. No new complete native-suite run is claimed for this UI/CLI-only step.

Internal UI review: usability, hierarchy, consistency, responsiveness and feedback 9/10; accessibility and measured performance 8/10. Priorities are real assistive/touch-device verification and startup/frame-time measurement before distribution work. Screenshots are under ignored `build/pond-ui-checks/`. These are Windows window-size and synthetic-input checks, not physical mobile-device validation. The broader product remains a dependable young swimmer → challenging interactions → measured adaptation → growth and new movement skills.
