# Greedy decisions with extra room to turn

The selected new update +104 qualified on 6/12 previously tested placements with 60 mm more room and 6/12 with 120 mm more room. Each qualifying swimmer completed 180 seconds with positive whole-body clearance and retained motion. Each of the twelve remaining swimmers stopped on clearance. This assessment started after selection locked SHA-256 `16B096276A93E43E46EA2A62D2A9AFD9CD81E1E793EBAC43DFBEAB9587B1927F`.

| Upstream translation | New +104 | Previous +104 | +112 | Three-depth parent 128 | Any passing branch from the earlier assay |
|---|---:|---:|---:|---:|---:|
| 60 mm | 6/12 | 3/12 | 5/12 | 4/12 | 6/12 |
| 120 mm | 6/12 | 6/12 | 7/12 | 12/12 | 12/12 |

The new chooser passed each of the six 60-mm starts with a previously demonstrated recovery. At 120 mm it passed A and C at every depth. It stopped on B and D at all three depths after 59.93–121.96 seconds. The three-depth parent qualified on all six of those B/D cases.

On the 60-mm translation of approach D at 80 mm depth, the new and previous +104 both requested 12 mm/s at +8 degrees/s on the first cue. Their first difference occurred at 1.6 seconds. The new chooser reduced requested speed to 6 mm/s with the same turn rate. The previous chooser retained 12 mm/s. The new chooser went on to complete 180 seconds with 4.154 mm minimum gap. The previous chooser stopped at 76.68125 seconds. Subsequent actions belonged to each chooser, which limits the causal claim about the first difference.

The handoff table includes initial speed/rate, first disagreement with each archived chooser, survival, minimum clearance and previous successful alternatives for each swimmer. Raw files give stall controls, transport speed, tilt and depth error. Disagreement is evaluated before the two policies first diverge. [All 24 paired results](comparison.csv)

All cases used the original placement code, source fixture hashes, fitted gait, motor 88, habitat and first 0.8-second warmup. Spawn gap, cue gap and cue speed matched all 24 archived cases to within 0.000001 in their recorded mm or mm/s units. All 72 old first actions matched independent inference. The raw C++ world checksums differed between executable builds for all 24 cases. That checksum includes struct padding. Archived field measurements establish matched placement and observed motion at the cue. Byte-level identity across builds remains unverified.

The three candidate processes attempted 2,131,626 native substeps and took 7.26 seconds total. This descriptive supplement ran zero optimizer updates, and its results play no role in checkpoint selection or Godot publication. [Run summary](summary.json) · [Bound inputs](bindings.json) · [Frozen protocol](protocol.md)
