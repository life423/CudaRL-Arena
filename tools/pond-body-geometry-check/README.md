# Pure whole-body pond geometry checks

These utilities execute geometry queries only: no physical substeps, policy
inference, training or optimizer calls. CPU build uses `/W4 /WX` on MSVC.

```powershell
cmake -S tools/pond-body-geometry-check -B build/pb26/native
cmake --build build/pb26/native --config Release
& build/pb26/native/Release/pond_body_geometry_check.exe
```

The query covers the head ellipsoid, rod cylinders and the original composite
tail ellipse **union** the actual thin membrane box. The explicit collision
envelope version2 fixes membrane corners outside the inherited ellipse without
changing mass, inertia or the old proxy dimensions. Both cylinder end discs and
all six membrane box faces are covered.

Patch lower bounds follow continuous surface coverage and analytic terrain
gradient/Hessian bounds. A finite sample minimum alone is never a certificate.
The exact flat-floor path uses the convex flat ellipse to certify the entire
bounding rectangle. An exhausted visit cap returns uncertified; the tangent
fixture deliberately demonstrates this at32,768visits. The product protocol
uses its separately fixed131,072visit cap. Surface/dry-land scope failures are
explicit; the fully submerged query does not provide water-exit physics.

The standalone CPU checker compares against dense independent surface searches,
checks derivative bounds and includes a concave-bank rod whose end-box corners
are clear while its middle penetrates. Dense searches corroborate the proof;
they do not establish it by themselves. Float contact-point rounding is exposed
separately from the double certificate arithmetic margin.

Optional CUDA compilation and one pure query kernel:

```powershell
cmake -S tools/pond-body-geometry-check -B build/pb26/cuda -DCUDARL_BODY_GEOMETRY_CUDA=ON
cmake --build build/pb26/cuda --config Release --target pond_body_geometry_cuda_check
& build/pb26/cuda/Release/pond_body_geometry_cuda_check.exe
```

This targets sm120 and explicitly disables FMA contraction. Sixteen fixed states
use byte-identical host/device input. Criteria are fixed in source: matching
validity/certification/reason, overlapping certified intervals for every link,
and sampled upper-gap differences at most1µm. This is not a test of CUDA physical
integration, fluid arithmetic, PPO or retained learned behavior. Preserve failed
attempts and distinguish geometry kernels from physical substeps.
