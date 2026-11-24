This standalone native CPU regression performs zero integration/force/inference/
optimizer/CUDA work. Pass the three existing eight-world raw snapshots from
ticks1,8,15 of the closed optimized GPU gate. It queries each saved GPU body once.

It checks gap-gradient and upward-normal signs against finite differences of a
separately written polar height formula, including512fixed wet points and36bank
link minima. The independent geometry oracle uses an explicit quaternion matrix
and direct ellipsoid/cylinder/disc/six-face-box parameterization, without the
adaptive query's support, patch, minimum-linear or traversal functions.

The12flat bodies have independent exact analytic support minima. For12bank
bodies, all complete surfaces are densely gridded. The reference is certified:
bilinear interpolation error is bounded by Muu*du²/8+Mvv*dv²/8, with parametric
second derivatives bounded using |grad H|<=3 and ||Hess H||<=150. These are
conservative roundings above the fixed pond's analytic bounds2.555405/132.244555.
Every complete parameter cell is covered; grid samples alone are not called a
certificate. A separate bounding sphere proves the reference stays wet. Each
reference interval must be narrower than2.5um; the unchanged query precision is
10um. Reference and query intervals must overlap for every link and fin union.

Returned gradient_x/z remain height derivatives. The shared contact runtime
continues using their negatives for the gap/contact gradient. The arithmetic
correction changes no collision dimensions, force equation or motor parameter.
