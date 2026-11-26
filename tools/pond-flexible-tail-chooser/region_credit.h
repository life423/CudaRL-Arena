#pragma once
#include "rl/flexible_tail_chooser.h"

namespace cudarl::flexible_tail::regional {
// The bit map already contains the just-earned first visit. Inspect its four
// overlapping 3 x 3 neighborhoods on two offset tilings along each axis.
// Each (tiling, region) can pay once per continuous training episode.
template<class Memory>
CUDARL_HD int new_regions(const Memory& w, int previous_visits) {
    if (previous_visits < 8 || w.visits != previous_visits + 1) return 0;
    const int cell = w.recent[(w.recent_index + chooser::recent_cells - 1) % chooser::recent_cells].cell;
    if (cell < 0 || cell >= chooser::cells_x * chooser::cells_z || !chooser::seen(w, cell)) return 0;
    const int x = cell % chooser::cells_x, z = cell / chooser::cells_x;
    int count = 0;
    for (int dz = 0; dz < 2; ++dz) for (int dx = 0; dx < 2; ++dx) {
        const int x0 = x - (x + dx) % 3, z0 = z - (z + dz) % 3;
        bool fresh = true;
        for (int j = z0; j < z0 + 3 && fresh; ++j) for (int i = x0; i < x0 + 3; ++i) {
            if (i < 0 || j < 0 || i >= chooser::cells_x || j >= chooser::cells_z) continue;
            const int neighbor = j * chooser::cells_x + i;
            if (neighbor != cell && chooser::seen(w, neighbor)) { fresh = false; break; }
        }
        count += fresh;
    }
    return count;
}
inline constexpr float reward_per_region = .30f;
}
