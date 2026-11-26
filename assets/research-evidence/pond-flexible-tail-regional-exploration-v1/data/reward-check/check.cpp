#include "region_credit.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>
using namespace cudarl::flexible_tail;
int main() {
    constexpr int cells = chooser::cells_x * chooser::cells_z;
    std::mt19937 rng(42);
    int checks = 0;
    for (int round=0; round<5; ++round) {
        chooser::World w{};
        std::array<std::set<std::pair<int,int>>,4> prior{};
        std::vector<int> order(cells);
        std::iota(order.begin(),order.end(),0);
        if (round) std::shuffle(order.begin(),order.end(),rng);
        for (int id : order) {
            const int before = w.visits++;
            w.visited[id/64] |= uint64_t{1} << (id%64);
            w.recent[w.recent_index].cell=id;
            w.recent_index=(w.recent_index+1)%chooser::recent_cells;
            int expected=0;
            for (int layout=0;layout<4;++layout) {
                const auto tile=std::make_pair((id%chooser::cells_x+(layout%2))/3,(id/chooser::cells_x+(layout/2))/3);
                if (prior[layout].insert(tile).second && before>=8) ++expected;
            }
            if (regional::new_regions(w,before)!=expected || regional::new_regions(w,w.visits)!=0) return 1;
            checks+=2;
        }
    }
    std::cout << "passed " << checks << " independent region and repeat checks across five full-grid traversals\n";
}
