#include <chrono>
#include <cstdio>
#include "demos/pond_playback.h"
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::invalid_argument("pond_player_bench POLICY [auto|cpu|cuda] [CUDA_MODULE] [STEPS=1200]");
        const auto policy=cudarl::rl::SelectorPolicy::load(argv[1]);const int steps=argc>4?std::stoi(argv[4]):1200;
        if(steps<1) throw std::invalid_argument("Steps must be positive");
        auto player=cudarl::demos::create_pond_playback(policy,argc>2?argv[2]:"auto",argc>3?argv[3]:"");
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<steps;++i) player->step();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::printf("POND_PLAYER backend=%s steps=%d foods=%d simulation_seconds=%.2f wall_seconds=%.6f ms_per_step=%.6f realtime_x=%.2f\n%s\n",
            player->backend(),steps,player->frame().state.eaten,steps*.05,seconds,seconds*1000/steps,steps*.05/seconds,player->detail().c_str());return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Pond player: %s\n",e.what());return 1;}
}
