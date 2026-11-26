#include "rl/flexible_tail_navigation.h"
#include <cmath>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    namespace tail = cudarl::flexible_tail;
    namespace fs = std::filesystem;
    try {
        if (argc != 3) throw std::runtime_error("Expected the held-out straight archive and a new output directory");
        const auto original = tail::read_navigation_worlds(argv[1]);
        if (original.size() != 12 || !fs::create_directory(argv[2]))
            throw std::runtime_error("Expected 12 held-out poses and an unused fixture directory");
        for (const int mm : {72, 75}) {
            auto starts = original;
            // Only these two poses seed the eight tangent approaches; row zero
            // still supplies the original fixed-posture scoring denominator.
            for (const int phase : {8, 9}) {
                auto& pose = starts.at(phase);
                const float source_y = pose.legs[pose.active_leg].goal_m.y;
                if (std::abs(source_y + .080f) > .00001f || pose.motion.controls != 0)
                    throw std::runtime_error("Unexpected source depth or time for the held-out pose");
                const float y = -mm * .001f;
                const float dy = y - source_y;
                pose.motion.body.position.y += dy / cudarl::physics::scale;
                for (auto& point : pose.motion.history) point.y += dy;
                for (int i = 0; i < pose.leg_count; ++i) pose.legs[i].goal_m.y += dy;
                tail::prepare_navigation(pose);
                if (std::abs(tail::snapshot(pose.motion.body).com_m.y-y) > .00001f)
                    throw std::runtime_error("Misplaced held-out pose");
            }
            const auto path = fs::path(argv[2]) / ("shore-" + std::to_string(mm) + ".bin");
            tail::write_navigation_worlds(path.u8string(), starts);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
