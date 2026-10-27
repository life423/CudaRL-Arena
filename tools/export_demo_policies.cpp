// Regenerate the two small actor-only demo policies using our C++/CUDA trainers.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "rl/ppo.h"
#include "rl/racing_ppo.h"
#include "demos/cartpole_showcase.h"
#include "demos/racing_showcase.h"

void save(const std::filesystem::path& directory,const char* demo,int inputs,int updates,
          const std::vector<float>& parameters) {
    const auto path=directory/(std::string(demo)+".json");
    std::ofstream file(path);
    file.exceptions(std::ios::failbit|std::ios::badbit);
    file << "{\n  \"version\": 1,\n  \"demo\": \"" << demo << "\",\n  \"inputs\": " << inputs
         << ",\n  \"hidden\": 64,\n  \"outputs\": 2,\n  \"seed\": 1,\n  \"updates\": " << updates
         << ",\n  \"parameters\": [\n" << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (size_t i=0;i<parameters.size();++i) {
        if (i) file << ',';
        if (i%8==0) file << "\n    ";
        file << parameters[i];
    }
    file << "\n  ]\n}\n";
    std::cout << "Saved " << path.string() << " (" << parameters.size() << " actor parameters)\n";
}

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::invalid_argument("Usage: export_demo_policies <output directory>");
        const std::filesystem::path directory(argv[1]);
        std::filesystem::create_directories(directory);
        cudarl::rl::PpoConfig cp_config;
        cudarl::rl::PpoTrainer cp(cp_config);
        for (int i=0;i<20;++i) cp.update();
        const auto cp_parameters=cp.actor_params();
        cudarl::demos::CartPoleShowcase carts(12,64,8);
        carts.set_policy(cp_parameters);
        for (int tick=0;tick<500;++tick) {
            carts.step(true);
            for (int i=0;i<12;++i) if (carts.just_failed(i)) throw std::runtime_error("CartPole playback validation failed");
        }
        save(directory,"cartpole",4,20,cp_parameters);

        cudarl::rl::RacingPpoConfig race_config;
        race_config.total_updates=1200;
        cudarl::rl::RacingPpoTrainer race(race_config);
        for (int i=0;i<200;++i) race.update();
        const auto race_parameters=race.actor_params();
        cudarl::demos::RacingShowcase cars(8,64,55);
        cars.set_policy(race_parameters);
        for (int tick=0;tick<1200;++tick) cars.step();
        if (cars.clean_laps()<8 || cars.contacts()!=0 || cars.wall_crashes()!=0)
            throw std::runtime_error("Racing playback validation failed");
        save(directory,"racing",30,200,race_parameters);
        std::cout << "Validated 12 CartPoles for 500 steps; racing " << cars.clean_laps()
                  << " clean laps, zero contacts/walls in 60 simulated seconds.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
