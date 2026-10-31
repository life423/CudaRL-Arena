#ifndef CUDARL_FROZEN_PPO_EVAL_TOOL_H
#define CUDARL_FROZEN_PPO_EVAL_TOOL_H
// Functions inside the existing native executable; no new world owner or solver.
namespace frozen_evaluation {
constexpr int case_controls = 1200, total_controls = 2400, total_substeps = 24000;
std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_s(&utc, &time);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
        << (std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000)
        << 'Z';
    return out.str();
}
std::vector<uint8_t> raw_file(const fs::path &file) {
    std::ifstream in(file, std::ios::binary);
    require(bool(in), "Missing frozen reference");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
struct Reference {
    float speed = 0, rmse = 0, lateral = 0, work = 0;
};
std::vector<Reference> references(const fs::path &file) {
    std::ifstream in(file);
    require(bool(in), "Missing four-second reference");
    std::string line;
    require(bool(std::getline(in, line)), "Empty reference");
    std::vector<Reference> out(4);
    bool seen[4]{};
    int count = 0;
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int pose = 0, passed = 0;
        Reference value;
        require(bool(row >> pose >> passed >> value.speed >> value.rmse >> value.lateral >> value.work) &&
                    pose >= 0 && pose < 4 && !seen[pose] && passed == 1,
                "Invalid frozen reference row");
        seen[pose] = true;
        out[pose] = value;
        ++count;
    }
    require(count == 4, "Incomplete frozen reference");
    return out;
}
std::vector<uint8_t> binding(const fs::path &checkpoint) {
    const auto dir = checkpoint.parent_path();
    cp::Writer out;
    out.value(uint32_t{0x31455650});
    out.value(uint32_t{1});
    for (int value : {6, 5, 2, case_controls, total_controls, total_substeps, 0, 3, 80})
        out.value(value);
    for (float value : {.008f, .6f, .4f, .0072f, .0088f, .0025f, .002f, .001f, 5.f, 10.f, .001f, .01f, 10.f})
        out.value(value);
    out.value(cp::checksum(cp::read(checkpoint.string())));
    out.value(cp::checksum(cp::read((dir / "training-protocol.bin").string())));
    out.value(cp::checksum(raw_file(dir / "reference.csv")));
    return out.data;
}
rl::FrozenPpoActor load(const fs::path &checkpoint) {
    const auto p = cp::read((checkpoint.parent_path() / "training-protocol.bin").string());
    require(p == protocol(), "Saved PPO training configuration differs from this runner");
    auto actor = rl::read_ppo_actor(cp::read(checkpoint.string()), cp::checksum(p), seed);
    require(actor.update == 64 && actor.actor_steps == 1210 && actor.critic_steps == 1210,
            "Wrong pinned PPO training milestone");
    return actor;
}
int prepare(const fs::path &root, const fs::path &source) {
    require(!fs::exists(root), "Evaluation identity already exists");
    fs::create_directories(root);
    fs::copy_file(source, root / "checkpoint.bin");
    fs::copy_file(source.parent_path() / "protocol.bin", root / "training-protocol.bin");
    fs::copy_file(source.parent_path() / "frozen-evaluation.csv", root / "reference.csv");
    const auto actor = load(root / "checkpoint.bin");
    (void)references(root / "reference.csv");
    require(actor.parameters.size() == static_cast<size_t>(spec(2).num_params()),
            "Actor architecture differs");
    cp::write_atomic((root / "evaluation-protocol.bin").string(), binding(root / "checkpoint.bin"));
    std::ofstream out(root / "evaluation-contract.json");
    out << "{\"ready\":true,\"algorithm\":\"frozen "
           "PPO\",\"optimizer_updates\":0,\"physical_steps\":0,\"poses\":[0,3],\"seconds_per_pose\":60,"
           "\"controls_cap\":2400,\"substeps_cap\":24000,\"native_wall_seconds\":10,\"process_wall_seconds\":"
           "15,\"backend\":\"CPU shared "
           "model6\",\"arena_m\":[1.2,0.8,0.16],\"original_benchmark_arena_m\":[0.36,0.28,0.16],\"resets_"
           "during_case\":0,\"wrap\":false,\"first4s\":\"Original80-control task and saved-result parity "
           "within printed precision\",\"after4s\":\"Original time feature remains0; no "
           "task/world/filter/phase reset; whole4s "
           "windows\",\"speed_bounds_mm_s\":[7.2,8.8],\"speed_RMSE_limit_mm_s\":2.5,\"lane_limit_mm_from_"
           "original_heading\":2,\"depth_limit_mm\":1,\"tilt_limit_deg\":10,\"COM_path_limit_deg\":5,\"work_"
           "limit_J_per_m\":0.001,\"clearance_min_mm\":10,\"accuracy_failure\":\"Record failure and continue "
           "diagnostic if physically safe; never "
           "reset\",\"stop_on\":\"Numerical,energy,actuator,domain,physical or budget/wall "
           "failure\",\"parity_speed_RMSE_lateral_tolerance_mm\":0.00002,\"parity_work_tolerance_J_per_m\":0."
           "000000002,\"checkpoint_and_actor_unchanged_required\":true,\"retries\":0}\n";
    out.close();
    status(root, "ready", {});
    std::ifstream immediate(root / "evaluation-contract.json");
    std::cout << immediate.rdbuf() << std::flush;
    return 0;
}
int run(const fs::path &root, const fs::path &checkpoint) {
    require(cp::read((root / "evaluation-protocol.bin").string()) == binding(checkpoint),
            "Frozen evaluation inputs changed");
    require(fs::create_directory(root / "run-started"), "Evaluation already attempted; no retry");
    started = std::chrono::steady_clock::now();
    Budget budget{0, 0, total_controls, total_substeps, 10};
    int parity_cases = 0, completed_cases = 0, passed_cases = 0, passed_windows = 0, windows = 0;
    bool hashes_unchanged = false;
    double inference_seconds = 0, physics_seconds = 0;
    float maximum_lateral = 0, maximum_depth = 0, maximum_tilt = 0, minimum_clearance = 1e6f;
    std::string fault, execution_started, execution_ended;
    std::ofstream trace(root / "control-trace.csv"), window_log(root / "windows.csv");
    trace << std::setprecision(9)
          << "pose,control,time_s,forward_m,lateral_m,depth_m,forward_velocity_m_s,speed_error_m_s,tilt_deg,"
             "head_yaw_deg,clearance_m,work_J,phase,actual_amplitude,actual_cadence,latent0,latent1,command0,"
             "command1";
    for (int j = 0; j < 42; ++j)
        trace << ",obs" << j;
    trace << '\n';
    window_log << std::setprecision(9)
               << "pose,end_time_s,basis,passed,speed_mm_s,RMSE_mm_s,lateral_mm,depth_mm,tilt_deg,COM_path_"
                  "deg,work_J_per_m\n";
    try {
        const auto bytes_before = cp::read(checkpoint.string());
        const auto saved = load(checkpoint);
        const auto reference = references(root / "reference.csv");
        nn::CpuMlp<float> actor(spec(2));
        actor.params() = saved.parameters;
        const auto config = m::config_for_speed(.008f);
        execution_started = utc_now();
        status(root, "frozen_evaluation", budget);
        std::cout << "FROZEN_EVALUATION_STARTED " << execution_started
                  << " no training; two uninterrupted60s cases" << std::endl;
        for (int pose : {0, 3}) {
            auto world = m::initial_world(pose);
            world.arena = {.6f, .4f};
            m::ForwardTask task;
            require(m::begin_forward(world, config, task), "Invalid long-duration start");
            auto window_start = cudarl::motor::motion(world.physical.body).com_model;
            float work_start = world.physical.body.work;
            double error_squared = 0;
            float lateral = 0, depth = 0, tilt = 0;
            bool case_passed = true;
            for (int tick = 1; tick <= case_controls; ++tick) {
                auto observed = task;
                observed.outcome = m::Outcome::Running;
                float obs[42]{};
                require(m::observe(observed, world, config, obs), "Invalid frozen long-duration observation");
                const auto inference_start = std::chrono::steady_clock::now();
                actor.forward(obs, 1);
                inference_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - inference_start).count();
                m::Action action;
                require(m::action_from_latent(actor.output(), action), "Invalid frozen action");
                const auto physical_start = std::chrono::steady_clock::now();
                budget.step(world, action.command);
                physics_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - physical_start).count();
                if (tick <= 80)
                    require(m::record_control(task, world, config).accepted,
                            "Original benchmark rejected interval");
                const auto metrics = m::measure(task, world);
                const auto motion = cudarl::motor::motion(world.physical.body);
                const float clearance = cudarl::movement::clearance(world.physical.body, true, world.arena);
                error_squared += metrics.speed_error_m_s * metrics.speed_error_m_s;
                lateral = fmaxf(lateral, metrics.lateral_m);
                depth = fmaxf(depth, metrics.depth_m);
                tilt = fmaxf(tilt, metrics.tilt_deg);
                maximum_lateral = fmaxf(maximum_lateral, lateral);
                maximum_depth = fmaxf(maximum_depth, depth);
                maximum_tilt = fmaxf(maximum_tilt, tilt);
                minimum_clearance = fminf(minimum_clearance, clearance);
                trace << pose << ',' << tick << ',' << tick * p::dt << ',' << metrics.forward_m << ','
                      << metrics.lateral_m << ',' << metrics.depth_m << ','
                      << p::dot(motion.velocity_m_s, task.request.heading) << ',' << metrics.speed_error_m_s
                      << ',' << metrics.tilt_deg << ',' << metrics.head_yaw_deg << ',' << clearance << ','
                      << world.physical.body.work << ',' << world.phase << ',' << world.parameter[0] << ','
                      << world.parameter[1];
                for (int j = 0; j < 2; ++j)
                    trace << ',' << action.latent[j];
                for (int j = 0; j < 2; ++j)
                    trace << ',' << action.command[j];
                for (float value : obs)
                    trace << ',' << value;
                trace << '\n';
                if (tick % 80 == 0) {
                    const auto delta = (motion.com_model - window_start) * p::scale;
                    const float forward = p::dot(delta, task.request.heading);
                    const auto across = delta - task.request.heading * forward;
                    float speed = forward / 4, rmse = sqrtf(static_cast<float>(error_squared / 80)),
                          angle =
                              atan2f(p::norm({across.x, 0, across.z}), forward) * 180 / cudarl::movement::pi;
                    const float work = (world.physical.body.work - work_start) / fmaxf(forward, 1e-9f);
                    bool passed = speed >= config.minimum_speed_m_s && speed <= config.maximum_speed_m_s &&
                                  rmse <= config.speed_rmse_limit && lateral <= config.lateral_limit_m &&
                                  depth <= config.depth_limit_m && tilt <= config.tilt_limit_deg &&
                                  angle <= config.path_angle_limit_deg &&
                                  work <= config.work_per_distance_limit;
                    if (tick == 80) {
                        speed = task.steady_speed_m_s;
                        rmse = task.speed_rmse;
                        angle = task.path_angle_deg;
                        passed = task.outcome == m::Outcome::Success;
                        const auto &ref = reference[pose];
                        require(passed && fabsf(speed * 1000 - ref.speed) <= 2e-5f &&
                                    fabsf(rmse * 1000 - ref.rmse) <= 2e-5f &&
                                    fabsf(task.maximum_lateral * 1000 - ref.lateral) <= 2e-5f &&
                                    fabsf(work - ref.work) <= 2e-9f,
                                "Four-second saved-reference parity failed; no continuation");
                        ++parity_cases;
                    }
                    ++windows;
                    passed_windows += passed ? 1 : 0;
                    case_passed = case_passed && passed;
                    window_log << pose << ',' << tick * p::dt << ','
                               << (tick == 80 ? "steady2-4s" : "whole4s") << ',' << passed << ','
                               << speed * 1000 << ',' << rmse * 1000 << ',' << lateral * 1000 << ','
                               << depth * 1000 << ',' << tilt << ',' << angle << ',' << work << '\n';
                    window_log.flush();
                    trace.flush();
                    status(root, "frozen_evaluation", budget);
                    window_start = motion.com_model;
                    work_start = world.physical.body.work;
                    error_squared = 0;
                    lateral = depth = tilt = 0;
                }
            }
            ++completed_cases;
            passed_cases += case_passed ? 1 : 0;
        }
        hashes_unchanged =
            actor.params() == saved.parameters && cp::read(checkpoint.string()) == bytes_before;
        require(hashes_unchanged, "Frozen checkpoint/actor changed");
        require(budget.controls == total_controls && budget.substeps == total_substeps,
                "Long evaluation accounting mismatch");
    } catch (const std::exception &error) {
        fault = error.what();
    }
    execution_ended = utc_now();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const bool passed = fault.empty() && completed_cases == 2 && passed_cases == 2 && passed_windows == 30 &&
                        hashes_unchanged;
    std::ofstream out(root / "result.json");
    out << std::setprecision(9)
        << "{\"algorithm\":\"frozen PPO\",\"execution_started_utc\":" << std::quoted(execution_started)
        << ",\"execution_ended_utc\":" << std::quoted(execution_ended) << ",\"fault\":" << std::quoted(fault)
        << ",\"passed\":" << (passed ? "true" : "false")
        << ",\"short_reference_parity_cases\":" << parity_cases << ",\"completed_cases\":" << completed_cases
        << ",\"passed_cases\":" << passed_cases << ",\"passed_windows\":" << passed_windows
        << ",\"windows\":" << windows << ",\"controls\":" << budget.controls
        << ",\"physical_substeps\":" << budget.substeps
        << ",\"optimizer_updates\":0,\"resets_during_case\":0,\"maximum_lateral_mm\":"
        << maximum_lateral * 1000 << ",\"maximum_depth_mm\":" << maximum_depth * 1000
        << ",\"maximum_tilt_deg\":" << maximum_tilt
        << ",\"minimum_clearance_mm\":" << minimum_clearance * 1000
        << ",\"actor_and_checkpoint_unchanged\":" << (hashes_unchanged ? "true" : "false")
        << ",\"CPU_inference_seconds\":" << inference_seconds
        << ",\"CPU_physics_seconds\":" << physics_seconds << ",\"CPU_physics_controls_per_second\":"
        << (physics_seconds > 0 ? budget.controls / physics_seconds : 0) << ",\"native_seconds\":" << seconds
        << ",\"closed\":true,\"retries\":0}\n";
    out.close();
    status(root, fault.empty() ? "done" : "failed", budget);
    std::ifstream immediate(root / "result.json");
    std::cout << immediate.rdbuf() << std::flush;
    return fault.empty() ? 0 : 3;
}
} // namespace frozen_evaluation
#endif
