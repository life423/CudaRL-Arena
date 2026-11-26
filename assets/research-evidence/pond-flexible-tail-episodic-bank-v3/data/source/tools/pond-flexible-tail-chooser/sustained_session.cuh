#pragma once

#ifndef CUDARL_SESSION_TASK_ID
#error The sustained target must bind its task sources at configure time.
#endif

constexpr float session_actor_lr = .0003f;
constexpr float session_critic_lr = .001f;

struct SessionIdentity {
    uint64_t gait = 0, offset = 0, starts = 0;
};

chooser::SteeringMode read_steering(const std::string& mode) {
    if (mode == "held-heading") return chooser::SteeringMode::PersistentHeading;
    if (mode == "measured-course") return chooser::SteeringMode::MeasuredCourse;
    throw std::runtime_error("Steering must be held-heading or measured-course");
}

chooser::MotorFeedbackMode read_motor_feedback(const std::string& mode) {
    if (mode == "whole-beat") return chooser::MotorFeedbackMode::WholeBeat;
    if (mode == "half-second") return chooser::MotorFeedbackMode::HalfSecond;
    throw std::runtime_error("Motor feedback must be whole-beat or half-second");
}

chooser::HorizonMode read_horizon(const std::string& mode) {
    if (mode == "long-horizon") return chooser::HorizonMode::Long;
    if (mode == "original-horizon") return chooser::HorizonMode::Original;
    throw std::runtime_error("Horizon must be long-horizon or original-horizon");
}

const char* motor_feedback_name(chooser::MotorFeedbackMode mode) {
    return mode == chooser::MotorFeedbackMode::WholeBeat ? "whole-beat" : "half-second";
}

const char* horizon_name(chooser::HorizonMode mode) {
    return mode == chooser::HorizonMode::Long ? "long-horizon" : "original-horizon";
}

void write_identity(cudarl::checkpoint::Writer& out, const SessionIdentity& identity) {
    out.value<uint32_t>(0x53435446); // FTCS
    out.value<uint32_t>(2);
    const std::string task(CUDARL_SESSION_TASK_ID);
    out.array(std::vector<char>(task.begin(), task.end()));
    out.value<uint32_t>(sizeof(Lane));
    out.value<uint32_t>(sizeof(Counts));
    for (const int value : {worlds, decisions, batch_size, epochs, macro_controls}) out.value(value);
    out.value(seed);
    out.value(host_steering_mode);
    out.value(host_motor_feedback_mode);
    out.value(host_horizon_mode);
    out.value(identity.gait);
    out.value(identity.offset);
    out.value(identity.starts);
    for (const float value : {gamma, lambda, max_kl, loss.clip, loss.ent_coef, loss.vf_coef,
                             session_actor_lr, session_critic_lr}) out.value(value);
    out.value(larger_pond);
}

SessionIdentity input_identity(const std::string& gait, const std::string& offset, const std::string& starts) {
    return {cudarl::checkpoint::checksum(cudarl::checkpoint::read(gait)),
        cudarl::checkpoint::checksum(cudarl::checkpoint::read(offset)),
        cudarl::checkpoint::checksum(cudarl::checkpoint::read(starts))};
}

std::vector<uint8_t> session_payload(Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, const Ledger& ledger, const SessionIdentity& identity,
    cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    cudarl::checkpoint::Writer out;
    write_identity(out, identity);
    out.value(ledger.completed);
    out.value(ledger.actor_steps);
    out.value(ledger.critic_steps);
    out.array(actor.get_params());
    out.array(critic.get_params());
    actor_adam.save(out);
    critic_adam.save(out);
    std::vector<Lane> lanes(worlds);
    std::vector<Counts> counters(worlds);
    roll.lanes.copy_to_host(lanes.data(), lanes.size());
    roll.counts.copy_to_host(counters.data(), counters.size());
    out.array(lanes);
    out.array(counters);
    return std::move(out.data);
}

void restore_session(const std::vector<uint8_t>& bytes, Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Ledger& ledger, const SessionIdentity& identity,
    cudaStream_t stream) {
    cudarl::checkpoint::Writer expected;
    write_identity(expected, identity);
    if (bytes.size() < expected.data.size() ||
        !std::equal(expected.data.begin(), expected.data.end(), bytes.begin())) {
        throw std::runtime_error("Session task, steering, frozen models, starts, or PPO configuration differs");
    }
    cudarl::checkpoint::Reader in{bytes};
    in.position = expected.data.size();
    const int completed = in.value<int>();
    const int actor_steps = in.value<int>();
    const int critic_steps = in.value<int>();
    if (completed < 0 || completed > max_updates || actor_steps < 0 || critic_steps != actor_steps)
        throw std::runtime_error("Invalid saved training progress");
    auto actor_weights = in.array<float>(actor.num_params());
    auto critic_weights = in.array<float>(critic.num_params());
    chooser::validate_weights(actor_weights);
    if (critic_weights.size() != size_t(critic.num_params()) ||
        !std::all_of(critic_weights.begin(), critic_weights.end(), cudarl::physics::finite_number)) {
        throw std::runtime_error("Invalid saved critic parameters");
    }
    actor_adam.load(in);
    critic_adam.load(in);
    if (actor_adam.steps() != actor_steps || critic_adam.steps() != critic_steps)
        throw std::runtime_error("Saved optimizer ages disagree with session progress");
    auto lanes = in.array<Lane>(worlds);
    auto counts = in.array<Counts>(worlds);
    in.finish();
    if (lanes.size() != worlds || counts.size() != worlds)
        throw std::runtime_error("Saved session world count differs");
    for (const auto& lane : lanes) {
        if (!cudarl::physics::finite_body(lane.state.navigation.motion.body) ||
            lane.state.steering_mode != host_steering_mode ||
            lane.state.motor_feedback_mode != host_motor_feedback_mode ||
            lane.state.horizon_mode != host_horizon_mode ||
            lane.state.navigation.motion.controls < 0 ||
            !std::isfinite(lane.state.applied_offset)) {
            // An untouched lane has no user-facing pose yet. Its state will
            // be initialized before first inference after restoring update 0.
            if (completed != 0 || !lane.needs_start)
                throw std::runtime_error("Saved native world is invalid");
        }
    }
    actor.set_params(actor_weights);
    critic.set_params(critic_weights);
    roll.lanes.copy_from_host(lanes.data(), lanes.size());
    roll.counts.copy_from_host(counts.data(), counts.size());
    ledger.completed = completed;
    ledger.started = completed;
    ledger.actor_steps = actor_steps;
    ledger.critic_steps = critic_steps;
    ledger.bank_curriculum = true;
    count_work(roll, ledger, stream);
}

void session_result(const fs::path& output, const Ledger& ledger, int target, double invocation_s,
                    const SessionIdentity& identity) {
    cudarl::checkpoint::Writer record;
    write_identity(record, identity);
    std::ofstream out(output / "result.json.tmp");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(10)
        << "{\"schema\":1,\"worlds\":" << worlds << ",\"rollout_decisions_per_world\":" << decisions
        << ",\"target_updates\":" << target << ",\"updates_completed\":" << ledger.completed
        << ",\"motor_feedback\":\"" << motor_feedback_name(host_motor_feedback_mode)
        << "\",\"horizon\":\"" << horizon_name(host_horizon_mode) << "\""
        << ",\"actor_adam_steps\":" << ledger.actor_steps << ",\"critic_adam_steps\":" << ledger.critic_steps
        << ",\"sampled_decisions\":" << uint64_t(ledger.counts.samples[0]) + ledger.counts.samples[1] + ledger.counts.samples[2]
        << ",\"sampled_keep\":" << ledger.counts.samples[0] << ",\"sampled_left\":" << ledger.counts.samples[1]
        << ",\"sampled_right\":" << ledger.counts.samples[2]
        << ",\"attempted_substeps\":" << ledger.counts.attempted + ledger.counts.warmup_attempted
        << ",\"accepted_substeps\":" << ledger.counts.accepted + ledger.counts.warmup_accepted
        << ",\"invocation_attempted_substeps\":" << ledger.executed_attempted
        << ",\"invocation_accepted_substeps\":" << ledger.executed_accepted
        << ",\"invocation_sampled_decisions\":" << ledger.executed_decisions
        << ",\"invocation_actor_adam_steps\":" << ledger.executed_actor_steps
        << ",\"invocation_critic_adam_steps\":" << ledger.executed_critic_steps
        << ",\"invocation_physical_allocation\":" << ledger.execution_physical_limit
        << ",\"clearance_stops\":" << ledger.counts.stops << ",\"cue_timeouts\":" << ledger.counts.cue_timeouts
        << ",\"posture_stops\":" << ledger.counts.posture_stops
        << ",\"stage\":\"" << ledger.stage << "\",\"invocation_wall_s\":" << invocation_s
        << ",\"gpu_physics_and_frozen_motor_ms\":" << ledger.gpu_physics_ms
        << ",\"gpu_chooser_inference_ms\":" << ledger.gpu_inference_ms
        << ",\"gpu_optimization_ms\":" << ledger.gpu_optimization_ms
        << ",\"synchronization_and_readback_ms\":" << ledger.transfers_ms
        << ",\"evidence_writing_ms\":" << ledger.evidence_ms
        << ",\"identity_checksum\":" << cudarl::checkpoint::checksum(record.data) << "}\n";
    out.close();
    // Keep the prior readable status intact if replacement fails.
#if defined(_WIN32)
    const auto old = output / "result.json";
    const auto temp = output / "result.json.tmp";
    if (!MoveFileExW(temp.c_str(), old.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Could not atomically replace session status");
#else
    fs::rename(output / "result.json.tmp", output / "result.json");
#endif
}

void publish_session(const fs::path& output, Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Ledger& ledger, const SessionIdentity& identity,
    cudaStream_t stream) {
    const auto evidence_start = Clock::now();
    const auto payload = session_payload(roll, actor, critic, actor_adam, critic_adam, ledger, identity, stream);
    const auto path = output / "checkpoints" / ("session-" + std::to_string(ledger.completed) + ".bin");
    if (fs::exists(path)) {
        if (cudarl::checkpoint::read(path.u8string()) != payload)
            throw std::runtime_error("Immutable session checkpoint differs from the restored state");
    } else cudarl::checkpoint::write_atomic(path.u8string(), payload);
    const auto actor_path = output / ("actor-" + std::to_string(ledger.completed) + ".bin");
    const auto weights = actor.get_params();
    if (fs::exists(actor_path)) {
        if (chooser::read_weights(actor_path.u8string()) != weights)
            throw std::runtime_error("Immutable actor export differs from the restored state");
    } else chooser::write_weights(actor_path.u8string(), weights);
    const auto metadata_path = output / ("actor-" + std::to_string(ledger.completed) + ".metadata.json");
    if (!fs::exists(metadata_path)) {
        std::ofstream metadata(metadata_path);
        metadata.exceptions(std::ios::failbit | std::ios::badbit);
        metadata << "{\"schema\":1,\"motor_feedback\":\"" << motor_feedback_name(host_motor_feedback_mode)
            << "\",\"horizon\":\"" << horizon_name(host_horizon_mode)
            << "\",\"steering\":\"" << (host_steering_mode == chooser::SteeringMode::PersistentHeading ? "held-heading" : "measured-course")
            << "\",\"update\":" << ledger.completed << ",\"task_identity\":\"" << CUDARL_SESSION_TASK_ID << "\"}\n";
    }
    cudarl::checkpoint::write_atomic((output / "latest.checkpoint").u8string(), payload);
    chooser::write_weights((output / "actor-final.bin").u8string(), actor.get_params());
    ledger.evidence_ms += elapsed(evidence_start) * 1000.;
}

void run_session(const std::string& gait_path, const std::string& offset_path, const std::string& starts_path,
    const char* initial_actor, const fs::path& output, int target_updates, double wall_budget,
    chooser::SteeringMode mode, bool verify_resume,
    chooser::MotorFeedbackMode feedback_mode = chooser::MotorFeedbackMode::HalfSecond,
    chooser::HorizonMode horizon_mode = chooser::HorizonMode::Original) {
    if (target_updates < 1 || target_updates > max_updates || !std::isfinite(wall_budget) || wall_budget <= 0)
        throw std::runtime_error("Invalid session update or wall budget");
    const bool resume = initial_actor == nullptr;
    if (!resume) {
        if (!fs::create_directory(output)) throw std::runtime_error("New session output must not already exist");
        fs::create_directory(output / "checkpoints");
    } else if (!fs::is_directory(output / "checkpoints")) {
        throw std::runtime_error("Resume requires an existing session directory");
    }
    cudarl::checkpoint::Lease lease((output / "session-owner").u8string());
    const auto invocation_start = Clock::now();
    const auto identity = input_identity(gait_path, offset_path, starts_path);
    host_steering_mode = mode;
    host_motor_feedback_mode = feedback_mode;
    host_horizon_mode = horizon_mode;
    CUDA_CHECK_THROW(cudaMemcpyToSymbol(session_steering_mode, &mode, sizeof(mode)));
    CUDA_CHECK_THROW(cudaMemcpyToSymbol(session_motor_feedback_mode, &feedback_mode, sizeof(feedback_mode)));
    CUDA_CHECK_THROW(cudaMemcpyToSymbol(session_horizon_mode, &horizon_mode, sizeof(horizon_mode)));
    const auto gait_weights = tail::read_navigation_weights(gait_path, tail::NavigationMode::Gait);
    const auto offset_weights = tail::read_yaw_offset_weights(offset_path);
    chooser::validate_frozen_layout(gait_weights, offset_weights);
    const auto initial = tail::read_navigation_worlds(starts_path);
    if (initial.size() != 256) throw std::runtime_error("Expected the archived training-only 256 poses");

    Ledger ledger;
    ledger.execution_physical_limit = verify_resume
        ? uint64_t(3) * worlds * decisions * 2 * macro_controls * tail::turn_substeps_per_control
        : max_substeps;
    ledger.bank_curriculum = true;
    verify_frozen_motor(initial, gait_weights, offset_weights, ledger);
    size_t stack = 0;
    CUDA_CHECK_THROW(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
    CUDA_CHECK_THROW(cudaDeviceSetLimit(cudaLimitStackSize, (std::max)(stack, size_t{16384})));
    Stream stream;
    Events events;
    Rollout roll;
    Batch batch;
    device_ptr<tail::NavigationWorld> starts(initial.size());
    device_ptr<float> gait(gait_weights.size()), offset(offset_weights.size());
    starts.copy_from_host(initial.data(), initial.size());
    gait.copy_from_host(gait_weights.data(), gait_weights.size());
    offset.copy_from_host(offset_weights.data(), offset_weights.size());
    nn::GpuMlp actor(chooser::actor_spec(), batch_size, seed);
    nn::GpuMlp critic({{channels, 32, 1}, {nn::Activation::Tanh, nn::Activation::Identity}}, batch_size, seed + 1);
    nn::AdamConfig actor_cfg, critic_cfg;
    actor_cfg.lr = session_actor_lr;
    critic_cfg.lr = session_critic_lr;
    actor_cfg.max_grad_norm = critic_cfg.max_grad_norm = 1.f;
    nn::GpuAdam actor_adam(actor.num_params(), actor_cfg), critic_adam(critic.num_params(), critic_cfg);
    constexpr int no_fault = INT_MAX;
    CUDA_CHECK_THROW(cudaMemcpyAsync(roll.first_fault.get(), &no_fault, sizeof(no_fault), cudaMemcpyHostToDevice, stream.value));
    zero(roll.fault_code.get(), roll.fault_code.size_bytes(), stream.value);
    zero(roll.zero.get(), roll.zero.size_bytes(), stream.value);
    if (resume) {
        restore_session(cudarl::checkpoint::read((output / "latest.checkpoint").u8string()), roll,
            actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
        publish_session(output, roll, actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
    } else {
        actor.set_params(chooser::read_weights(initial_actor));
        zero(roll.lanes.get(), roll.lanes.size_bytes(), stream.value);
        zero(roll.counts.get(), roll.counts.size_bytes(), stream.value);
        initialize_lanes<<<(worlds + lane_threads - 1) / lane_threads, lane_threads, 0, stream.value>>>(roll.lanes.get());
        check_launch();
        publish_session(output, roll, actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
    }

    cudaDeviceProp device{};
    int device_index = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&device_index));
    CUDA_CHECK_THROW(cudaGetDeviceProperties(&device, device_index));
    std::cout << "device=" << device.name << " worlds=" << worlds << " decisions_per_world=" << decisions
        << " motor_feedback=" << motor_feedback_name(feedback_mode) << " horizon=" << horizon_name(horizon_mode)
        << " restored_update=" << ledger.completed << " target_updates=" << target_updates << std::endl;
    const auto progress_path = output / "updates.csv";
    const bool had_progress = fs::exists(progress_path);
    if (had_progress) {
        // A crash can leave a flushed metric row immediately before its
        // checkpoint commit. Retain only rows covered by restored state.
        std::ifstream previous(progress_path);
        std::string line, reconciled;
        if (std::getline(previous, line)) reconciled = line + '\n';
        while (std::getline(previous, line)) {
            const auto comma = line.find(',');
            if (comma != std::string::npos && std::stoi(line.substr(0, comma)) <= ledger.completed)
                reconciled += line + '\n';
        }
        previous.close();
        std::ofstream repaired(progress_path, std::ios::trunc);
        repaired.exceptions(std::ios::failbit | std::ios::badbit);
        repaired << reconciled;
    }
    std::ofstream progress(progress_path, std::ios::app);
    progress.exceptions(std::ios::failbit | std::ios::badbit);
    progress << std::setprecision(10);
    if (!had_progress) progress << "update,policy_samples,reward_mean,exact_kl,policy_loss,value_loss,entropy,clip_fraction,"
        "explained_variance,adam_minibatches,attempted_substeps,accepted_substeps,clearance_stops,cue_timeouts,"
        "posture_stops,keep,left,right,gpu_physics_motor_ms,gpu_chooser_ms,gpu_optimization_ms,"
        "synchronization_readback_ms,evidence_cumulative_before_commit_ms,invocation_wall_s\n";
    std::vector<uint8_t> checkpoint_one;
    std::vector<uint8_t> continuous_two;
    bool replaying = false;
    bool verification_complete = false;
    try {
        while (ledger.completed < target_updates) {
            if (!verify_resume && (elapsed(invocation_start) >= wall_budget || fs::exists(output / "stop.request"))) {
                ledger.stage = "checkpointed-stop";
                break;
            }
            const int update = ledger.completed + 1;
            const auto physics_before = ledger.gpu_physics_ms;
            const auto inference_before = ledger.gpu_inference_ms;
            const auto transfers_before = ledger.transfers_ms;
            collect(roll, actor, critic, starts, gait, offset, ledger, update, stream.value, events, invocation_start);
            const float reward = finite_mean(roll, roll.reward.get(), rows, stream.value);
            CUDA_CHECK_THROW(cudaEventRecord(events.start, stream.value));
            const float kl = optimize(roll, batch, actor, critic, actor_adam, critic_adam, ledger, update, stream.value);
            CUDA_CHECK_THROW(cudaEventRecord(events.end, stream.value));
            CUDA_CHECK_THROW(cudaEventSynchronize(events.end));
            float optimization_ms = 0;
            CUDA_CHECK_THROW(cudaEventElapsedTime(&optimization_ms, events.start, events.end));
            ledger.gpu_optimization_ms += optimization_ms;
            ledger.completed = update;
            ledger.wall_s = elapsed(invocation_start);
            ledger.stage = "checkpoint";
            if (verify_resume) {
                const auto payload = session_payload(roll, actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
                if (update == 1) checkpoint_one = payload;
                else if (!replaying) {
                    continuous_two = payload;
                    cudarl::checkpoint::write_atomic((output / "continuous-2.bin").u8string(), payload);
                    cudarl::checkpoint::write_atomic((output / "checkpoint-1.bin").u8string(), checkpoint_one);
                    restore_session(cudarl::checkpoint::read((output / "checkpoint-1.bin").u8string()), roll,
                        actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
                    replaying = true;
                    continue;
                } else {
                    cudarl::checkpoint::write_atomic((output / "resumed-2.bin").u8string(), payload);
                    const bool identical = payload == continuous_two;
                    std::ofstream verification(output / "verification.json");
                    verification << "{\"pass\":" << (identical ? "true" : "false")
                        << ",\"worlds\":" << worlds << ",\"physical_updates_executed\":3,"
                        << "\"logical_final_update\":2,\"steering_mode\":\""
                        << (mode == chooser::SteeringMode::PersistentHeading ? "held-heading" : "measured-course")
                        << "\",\"task_identity\":\"" << CUDARL_SESSION_TASK_ID
                        << "\",\"motor_feedback\":\"" << motor_feedback_name(feedback_mode)
                        << "\",\"horizon\":\"" << horizon_name(horizon_mode)
                        << "\",\"actual_attempted_substeps\":" << ledger.executed_attempted
                        << ",\"actual_accepted_substeps\":" << ledger.executed_accepted
                        << ",\"actual_sampled_decisions\":" << ledger.executed_decisions
                        << ",\"actual_actor_adam_steps\":" << ledger.executed_actor_steps
                        << ",\"actual_critic_adam_steps\":" << ledger.executed_critic_steps
                        << ",\"physical_allocation\":" << ledger.execution_physical_limit
                        << ",\"compared\":[\"actor\",\"critic\",\"actor_adam\",\"critic_adam\",\"worlds\",\"counters\",\"rng_progress\"],"
                        << "\"uninterrupted_checksum\":"
                        << cudarl::checkpoint::checksum(continuous_two) << ",\"resumed_checksum\":"
                        << cudarl::checkpoint::checksum(payload) << ",\"wall_s\":" << elapsed(invocation_start) << "}\n";
                    if (!identical) throw std::runtime_error("Resume changes full deterministic training state");
                    verification_complete = true;
                }
            }
            const auto& count = ledger.counts;
            progress << update << ',' << uint64_t(update) * rows << ',' << reward << ',' << kl
                << ',' << ledger.policy_loss << ',' << ledger.value_loss << ',' << ledger.entropy
                << ',' << ledger.clip_fraction << ',' << ledger.explained_variance << ',' << ledger.minibatches
                << ',' << count.attempted + count.warmup_attempted << ',' << count.accepted + count.warmup_accepted
                << ',' << count.stops << ',' << count.cue_timeouts << ',' << count.posture_stops
                << ',' << count.samples[0] << ',' << count.samples[1] << ',' << count.samples[2]
                << ',' << ledger.gpu_physics_ms - physics_before << ',' << ledger.gpu_inference_ms - inference_before
                << ',' << optimization_ms << ',' << ledger.transfers_ms - transfers_before
                << ',' << ledger.evidence_ms << ',' << elapsed(invocation_start) << '\n' << std::flush;
            publish_session(output, roll, actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
            session_result(output, ledger, target_updates, elapsed(invocation_start), identity);
            std::cout << "update=" << update << " samples=" << uint64_t(update) * rows << " reward=" << reward
                << " exact_kl=" << kl << " stops=" << count.stops << " physics_ms=" << ledger.gpu_physics_ms - physics_before
                << " wall_s=" << elapsed(invocation_start) << std::endl;
        }
        if (verify_resume && !verification_complete)
            throw std::runtime_error("Full resume verification did not complete its comparison");
        if (ledger.completed == target_updates) ledger.stage = "complete";
        session_result(output, ledger, target_updates, elapsed(invocation_start), identity);
    } catch (...) {
        ledger.stage = "fault";
        session_result(output, ledger, target_updates, elapsed(invocation_start), identity);
        throw;
    }
}

int sustained_main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "new" && argc >= 10 && argc <= 12) {
            run_session(argv[2], argv[3], argv[4], argv[5], fs::u8path(argv[6]), std::stoi(argv[7]),
                std::stod(argv[8]), read_steering(argv[9]), false,
                argc > 10 ? read_motor_feedback(argv[10]) : chooser::MotorFeedbackMode::HalfSecond,
                argc > 11 ? read_horizon(argv[11]) : chooser::HorizonMode::Original);
        } else if (mode == "resume" && argc >= 9 && argc <= 11) {
            run_session(argv[2], argv[3], argv[4], nullptr, fs::u8path(argv[5]), std::stoi(argv[6]),
                std::stod(argv[7]), read_steering(argv[8]), false,
                argc > 9 ? read_motor_feedback(argv[9]) : chooser::MotorFeedbackMode::HalfSecond,
                argc > 10 ? read_horizon(argv[10]) : chooser::HorizonMode::Original);
        } else if (mode == "verify-resume" && argc >= 8 && argc <= 10) {
            run_session(argv[2], argv[3], argv[4], argv[5], fs::u8path(argv[6]), 2, 1800.,
                read_steering(argv[7]), true,
                argc > 8 ? read_motor_feedback(argv[8]) : chooser::MotorFeedbackMode::HalfSecond,
                argc > 9 ? read_horizon(argv[9]) : chooser::HorizonMode::Original);
        } else {
            throw std::runtime_error("Usage: session new FITTED OFFSET STARTS INITIAL_ACTOR NEW_DIR TARGET_UPDATES WALL_SECONDS STEERING | "
                "session resume FITTED OFFSET STARTS SESSION_DIR TARGET_UPDATES WALL_SECONDS STEERING | "
                "session verify-resume FITTED OFFSET STARTS INITIAL_ACTOR NEW_DIR STEERING "
                "[whole-beat|half-second] [long-horizon|original-horizon] (optional trailing modes for every command)");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
