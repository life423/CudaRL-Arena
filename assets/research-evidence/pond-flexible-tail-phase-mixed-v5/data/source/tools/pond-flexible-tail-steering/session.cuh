#pragma once

#ifndef CUDARL_STEERING_TASK_ID
#error Bind the steering task and dynamics at configure time.
#endif

struct InputIdentity {
    uint64_t gait = 0, offset = 0, starts = 0;
    uint64_t initial_actor_weights = 0;
    std::array<float, tail::steering::base_parameters> initial_coefficients{};
    uint64_t parent_payload = 0, parent_initial_actor_weights = 0;
    std::string parent_task;
    int parent_completed = 0, parent_actor_steps = 0, parent_critic_steps = 0;
};

tail::steering::Model initial_policy(const std::string& offset) {
    auto source = tail::steering::read_model(offset);
    if (source.kind != tail::steering::Kind::Legacy4)
        throw std::runtime_error("A new steering study requires its declared preserved legacy source");
    if (host_feedback_mode == fast::FeedbackMode::PhaseAware)
        return tail::steering::initial_phase_model(source.weights);
    if (host_feedback_mode == fast::FeedbackMode::WholeBeat90)
        source.weights = fast::initial_actor90(source.weights);
    tail::steering::validate_model(source);
    return source;
}

void write_identity(cp::Writer& out, const InputIdentity& identity, bool parent = false) {
    out.value<uint32_t>(0x53535446); // FTSS: complete steering training session.
    const uint32_t version = !parent && host_practice != Practice::Original ? 5u : 4u;
    out.value(version);
    const std::string task = parent ? identity.parent_task : CUDARL_STEERING_TASK_ID;
    out.array(std::vector<char>(task.begin(), task.end()));
    out.value<uint32_t>(sizeof(Lane));
    out.value<uint32_t>(sizeof(Counts));
    for (const int value : {worlds, decisions, channels, critic_channels, batch_size, epochs,
                            action_controls, max_updates}) out.value(value);
    out.value(seed);
    out.value(host_feedback_mode);
    out.value(fast::policy_kind(host_feedback_mode));
    out.value(tail::steering::feature_schema(fast::policy_kind(host_feedback_mode)));
    for (const float value : {gamma, lambda, target_kl, hard_kl, actor_lr, critic_lr, log_std,
                              coefficients.clip, coefficients.ent_coef, coefficients.vf_coef}) out.value(value);
    out.value(larger_pond);
    out.value(identity.gait);
    out.value(identity.offset);
    out.value(identity.starts);
    out.value(parent ? identity.parent_initial_actor_weights : identity.initial_actor_weights);
    out.value(identity.initial_coefficients);
    if (version == 5) {
        out.value(host_practice);
        out.value(identity.parent_payload);
        out.array(std::vector<char>(identity.parent_task.begin(), identity.parent_task.end()));
        out.value(identity.parent_completed);
        out.value(identity.parent_actor_steps);
        out.value(identity.parent_critic_steps);
        out.value(identity.parent_initial_actor_weights);
    }
}

InputIdentity input_identity(const std::string& gait, const std::string& offset, const std::string& starts) {
    const auto initial = initial_policy(offset).weights;
    cp::Writer weights;
    weights.array(initial);
    InputIdentity result{cp::checksum(cp::read(gait)), cp::checksum(cp::read(offset)), cp::checksum(cp::read(starts))};
    result.initial_actor_weights = cp::checksum(weights.data);
    std::copy_n(initial.data(), result.initial_coefficients.size(), result.initial_coefficients.begin());
    return result;
}

void bind_parent(InputIdentity& identity, const std::vector<uint8_t>& bytes,
                 const std::string& parent_task) {
    if (host_feedback_mode != fast::FeedbackMode::PhaseAware ||
        host_practice != Practice::Mixed15And30 || parent_task.size() != 64)
        throw std::runtime_error("A fork requires the declared phase-aware mixed curriculum and parent task identity");
    identity.parent_task = parent_task;
    identity.parent_initial_actor_weights = identity.initial_actor_weights;
    cp::Writer expected;
    write_identity(expected, identity, true);
    if (bytes.size() < expected.data.size() ||
        !std::equal(expected.data.begin(), expected.data.end(), bytes.begin()))
        throw std::runtime_error("Fork parent task, layouts, frozen inputs, physics or PPO settings differ");
    cp::Reader in{bytes};
    in.position = expected.data.size();
    identity.parent_completed = in.value<int>();
    identity.parent_actor_steps = in.value<int>();
    identity.parent_critic_steps = in.value<int>();
    const auto rng_progress = in.value<uint64_t>();
    if (identity.parent_completed < 1 || identity.parent_completed > max_updates ||
        identity.parent_actor_steps < 0 || identity.parent_critic_steps != identity.parent_actor_steps ||
        rng_progress != uint64_t(identity.parent_completed) * decisions)
        throw std::runtime_error("Invalid fork parent sampling or optimizer progress");
    const auto imported_actor = in.array<float>(tail::steering::spec().num_params());
    tail::steering::validate_weights(imported_actor, tail::steering::Kind::PhaseAware13,
        identity.initial_coefficients.data());
    cp::Writer weights;
    weights.array(imported_actor);
    identity.initial_actor_weights = cp::checksum(weights.data);
    identity.parent_payload = cp::checksum(bytes);
    host_sampling_update_origin = identity.parent_completed;
}

std::vector<uint8_t> session_payload(Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, const Ledger& ledger,
    const InputIdentity& identity, cudaStream_t stream) {
    static_assert(std::is_trivially_copyable<Lane>::value && std::is_trivially_copyable<Counts>::value);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    cp::Writer out;
    write_identity(out, identity);
    out.value(ledger.completed);
    out.value(ledger.actor_steps);
    out.value(ledger.critic_steps);
    // Counter-based RNG has no hidden generator allocation. The next collection
    // stream is completed*64; every lane's episode draw position is saved below.
    out.value<uint64_t>(uint64_t(host_sampling_update_origin + ledger.completed) * decisions);
    out.array(actor.get_params());
    out.array(critic.get_params());
    actor_adam.save(out);
    critic_adam.save(out);
    std::vector<Lane> lanes(worlds);
    std::vector<Counts> counts(worlds);
    roll.lanes.copy_to_host(lanes.data(), lanes.size());
    roll.counts.copy_to_host(counts.data(), counts.size());
    out.array(lanes);
    out.array(counts);
    return std::move(out.data);
}

void restore_session(const std::vector<uint8_t>& bytes, Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Ledger& ledger,
    const InputIdentity& identity, cudaStream_t stream, bool parent = false) {
    cp::Writer expected;
    write_identity(expected, identity, parent);
    if (bytes.size() < expected.data.size() ||
        !std::equal(expected.data.begin(), expected.data.end(), bytes.begin()))
        throw std::runtime_error("Session task, frozen checkpoints, starts, geometry or PPO configuration differs");
    cp::Reader in{bytes};
    in.position = expected.data.size();
    const int completed = in.value<int>();
    const int actor_steps = in.value<int>();
    const int critic_steps = in.value<int>();
    const auto rng_progress = in.value<uint64_t>();
    if (completed < 0 || completed > max_updates || actor_steps < 0 || critic_steps != actor_steps ||
        rng_progress != uint64_t((parent ? 0 : host_sampling_update_origin) + completed) * decisions)
        throw std::runtime_error("Invalid saved optimizer or sampling progress");
    auto actor_weights = in.array<float>(actor.num_params());
    auto critic_weights = in.array<float>(critic.num_params());
    tail::steering::validate_weights(actor_weights, fast::policy_kind(host_feedback_mode),
        identity.initial_coefficients.data());
    if (critic_weights.size() != size_t(critic.num_params()) ||
        !std::all_of(critic_weights.begin(), critic_weights.end(), cudarl::physics::finite_number))
        throw std::runtime_error("Invalid saved critic parameters");
    const auto optimizer_begin = in.position;
    actor_adam.load(in);
    critic_adam.load(in);
    const auto optimizer_end = in.position;
    if (actor_adam.steps() != actor_steps || critic_adam.steps() != critic_steps ||
        actor_adam.config().lr != actor_lr || critic_adam.config().lr != critic_lr)
        throw std::runtime_error("Saved optimizer ages or learning rates disagree with session identity");
    auto lanes = in.array<Lane>(worlds);
    auto counts = in.array<Counts>(worlds);
    in.finish();
    if (lanes.size() != worlds || counts.size() != worlds)
        throw std::runtime_error("Saved world count differs");
    for (const auto& lane : lanes) {
        if (completed == 0 && lane.needs_start && lane.episode == 0) continue;
        if (!cudarl::physics::finite_body(lane.state.navigation.motion.body) ||
            !std::isfinite(lane.state.applied_offset) || !std::isfinite(lane.angle_rad) ||
            lane.state.navigation.motion.controls < 0 ||
            lane.state.navigation.motion.controls > fast::episode_controls ||
            fast::is_global_fault(lane.state.fault) ||
            lane.state.feedback_mode != host_feedback_mode ||
            lane.state.done != lane.needs_start)
            throw std::runtime_error("Saved native motor world is invalid");
    }
    actor.set_params(actor_weights);
    critic.set_params(critic_weights);
    roll.lanes.copy_from_host(lanes.data(), lanes.size());
    roll.counts.copy_from_host(counts.data(), counts.size());
    if (parent) {
        // This comparison is zero-step: prove the fork imported the exact live
        // body/history and both optimizers before rebasing evidence counters.
        std::vector<Lane> restored(worlds);
        roll.lanes.copy_to_host(restored.data(), restored.size());
        const auto restored_actor = actor.get_params(), restored_critic = critic.get_params();
        cp::Writer optimizers;
        actor_adam.save(optimizers);
        critic_adam.save(optimizers);
        if (std::memcmp(lanes.data(), restored.data(), lanes.size() * sizeof(Lane)) ||
            std::memcmp(actor_weights.data(), restored_actor.data(), actor_weights.size() * sizeof(float)) ||
            std::memcmp(critic_weights.data(), restored_critic.data(), critic_weights.size() * sizeof(float)) ||
            optimizers.data.size() != optimizer_end - optimizer_begin ||
            !std::equal(optimizers.data.begin(), optimizers.data.end(), bytes.begin() + optimizer_begin))
            throw std::runtime_error("Fork upload changed native worlds, policy, critic or optimizer state");
    }
    ledger.completed = completed;
    ledger.started = completed;
    ledger.actor_steps = actor_steps;
    ledger.critic_steps = critic_steps;
    count_work(roll, ledger, stream);
}

void immutable_checkpoint(const fs::path& path, const std::vector<uint8_t>& bytes) {
    if (fs::exists(path)) {
        if (cp::read(path.u8string()) != bytes)
            throw std::runtime_error("Immutable checkpoint already exists with different state");
    } else cp::write_atomic(path.u8string(), bytes);
}

void publish_session(const fs::path& output, Rollout& roll, nn::GpuMlp& actor, nn::GpuMlp& critic,
    nn::GpuAdam& actor_adam, nn::GpuAdam& critic_adam, Ledger& ledger,
    const InputIdentity& identity, cudaStream_t stream, int target, double budget_s, double used_s) {
    const auto start = Clock::now();
    const auto bytes = session_payload(roll, actor, critic, actor_adam, critic_adam, ledger, identity, stream);
    immutable_checkpoint(output / "checkpoints" / ("session-" + std::to_string(ledger.completed) + ".bin"), bytes);
    const auto actor_weights = actor.get_params();
    const auto kind = fast::policy_kind(host_feedback_mode);
    tail::steering::validate_weights(actor_weights, kind, identity.initial_coefficients.data());
    const auto actor_path = output / ("actor-" + std::to_string(ledger.completed) + ".bin");
    if (fs::exists(actor_path)) {
        const auto saved_actor = tail::steering::read_model(actor_path.u8string());
        if (saved_actor.kind != kind || saved_actor.weights != actor_weights)
            throw std::runtime_error("Immutable actor export already exists with different weights");
    } else tail::steering::write_model(actor_path.u8string(), actor_weights, kind,
        identity.initial_coefficients.data());
    const auto actor_metadata = output / ("actor-" + std::to_string(ledger.completed) + ".metadata.json");
    if (!fs::exists(actor_metadata)) {
        std::ofstream metadata(actor_metadata);
        metadata.exceptions(std::ios::failbit | std::ios::badbit);
        metadata << std::setprecision(12) << "{\"schema\":3,\"weight_format\":\""
            << (kind == tail::steering::Kind::PhaseAware13 ? "FTYO2" : "FTYO1")
            << "\",\"policy_kind\":" << static_cast<uint32_t>(kind)
            << ",\"feature_schema\":" << tail::steering::feature_schema(kind)
            << ",\"feature_layout\":\"" << tail::steering::feature_layout_name(kind)
            << "\",\"trainable_parameters\":" << tail::steering::trainable_count(kind)
            << ",\"frozen_base_parameters\":" << (kind == tail::steering::Kind::PhaseAware13
                ? tail::steering::base_parameters : 0)
            << ",\"feedback\":\""
            << feedback_name(host_feedback_mode) << "\",\"task_identity\":\"" << CUDARL_STEERING_TASK_ID
            << "\",\"update\":" << ledger.completed << ",\"frozen_gait_payload_checksum\":"
            << identity.gait << ",\"source_offset_payload_checksum\":" << identity.offset
            << ",\"initial_actor_weights_checksum\":" << identity.initial_actor_weights
            << ",\"practice\":\"" << practice_name(host_practice) << "\""
            << ",\"parent_payload_checksum\":" << identity.parent_payload
            << ",\"parent_update\":" << identity.parent_completed
            << ",\"sampling_update_origin\":" << host_sampling_update_origin
            << ",\"course_error_scale_deg\":" << (host_feedback_mode == fast::FeedbackMode::WholeBeat90 ? 90 : 5)
            << ",\"course_error_equation\":\""
            << (host_feedback_mode == fast::FeedbackMode::WholeBeat90
                ? "signed_yaw(whole_beat_COM_velocity,request_heading)/(pi/2)"
                : host_feedback_mode == fast::FeedbackMode::PhaseAware
                    ? "base0=clipped5deg;feature4=signed_yaw(whole_beat_COM_velocity,request_heading)/(pi/2)"
                    : "clamp(signed_yaw(measured_course,request_heading)/(5*pi/180),-4,4)") << "\""
            << ",\"initial_course_weight_multiplier\":" << (host_feedback_mode == fast::FeedbackMode::WholeBeat90 ? 18 : 1)
            << ",\"initial_coefficients\":[";
        for (size_t i = 0; i < identity.initial_coefficients.size(); ++i) {
            if (i) metadata << ',';
            metadata << identity.initial_coefficients[i];
        }
        metadata << "]}\n";
    }
    cp::write_atomic((output / "latest.checkpoint").u8string(), bytes);
    tail::steering::write_model((output / "actor-final.bin").u8string(), actor_weights, kind,
        identity.initial_coefficients.data());
    // Runtime accounting is deliberately outside deterministic physical state.
    // It binds to the exact checkpoint so a crash cannot silently reset a budget.
    cp::Writer budget;
    budget.value<uint32_t>(1);
    budget.value(cp::checksum(bytes));
    budget.value(target);
    budget.value(budget_s);
    budget.value(used_s + elapsed(start));
    cp::write_atomic((output / "budget.checkpoint").u8string(), budget.data);
    ledger.evidence_ms += elapsed(start) * 1000.;
}

double restore_budget(const fs::path& output, const std::vector<uint8_t>& checkpoint,
                       int target, double budget_s) {
    const auto bytes = cp::read((output / "budget.checkpoint").u8string());
    cp::Reader in{bytes};
    if (in.value<uint32_t>() != 1 || in.value<uint64_t>() != cp::checksum(checkpoint) ||
        in.value<int>() != target || in.value<double>() != budget_s)
        throw std::runtime_error("Resume budget or checkpoint binding differs; no silent budget extension is allowed");
    const auto used_s = in.value<double>();
    in.finish();
    if (!std::isfinite(used_s) || used_s < 0 || used_s > max_session_wall_s)
        throw std::runtime_error("Invalid saved runtime budget");
    return used_s;
}

void write_result(const fs::path& output, const Ledger& ledger, int target, double wall_s,
                   const InputIdentity& identity, bool verify_resume) {
    std::ofstream out(output / "result.json.tmp");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(12)
        << "{\"schema\":1,\"task_identity\":\"" << CUDARL_STEERING_TASK_ID
        << "\",\"stage\":\"" << ledger.stage << "\",\"worlds\":" << worlds
        << ",\"feedback\":\"" << feedback_name(host_feedback_mode) << "\""
        << ",\"rollout_decisions_per_world\":" << decisions << ",\"controls_per_decision\":" << action_controls
        << ",\"minibatch\":" << batch_size << ",\"passes_maximum\":" << epochs
        << ",\"target_updates\":" << target << ",\"updates_completed\":" << ledger.completed
        << ",\"updates_started\":" << ledger.started
        << ",\"optimizer_steps\":" << ledger.actor_steps
        << ",\"practice\":\"" << practice_name(host_practice) << "\""
        << ",\"parent_payload_checksum\":" << identity.parent_payload
        << ",\"parent_update\":" << identity.parent_completed
        << ",\"imported_actor_steps\":" << identity.parent_actor_steps
        << ",\"imported_critic_steps\":" << identity.parent_critic_steps
        << ",\"study_actor_steps\":" << ledger.actor_steps - identity.parent_actor_steps
        << ",\"study_critic_steps\":" << ledger.critic_steps - identity.parent_critic_steps
        << ",\"sampling_update_origin\":" << host_sampling_update_origin
        << ",\"logical_samples\":" << uint64_t(ledger.completed) * rows
        << ",\"sampled_actions\":" << ledger.counts.actions
        << ",\"attempted_substeps\":" << ledger.counts.attempted
        << ",\"accepted_substeps\":" << ledger.counts.accepted
        << ",\"invocation_sampled_actions\":" << ledger.executed_counts.actions
        << ",\"invocation_attempted_substeps\":" << ledger.executed_counts.attempted
        << ",\"invocation_accepted_substeps\":" << ledger.executed_counts.accepted
        << ",\"invocation_actor_steps\":" << ledger.executed_actor_steps
        << ",\"invocation_critic_steps\":" << ledger.executed_critic_steps
        << ",\"episode_starts\":" << ledger.counts.starts
        << ",\"episode_horizons\":" << ledger.counts.horizons
        << ",\"left_starts\":" << ledger.counts.left << ",\"right_starts\":" << ledger.counts.right
        << ",\"clearance_stops\":" << ledger.counts.clearance_stops
        << ",\"posture_stops\":" << ledger.counts.posture_stops
        << ",\"reliable_course_controls\":" << ledger.counts.reliable_course_controls
        << ",\"low_course_controls\":" << ledger.counts.low_course_controls
        << ",\"measured_beat_periods\":" << ledger.counts.measured_beat_periods
        << ",\"rejected_beat_periods\":" << ledger.counts.rejected_beat_periods
        << ",\"first_fault_lane\":" << (ledger.first_fault_lane == INT_MAX ? -1 : ledger.first_fault_lane)
        << ",\"first_fault_code\":" << ledger.first_fault_code
        << ",\"physics_and_observation_gpu_ms\":" << ledger.physics_ms
        << ",\"network_gpu_ms\":" << ledger.network_ms
        << ",\"optimization_gpu_ms\":" << ledger.optimization_ms
        << ",\"sync_and_readback_wall_ms\":" << ledger.transfers_ms
        << ",\"evidence_writing_wall_ms\":" << ledger.evidence_ms
        << ",\"total_session_wall_s\":" << wall_s
        << ",\"verification_mode\":" << (verify_resume ? "true" : "false")
        << ",\"frozen_gait_payload_checksum\":" << identity.gait
        << ",\"original_offset_payload_checksum\":" << identity.offset
        << ",\"initial_actor_weights_checksum\":" << identity.initial_actor_weights
        << ",\"policy_kind\":" << static_cast<uint32_t>(fast::policy_kind(host_feedback_mode))
        << ",\"feature_schema\":" << tail::steering::feature_schema(fast::policy_kind(host_feedback_mode))
        << ",\"feature_layout\":\"" << tail::steering::feature_layout_name(fast::policy_kind(host_feedback_mode)) << "\""
        << ",\"trainable_actor_parameters\":" << tail::steering::trainable_count(fast::policy_kind(host_feedback_mode))
        << ",\"frozen_actor_parameters\":" << (host_feedback_mode == fast::FeedbackMode::PhaseAware
            ? tail::steering::base_parameters : 0)
        << ",\"course_error_scale_deg\":" << (host_feedback_mode == fast::FeedbackMode::WholeBeat90 ? 90 : 5)
        << ",\"training_starts_payload_checksum\":" << identity.starts << "}\n";
    out.close();
#if defined(_WIN32)
    if (!MoveFileExW((output / "result.json.tmp").c_str(), (output / "result.json").c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Could not atomically replace training status");
#else
    fs::rename(output / "result.json.tmp", output / "result.json");
#endif
}

void reconcile_progress(const fs::path& path, int completed) {
    if (!fs::exists(path)) return;
    std::ifstream previous(path);
    std::string line, preserved;
    if (std::getline(previous, line)) preserved = line + '\n';
    while (std::getline(previous, line)) {
        const auto comma = line.find(',');
        if (comma != std::string::npos && std::stoi(line.substr(0, comma)) <= completed)
            preserved += line + '\n';
    }
    previous.close();
    std::ofstream out(path, std::ios::trunc);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << preserved;
}

void write_feedback_parity(const fs::path& path, const FeedbackParity& parity) {
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(12) << "{\"pass\":" << (parity.pass ? "true" : "false")
        << ",\"feedback\":\"" << feedback_name(host_feedback_mode) << "\",\"absolute_tolerance\":0.000001"
        << ",\"physics_steps\":0,\"optimizer_updates\":0,\"worlds\":" << worlds
        << ",\"actor_max_difference\":" << parity.actor_max_difference
        << ",\"critic_max_difference\":" << parity.critic_max_difference
        << ",\"nonfinite_values\":" << parity.nonfinite_values
        << ",\"actor_lane\":" << parity.actor_lane << ",\"actor_feature\":" << parity.actor_feature
        << ",\"actor_cpu\":" << parity.actor_cpu << ",\"actor_cuda\":" << parity.actor_cuda
        << ",\"critic_lane\":" << parity.critic_lane << ",\"critic_feature\":" << parity.critic_feature
        << ",\"critic_cpu\":" << parity.critic_cpu << ",\"critic_cuda\":" << parity.critic_cuda << "}\n";
}

void run_session(const std::string& gait_path, const std::string& offset_path, const std::string& starts_path,
    const fs::path& output, int target, double wall_budget, bool resume, bool verify_resume,
    fast::FeedbackMode feedback, Practice practice = Practice::Original,
    const std::string& parent_path = {}, const std::string& parent_task = {}) {
    host_feedback_mode = feedback;
    host_practice = practice;
    host_sampling_update_origin = 0;
    if (target < 1 || target > max_updates || !std::isfinite(wall_budget) ||
        wall_budget <= 0 || wall_budget > max_session_wall_s)
        throw std::runtime_error("Session exceeds the128-update/1200-second authorization");
    if (!resume) {
        if (!fs::create_directory(output)) throw std::runtime_error("New session output must not already exist");
        fs::create_directory(output / "checkpoints");
    } else if (!fs::is_directory(output / "checkpoints"))
        throw std::runtime_error("Resume requires the existing session directory");
    cp::Lease lease((output / "session-owner").u8string());
    const auto invocation_start = Clock::now();
    auto identity = input_identity(gait_path, offset_path, starts_path);
    std::vector<uint8_t> parent_checkpoint;
    if (!parent_path.empty()) {
        parent_checkpoint = cp::read(parent_path);
        bind_parent(identity, parent_checkpoint, parent_task);
    } else if (practice != Practice::Original) {
        throw std::runtime_error("Mixed practice requires an explicit preserved full-state parent");
    }
    const auto fitted = tail::read_navigation_weights(gait_path, tail::NavigationMode::Gait);
    const auto preserved = tail::steering::read_model(offset_path);
    const auto& original = preserved.weights;
    const auto initial = tail::read_navigation_worlds(starts_path);
    fast::validate_frozen_gait(fitted);
    fast::validate_actor(original);
    const auto initial_model = initial_policy(offset_path);
    const auto& initial_actor = initial_model.weights;
    if (initial.size() != 256) throw std::runtime_error("Expected256 archived training-only starts");
    size_t stack = 0;
    CUDA_CHECK_THROW(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
    CUDA_CHECK_THROW(cudaDeviceSetLimit(cudaLimitStackSize, (std::max)(stack, size_t{16384})));
    Stream stream;
    Rollout roll;
    Batch batch;
    EventPair optimization_timer;
    device_ptr<tail::NavigationWorld> starts(initial.size());
    device_ptr<float> gait(fitted.size());
    starts.copy_from_host(initial.data(), initial.size());
    gait.copy_from_host(fitted.data(), fitted.size());
    roll.frozen_actor_prefix.copy_from_host(identity.initial_coefficients.data(), identity.initial_coefficients.size());
    roll.sigma.copy_from_host(&log_std, 1);
    nn::GpuMlp actor(tail::steering::spec(), batch_size, seed);
    nn::MlpSpec critic_spec{{critic_channels, 32, 1}, {nn::Activation::Tanh, nn::Activation::Identity}};
    nn::GpuMlp critic(critic_spec, batch_size, seed + 1);
    critic.set_params(nn::orthogonal_params(critic_spec, seed + 1, {1.41421356f, 1.f}));
    nn::AdamConfig actor_cfg, critic_cfg;
    actor_cfg.lr = actor_lr;
    critic_cfg.lr = critic_lr;
    actor_cfg.max_grad_norm = critic_cfg.max_grad_norm = 1.f;
    nn::GpuAdam actor_adam(actor.num_params(), actor_cfg), critic_adam(critic.num_params(), critic_cfg);
    Ledger ledger;
    constexpr int no_fault = INT_MAX;
    CUDA_CHECK_THROW(cudaMemcpyAsync(roll.first_fault.get(), &no_fault, sizeof(no_fault), cudaMemcpyHostToDevice, stream.value));
    zero(roll.fault_codes.get(), roll.fault_codes.size_bytes(), stream.value);
    zero(roll.nonfinite.get(), roll.nonfinite.size_bytes(), stream.value);
    if (resume) {
        const auto checkpoint = cp::read((output / "latest.checkpoint").u8string());
        ledger.previous_wall_s = restore_budget(output, checkpoint, target, wall_budget);
        restore_session(checkpoint, roll, actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
    } else if (!parent_checkpoint.empty()) {
        restore_session(parent_checkpoint, roll, actor, critic, actor_adam, critic_adam,
            ledger, identity, stream.value, true);
        const auto imported_counts = ledger.counts;
        // Only the new study's diagnostic counters restart. Native worlds,
        // their episode RNG positions and global Adam/sampling ages continue.
        ledger.completed = ledger.started = 0;
        ledger.counts = {};
        ledger.transfers_ms = 0;
        zero(roll.counts.get(), roll.counts.size_bytes(), stream.value);
        std::ofstream receipt(output / "fork-equivalence.json");
        receipt.exceptions(std::ios::failbit | std::ios::badbit);
        receipt << "{\"pass\":true,\"physics_steps\":0,\"optimizer_updates\":0"
            << ",\"native_worlds_bit_exact\":true,\"actor_bit_exact\":true,\"critic_bit_exact\":true"
            << ",\"both_optimizers_bit_exact\":true,\"worlds\":" << worlds
            << ",\"parent_payload_checksum\":" << identity.parent_payload
            << ",\"parent_update\":" << identity.parent_completed
            << ",\"parent_actor_steps\":" << identity.parent_actor_steps
            << ",\"parent_critic_steps\":" << identity.parent_critic_steps
            << ",\"imported_actions\":" << imported_counts.actions
            << ",\"imported_attempted_substeps\":" << imported_counts.attempted
            << ",\"imported_accepted_substeps\":" << imported_counts.accepted
            << ",\"study_counters_rebased\":true,\"sampling_update_origin\":"
            << host_sampling_update_origin << "}\n";
        receipt.close();
        publish_session(output, roll, actor, critic, actor_adam, critic_adam, ledger, identity,
            stream.value, target, wall_budget, elapsed(invocation_start));
    } else {
        actor.set_params(initial_actor);
        zero(roll.lanes.get(), roll.lanes.size_bytes(), stream.value);
        zero(roll.counts.get(), roll.counts.size_bytes(), stream.value);
        initialize_lanes<<<worlds / lane_threads, lane_threads, 0, stream.value>>>(roll.lanes.get());
        check_launch();
        publish_session(output, roll, actor, critic, actor_adam, critic_adam, ledger, identity,
            stream.value, target, wall_budget, elapsed(invocation_start));
    }

    const auto progress_path = output / "updates.csv";
    reconcile_progress(progress_path, ledger.completed);
    const bool had_progress = fs::exists(progress_path);
    std::ofstream progress(progress_path, std::ios::app);
    progress.exceptions(std::ios::failbit | std::ios::badbit);
    progress << std::setprecision(10);
    if (!had_progress) progress << "update,logical_samples,reward_mean,exact_kl,policy_loss,value_loss,entropy,clip_fraction,"
        "explained_variance,minibatches,epochs,attempted_substeps,accepted_substeps,episode_starts,episode_horizons,"
        "left_starts,right_starts,clearance_stops,posture_stops,physics_gpu_ms,network_gpu_ms,optimization_gpu_ms,"
        "sync_readback_ms,evidence_cumulative_ms,total_session_wall_s,w_course,w_heading,w_previous_offset,w_yaw_rate,"
        "w_full_error90,w_root_phase_position,w_root_phase_velocity,w_body_lateral_speed,w_error_speed,"
        "w_error_phase_cos2,w_error_phase_sin2,w_error2_phase_position,w_error2_phase_velocity\n";
    cudaDeviceProp device{};
    int device_index = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&device_index));
    CUDA_CHECK_THROW(cudaGetDeviceProperties(&device, device_index));
    std::cout << "device=" << device.name << " worlds=" << worlds << " steps=" << decisions
        << " actor_inputs=" << channels << " trainable_actor_parameters="
        << tail::steering::trainable_count(fast::policy_kind(host_feedback_mode))
        << " critic_inputs=" << critic_channels
        << " feedback=" << feedback_name(host_feedback_mode)
        << " resumed_update=" << ledger.completed << " target=" << target << std::endl;
    std::vector<uint8_t> checkpoint_one, uninterrupted_two;
    bool replaying = false;
    bool verification_completed = false;
    Counts collection_before{};
    bool collection_accounted = true;
    FeedbackParity initial_parity;
    if (verify_resume) {
        initial_parity = parent_checkpoint.empty() ? initial_feedback_parity(starts, stream.value)
            : feedback_parity(roll.lanes, stream.value);
        write_feedback_parity(output / "feedback-parity-initial.json", initial_parity);
        if (!initial_parity.pass) throw std::runtime_error("Initial CPU/CUDA feedback parity exceeds 1e-6");
    }
    auto session_wall = [&] { return ledger.previous_wall_s + elapsed(invocation_start); };
    try {
        while (ledger.completed < target) {
            if (session_wall() >= wall_budget || fs::exists(output / "STOP")) {
                ledger.stage = "checkpointed-stop";
                break;
            }
            const int update = ledger.completed + 1;
            const double physics_before = ledger.physics_ms, network_before = ledger.network_ms;
            const double transfers_before = ledger.transfers_ms;
            collection_before = ledger.counts;
            collection_accounted = false;
            const float reward = collect(roll, actor, critic, starts, gait, ledger, update, stream.value);
            record_executed_work(ledger, collection_before);
            collection_accounted = true;
            optimization_timer.start(stream.value);
            const float kl = optimize(roll, batch, actor, critic, actor_adam, critic_adam, ledger, update, stream.value);
            optimization_timer.stop(stream.value);
            const double optimization_ms = optimization_timer.milliseconds();
            ledger.optimization_ms += optimization_ms;
            ledger.completed = update;
            ledger.stage = "checkpoint";
            if (verify_resume) {
                const auto payload = session_payload(roll, actor, critic, actor_adam, critic_adam,
                    ledger, identity, stream.value);
                if (update == 1) checkpoint_one = payload;
                else if (!replaying) {
                    uninterrupted_two = payload;
                    immutable_checkpoint(output / "continuous-2.bin", payload);
                    immutable_checkpoint(output / "checkpoint-1.bin", checkpoint_one);
                    restore_session(cp::read((output / "checkpoint-1.bin").u8string()), roll,
                        actor, critic, actor_adam, critic_adam, ledger, identity, stream.value);
                    replaying = true;
                    continue;
                } else {
                    immutable_checkpoint(output / "resumed-2.bin", payload);
                    const bool pass = payload == uninterrupted_two;
                    const auto final_parity = feedback_parity(roll.lanes, stream.value);
                    write_feedback_parity(output / "feedback-parity-restored.json", final_parity);
                    std::ofstream receipt(output / "verification.json");
                    receipt << std::setprecision(12) << "{\"pass\":" << (pass && final_parity.pass ? "true" : "false")
                        << ",\"feedback\":\"" << feedback_name(host_feedback_mode) << "\""
                        << ",\"worlds\":" << worlds << ",\"logical_updates\":2,\"physical_updates_executed\":3,"
                        << "\"sampled_actions_executed\":" << uint64_t(3) * rows
                        << ",\"actual_attempted_substeps\":" << ledger.executed_counts.attempted
                        << ",\"actual_accepted_substeps\":" << ledger.executed_counts.accepted
                        << ",\"actual_actor_steps\":" << ledger.executed_actor_steps
                        << ",\"actual_critic_steps\":" << ledger.executed_critic_steps
                        << ",\"feedback_parity_tolerance\":0.000001,\"parity_new_physics_steps\":0"
                        << ",\"initial_actor_max_difference\":" << initial_parity.actor_max_difference
                        << ",\"initial_critic_max_difference\":" << initial_parity.critic_max_difference
                        << ",\"initial_nonfinite_values\":" << initial_parity.nonfinite_values
                        << ",\"restored_actor_max_difference\":" << final_parity.actor_max_difference
                        << ",\"restored_critic_max_difference\":" << final_parity.critic_max_difference
                        << ",\"restored_nonfinite_values\":" << final_parity.nonfinite_values
                        << ",\"compared\":[\"actor\",\"critic\",\"actor_adam\",\"critic_adam\","
                        << "\"native_worlds\",\"servo_and_trim_state\",\"course_history\",\"episode_rng_positions\",\"counters\"],"
                        << "\"uninterrupted_checksum\":" << cp::checksum(uninterrupted_two)
                        << ",\"resumed_checksum\":" << cp::checksum(payload)
                        << ",\"wall_s\":" << session_wall() << "}\n";
                    receipt.close();
                    if (!pass) throw std::runtime_error("Resume changes deterministic full training state");
                    if (!final_parity.pass) throw std::runtime_error("Restored CPU/CUDA feedback parity exceeds 1e-6");
                    verification_completed = true;
                }
            }
            publish_session(output, roll, actor, critic, actor_adam, critic_adam, ledger,
                identity, stream.value, target, wall_budget, session_wall());
            const auto weights = actor.get_params();
            const auto& count = ledger.counts;
            progress << update << ',' << uint64_t(update) * rows << ',' << reward << ',' << kl
                << ',' << ledger.policy_loss << ',' << ledger.value_loss << ',' << ledger.entropy
                << ',' << ledger.clip_fraction << ',' << ledger.explained_variance << ',' << ledger.minibatches
                << ',' << ledger.epochs_completed << ',' << count.attempted << ',' << count.accepted
                << ',' << count.starts << ',' << count.horizons << ',' << count.left << ',' << count.right
                << ',' << count.clearance_stops << ',' << count.posture_stops
                << ',' << ledger.physics_ms - physics_before << ',' << ledger.network_ms - network_before
                << ',' << optimization_ms << ',' << ledger.transfers_ms - transfers_before
                << ',' << ledger.evidence_ms << ',' << session_wall();
            for (int i = 0; i < tail::steering::phase_feature_end; ++i) progress << ',' << weights[i];
            progress << '\n' << std::flush;
            write_result(output, ledger, target, session_wall(), identity, verify_resume);
            std::cout << "update=" << update << " samples=" << uint64_t(update) * rows
                << " reward=" << reward << " kl=" << kl << " course_weight=" << weights[0]
                << " wall_s=" << session_wall() << std::endl;
        }
        if (verify_resume && !verification_completed) throw std::runtime_error("Resume verification did not reach its complete comparison");
        if (ledger.completed == target) ledger.stage = "complete";
        else publish_session(output, roll, actor, critic, actor_adam, critic_adam, ledger,
            identity, stream.value, target, wall_budget, session_wall());
        // Confirm device-held fitted gait and preserved input files remain exact.
        std::vector<float> device_gait(fitted.size());
        gait.copy_to_host(device_gait.data(), device_gait.size());
        auto final_identity = input_identity(gait_path, offset_path, starts_path);
        if (!parent_path.empty()) bind_parent(final_identity, cp::read(parent_path), parent_task);
        if (device_gait != fitted || final_identity.gait != identity.gait ||
            final_identity.offset != identity.offset || final_identity.starts != identity.starts ||
            final_identity.initial_actor_weights != identity.initial_actor_weights ||
            final_identity.parent_payload != identity.parent_payload)
            throw std::runtime_error("Protected motor, offset or starting evidence changed");
        write_result(output, ledger, target, session_wall(), identity, verify_resume);
    } catch (...) {
        ledger.stage = "fault";
        try {
            count_work(roll, ledger, stream.value);
            if (!collection_accounted) record_executed_work(ledger, collection_before);
            // Preserve rejected physics/optimizer state without advancing the
            // resumable latest.checkpoint or exporting a candidate actor.
            const auto rejected = session_payload(roll, actor, critic, actor_adam, critic_adam,
                ledger, identity, stream.value);
            const auto path = output / ("rejected-update-" + std::to_string(ledger.started) + ".bin");
            immutable_checkpoint(path, rejected);
            write_result(output, ledger, target, session_wall(), identity, verify_resume);
        } catch (const std::exception& evidence_error) {
            std::cerr << "Fault evidence warning: " << evidence_error.what() << '\n';
        }
        throw;
    }
}
