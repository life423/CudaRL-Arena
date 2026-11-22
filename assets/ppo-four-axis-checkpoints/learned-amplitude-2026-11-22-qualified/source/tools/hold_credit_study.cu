// One opt-in, bounded hold-credit lesson. No runtime controller installation.
// Reuses shared physics, likelihood, GAE, GPU MLP/Adam and frozen assessments.
#include "motor/joint_motor.h"
namespace credit {
void physical_failure(const cudarl::joint_motor::World&, const cudarl::physics::BodyState&,
                      const cudarl::physics::BodyState&, int, int);
}
#define CUDARL_PHYSICAL_FAILURE_HOOK credit::physical_failure
#define CUDARL_REST_HOLD_MEASURED_PHASE_B 1
#define CUDARL_REST_HOLD_MEASURED_BASELINE 1
#define CUDARL_REST_HOLD_RESET_STUDY 1
#include "rest_hold_study.cu"
#include <set>

namespace credit {
static_assert(observations == 41 && actions == 4, "Compile with CUDARL_APPEND_HOLD41");
constexpr int batch_rows = 10240, cells = 800, calibration_controls = 20960;
constexpr int total_controls = 188640, max_actor_steps = 640;
constexpr uint32_t protocol_magic = 0x31434850, baseline_magic = 0x31434842;
constexpr uint32_t actor_magic = 0x31434841, ready_magic = 0x31434852;
fs::path failure_directory;
bool cpu_failure_saved = false;
void physical_failure(const m::World& world, const p::BodyState& before,
                      const p::BodyState& candidate, int controls, int substeps) {
    if (cpu_failure_saved) return;
    require(!failure_directory.empty() && fs::is_directory(failure_directory), "CPU failure evidence directory absent");
    const auto file = failure_directory/"first-cpu-physical-failure.bin";
    require(!fs::exists(file), "First CPU failure evidence already exists"); cpu_failure_saved = true;
    cp::Writer out; out.value(uint32_t{0x31434648}); out.value(uint32_t{1});
    out.value(controls); out.value(substeps); out.value(uint32_t{sizeof(m::World)}); out.value(world);
    out.value(before); out.value(candidate); cp::write_atomic(file.string(),out.data);
}
CUDARL_HD constexpr int length(int slot) { return slot % 80 < 20 ? 20 : 60; }
CUDARL_HD constexpr int age(int slot) { return slot % 80 < 20 ? slot % 80 : slot % 80 - 20; }
bool fitting(int row) { return row % 32 % 8 < 4; }
int mode(int row) { return (row % 32 / 2) % 2; }
int cell(int row) {
    const int t = row / 32, pose = row % 2;
    return mode(row) ? 640 + pose * 80 + (length(t) == 20 ? 0 : 20) + age(t)
                     : pose * 320 + t;
}
struct Baseline {
    std::vector<double> mean = std::vector<double>(cells);
    std::vector<int> n = std::vector<int>(cells);
    double rms[2]{};
};
struct Signal { double energy[2]{}, bias[2]{}, rmse[2]{}, share = 0;
    double duration_bias[2]{}, duration_mse[2]{}, duration_offset_fraction[2]{}; int duration_n[2]{}; };
Baseline fit_baseline(const std::vector<float>& raw) {
    require(raw.size() == batch_rows, "Bad baseline input shape");
    Baseline b;
    for (int r = 0; r < batch_rows; ++r) {
        require(std::isfinite(raw[r]), "Nonfinite raw return");
        if (fitting(r)) { b.mean[cell(r)] += raw[r]; ++b.n[cell(r)]; }
    }
    for (int j = 0; j < cells; ++j) {
        require(b.n[j] == (j < 640 ? 4 : 16), "Duration-conditioned fit occupancy changed");
        b.mean[j] /= b.n[j];
    }
    int count[2]{};
    for (int r = 0; r < batch_rows; ++r) if (fitting(r)) {
        const double e = raw[r] - b.mean[cell(r)];
        b.rms[mode(r)] += e * e; ++count[mode(r)];
    }
    for (int m = 0; m < 2; ++m) {
        require(count[m] == 2560, "Fit-world task balance changed");
        b.rms[m] = std::sqrt(b.rms[m] / count[m]);
        require(std::isfinite(b.rms[m]) && b.rms[m] > 0, "Invalid fixed residual scale");
    }
    return b;
}
Signal signal(const std::vector<float>& raw, const Baseline& b) {
    Signal s; int count[2]{};
    for (int r = 0; r < batch_rows; ++r) if (!fitting(r)) {
        const int m = mode(r); const double e = raw[r] - b.mean[cell(r)];
        s.bias[m] += e; s.rmse[m] += e * e; s.energy[m] += e * e / (b.rms[m] * b.rms[m]); ++count[m];
        if (m) { const int k = length(r/32) == 20 ? 0 : 1; s.duration_bias[k] += e;
            s.duration_mse[k] += e*e; ++s.duration_n[k]; }
    }
    for (int m = 0; m < 2; ++m) {
        require(count[m] == 2560 && std::isfinite(s.energy[m]), "Invalid held-out task signal");
        s.bias[m] /= count[m]; s.rmse[m] = std::sqrt(s.rmse[m] / count[m]);
    }
    require(s.energy[0] + s.energy[1] > 0, "No finite held-out signal");
    for (int k = 0; k < 2; ++k) {
        require(s.duration_n[k] == (k ? 1920 : 640), "Held-out duration balance changed");
        s.duration_bias[k] /= s.duration_n[k]; s.duration_mse[k] /= s.duration_n[k];
        require(std::isfinite(s.duration_mse[k]) && s.duration_mse[k] > 0, "No finite within-duration residual energy");
        s.duration_offset_fraction[k] = s.duration_bias[k]*s.duration_bias[k]/s.duration_mse[k];
    }
    s.share = s.energy[1] / (s.energy[0] + s.energy[1]); return s;
}
uint64_t text_hash(const fs::path& file) {
    std::ifstream f(file, std::ios::binary); require(bool(f), "Bound file missing");
    std::vector<uint8_t> bytes; for (char c; f.get(c);) bytes.push_back(static_cast<uint8_t>(c));
    require(f.eof(), "Bound file could not be read"); return cp::checksum(bytes);
}
std::vector<uint8_t> protocol_bytes(const fs::path& root) {
    cp::Writer w; w.value(protocol_magic); w.value(uint32_t{1}); w.array(rest_protocol());
    w.value(text_hash(root / "approved-protocol.md"));
    for (int n : {20,60,80,320,32,41,4,cells,calibration_controls,total_controls,total_controls*10,8,max_actor_steps,120}) w.value(n);
    const std::string method = "one B20_60 arm; fit-only task/pose/duration/age baseline; fixed task residual RMS; no critic; assess each update; stop at swimming regression; no adoption";
    w.array(std::vector<uint8_t>(method.begin(), method.end())); return w.data;
}
void configure(const fs::path& root, bool copied) {
    configure_rest(root, copied); control_cap = total_controls; substep_cap = total_controls * 10;
    training_cap = 81920; updates = 8; active_wall_cap = 120;
    require(horizon == 320 && rows == batch_rows && seed == 1247037243u && correlation == .8f,
            "Credit study inherited configuration mismatch");
}
void validate_protocol(const fs::path& root) {
    require(cp::read((root / "protocol.bin").string()) == protocol_bytes(root), "Frozen credit protocol changed");
}
std::vector<uint8_t> baseline_bytes(const Baseline& b, uint64_t id) {
    cp::Writer w; w.value(baseline_magic); w.value(uint32_t{1}); w.value(id); w.array(b.mean); w.array(b.n);
    for (double x : b.rms) w.value(x); return w.data;
}
Baseline read_baseline(const fs::path& root, uint64_t id) {
    const auto bytes = cp::read((root / "fixed-baseline.bin").string()); cp::Reader r{bytes};
    require(r.value<uint32_t>() == baseline_magic && r.value<uint32_t>() == 1 && r.value<uint64_t>() == id,
            "Wrong baseline identity"); Baseline b; b.mean = r.array<double>(cells); b.n = r.array<int>(cells);
    require(b.mean.size() == cells && b.n.size() == cells, "Bad baseline shape");
    for (int j = 0; j < cells; ++j) require(std::isfinite(b.mean[j]) && b.n[j] == (j < 640 ? 4 : 16), "Bad baseline cell");
    for (double& x : b.rms) { x = r.value<double>(); require(std::isfinite(x) && x > 0, "Bad fixed scale"); }
    r.finish(); return b;
}

// Research scheduling only. The shared physical advance/reward is unchanged.
// Its historical 20-control done flag is replaced by this study's exact terminal.
__global__ void boundary(m::World* worlds, m::ForwardTask* tasks, float* previous_obs,
                         float* previous_action, float* history, float* speed, int* fault,
                         int slot, m::TrainingEpisode episode) {
    const int i = threadIdx.x; if (i >= 32 || *fault) return;
    const bool hold = episode.hold_episode(i), reset = slot == 0 || (hold && age(slot) == 0);
    history[i] = reset ? 0.f : 1.f;
    if (reset) {
        if (slot != 0) {
            const int prior = length(slot - 1); const auto config = episode.task_config(0, i);
            if (tasks[i].controls != prior || worlds[i].physical.control_steps != uint64_t(prior) ||
                worlds[i].physical.substeps != uint64_t(prior*10) ||
                !m::begin_training_episode(i, episode.tilt_mixture, config, worlds[i], tasks[i],
                                           episode.arena(), episode.initial_vertical_offset(i)) ||
                !m::update_velocity_request(worlds[i], config, tasks[i])) { atomicExch(fault, 8); return; }
        }
        for (int j = 0; j < 41; ++j) previous_obs[i*41+j] = 0;
        for (int j = 0; j < 4; ++j) previous_action[i*4+j] = 0;
    }
    speed[i] = p::norm(m::measure(tasks[i], worlds[i]).velocity_m_s);
}
__global__ void terminal(const m::ForwardTask* tasks, float* done, int* fault, int slot,
                         m::TrainingEpisode episode) {
    const int i = threadIdx.x; if (i >= 32 || *fault) return;
    const bool hold = episode.hold_episode(i); const int local = hold ? age(slot) : slot;
    if (tasks[i].controls != local + 1) { atomicExch(fault, 9); return; }
    done[i] = float(local + 1 == (hold ? length(slot) : 320));
}
void collect(int update, Arm& a) {
    auto& d = *a.data; const auto s = a.stream->value;
    training_episode.offset_phase = update % 3;
    gpu::initialize<<<1,32,0,s>>>(d.worlds.get(),d.tasks.get(),32,d.fault.get(),training_episode);
    for (int t = 0; t < 320; ++t) {
        wall_guard(); const int offset = t * 32;
        auto obs = d.obs.get()+offset*41, prev = d.previous_obs.get()+offset*41;
        auto oldaction = d.previous_action.get()+offset*4, hist = d.history.get()+offset;
        if (t == 0) {
            CUDA_CHECK_THROW(cudaMemsetAsync(prev,0,32*41*sizeof(float),s));
            CUDA_CHECK_THROW(cudaMemsetAsync(oldaction,0,32*4*sizeof(float),s));
        } else {
            CUDA_CHECK_THROW(cudaMemcpyAsync(prev,d.obs.get()+(offset-32)*41,32*41*sizeof(float),cudaMemcpyDeviceToDevice,s));
            CUDA_CHECK_THROW(cudaMemcpyAsync(oldaction,d.action.get()+(offset-32)*4,32*4*sizeof(float),cudaMemcpyDeviceToDevice,s));
        }
        boundary<<<1,32,0,s>>>(d.worlds.get(),d.tasks.get(),prev,oldaction,hist,d.pre_action_speed.get()+offset,d.fault.get(),t,training_episode);
        gpu::observe<<<1,32,0,s>>>(d.worlds.get(),d.tasks.get(),obs,32,d.fault.get(),training_episode);
        CUDA_CHECK_THROW(cudaMemcpyAsync(d.actor_input.get(),obs,32*41*sizeof(float),cudaMemcpyDeviceToDevice,s));
        CUDA_CHECK_THROW(cudaMemcpyAsync(d.actor_input.get()+32*41,prev,32*41*sizeof(float),cudaMemcpyDeviceToDevice,s));
        a.actor->forward(d.actor_input.get(),64,s);
        rl::sample_ar_gaussian(a.actor->output(),d.sigma.get(),oldaction,hist,32,4,correlation,
                              rest_action_seed(),rest_action_counter(update,t),d.action.get()+offset*4,d.logp.get()+offset,s);
        CUDA_CHECK_THROW(cudaMemsetAsync(d.value.get()+offset,0,32*sizeof(float),s));
        gpu::advance<<<1,32,0,s>>>(d.worlds.get(),d.tasks.get(),d.action.get()+offset*4,d.reward.get()+offset,
            d.done.get()+offset,d.counts.get(),32,d.fault.get(),a.actor->output(),training_episode,d.cost_trace.get()+offset*5);
        terminal<<<1,32,0,s>>>(d.tasks.get(),d.done.get()+offset,d.fault.get(),t,training_episode);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
}
void account(Arm& a, Budget& b) {
    const auto counts = download(a.data->counts.get(),32,a.stream->value);
    unsigned long long c = 0, s = 0; for (const auto& x : counts) { c += x.controls; s += x.substeps; }
    require(c >= a.training_controls && s >= a.training_substeps, "Persistent attempt counters moved backwards");
    b.controls += int(c-a.training_controls); b.substeps += int(s-a.training_substeps);
    a.training_controls = c; a.training_substeps = s;
    require(b.controls <= b.maximum_controls && b.substeps <= b.maximum_substeps, "Credit physical cap exceeded");
}
void create_actor(Arm& a, const fs::path& root, bool optimizer) {
    a.root = root; a.stream = std::make_unique<Stream>();
    a.actor = std::make_unique<nn::GpuMlp>(spec(4),rows,seed); a.actor->set_params(resume_state->actor.parameters);
    require(a.actor->get_params() == resume_state->actor.parameters, "Parent actor binding failed");
    a.data = std::make_unique<Rollout>(a.stream->value); a.batch = std::make_unique<Batches>();
    if (optimizer) {
        a.actor_opt = std::make_unique<nn::GpuAdam>(2532,initial_optimizer_config(true));
        restore_optimizer(*a.actor_opt,resume_state->actor_adam); a.actor_opt->set_learning_rate(*actor_lr_override);
        a.actor_opt->set_fresh_column(41,32,0);
        require(a.actor_opt->steps() == 21700 && a.actor_opt->fresh_steps() == 0, "Parent Adam age changed");
    }
}
struct Batch { std::vector<float> raw; int quiet = 0, heldquiet = 0;
    int neutral_rows[2]{}, neutral_longest[2]{}; double gae_error = 0, logp_error = 0; };
Batch batch(int update, Arm& a, Budget& b) {
    require(b.controls+batch_rows <= b.maximum_controls && b.substeps+batch_rows*10 <= b.maximum_substeps,
            "Next batch would exceed physical budget");
    const int before = b.controls;
    try { collect(update,a); } catch (...) { account(a,b); throw; }
    account(a,b); auto& d = *a.data; const auto s = a.stream->value;
    if (download(d.fault.get(),1,s)[0] != 0) {
        cp::Writer failure; failure.value(uint32_t{1}); failure.value(uint32_t{sizeof(gpu::Counts)});
        failure.array(download(d.counts.get(),32,s)); failure.value(uint32_t{sizeof(m::World)});
        failure.array(download(d.worlds.get(),32,s));
        cp::write_atomic((a.root/"first-failed-state.bin").string(),failure.data);
        throw std::runtime_error("CUDA physical/data fault before GAE or optimizer");
    }
    require(b.controls-before == batch_rows, "Batch actual control count changed"); compute_rest_returns(d,s);
    Batch out; out.raw = download(d.returns.get(),rows,s);
    const auto obs = download(d.obs.get(),rows*41,s), prev = download(d.previous_obs.get(),rows*41,s);
    const auto oldaction = download(d.previous_action.get(),rows*4,s), value = download(d.value.get(),rows,s);
    const auto reward = download(d.reward.get(),rows,s), done = download(d.done.get(),rows,s);
    const auto history = download(d.history.get(),rows,s), speed = download(d.pre_action_speed.get(),rows,s);
    const auto cost = download(d.cost_trace.get(),rows*5,s), adv = download(d.advantage.get(),rows,s);
    const auto counts = download(d.counts.get(),32,s);
    const auto tasks = download(d.tasks.get(),32,s);
    const auto worlds = download(d.worlds.get(),32,s);
    const auto latent = download(d.action.get(),rows*4,s);
    require(out.raw == adv, "Critic-free raw return/advantage mismatch"); int resets = 0;
    for (int e = 0; e < 32; ++e) {
        const bool hold = mode(e) != 0; const int final = hold ? 60 : 320;
        require(tasks[e].controls == final && worlds[e].physical.control_steps == uint64_t(final) &&
                worlds[e].physical.substeps == uint64_t(final*10), "Actual mixed-duration end state differs");
        require(counts[e].controls == a.training_controls/32 && counts[e].substeps == a.training_substeps/32,
                "Per-world actual accounting mismatch");
        require(tasks[e].hold_anchor_set == hold && p::norm(worlds[e].physical.current) == 0 &&
                p::norm(tasks[e].request.terminal_velocity-tasks[e].request.heading*(hold?0.f:.008f)) < 1e-8f,
                "Actual request/anchor/current differs"); double target = 0;
        for (int t = 319; t >= 0; --t) {
            const int r = t*32+e, local = hold ? age(t) : t, len = hold ? length(t) : 320;
            const bool end = local+1 == len;
            require(value[r] == 0 && done[r] == float(end) && history[r] == float(local != 0), "Value/terminal/AR mask mismatch");
            if (end) target = 0; target = reward[r] + double(.99f)*target;
            const double error = std::abs(double(out.raw[r])-target); out.gae_error = std::max(out.gae_error,error);
            require(error < 2e-5+2e-5*std::abs(target), "Independent segmented return mismatch");
            float sum = 0; for (int j = 0; j < 5; ++j) sum += cost[r*5+j];
            require(std::abs(reward[r]+p::dt*sum) < 1e-6+1e-5*std::abs(reward[r]), "Reward components changed");
            const float* x = obs.data()+r*41;
            require(std::isfinite(speed[r]) && std::isfinite(out.raw[r]) && x[34] == p::clamp(1.f-float(local)/40,0,1), "Observation local clock changed");
            if (hold) {
                for (int j = 3; j < 6; ++j) require(x[j] == 0, "Hold goal missing");
                if (speed[r] <= .0005f) { ++out.quiet; if (!fitting(r)) ++out.heldquiet; }
            }
            if (local == 0) {
                resets += hold; require(x[34] == 1 && x[40] == 0 && speed[r] == 0, "Reset not at original rest state");
                for (int j = 12; j < 18; ++j) require(x[j] == 0, "Reset velocity nonzero");
                for (int j = 0; j < 41; ++j) require(prev[r*41+j] == 0, "Previous observation crossed terminal");
                for (int j = 0; j < 4; ++j) require(oldaction[r*4+j] == 0, "Previous action crossed terminal");
            }
        }
    }
    require(resets == 128, "Mixed-duration hold reset count changed");
    for (int e = 0; e < 32; ++e) if (mode(e)) {
        int streak[2]{};
        for (int t = 0; t < 320; ++t) for (int k = 0; k < 2; ++k) {
            if (age(t) == 0) streak[k] = 0;
            bool neutral = true; for (int j = 0; j < 4; ++j)
                neutral = neutral && std::abs(std::tanh(latent[(t*32+e)*4+j])) <= (k ? .1f : .03f);
            if (neutral) { ++out.neutral_rows[k]; ++streak[k]; out.neutral_longest[k] = std::max(out.neutral_longest[k],streak[k]); }
            else streak[k] = 0;
        }
    }
    const auto rescored = score_batch(*a.actor,d,*a.batch,s), oldlogp = download(d.logp.get(),rows,s);
    for (int r = 0; r < rows; ++r) out.logp_error = std::max(out.logp_error,std::abs(double(rescored[r])-oldlogp[r]));
    require(out.logp_error < 1e-4, "Stored latent action probability rescore failed"); cache_batch(a,update);
    std::ofstream csv(a.root/("rows-"+std::to_string(update)+".csv"));
    csv << std::setprecision(12) << "slot,world,task,pose,fit,cell,local_age,segment_length,speed_m_s,done,history,reward,return,logp\n";
    for (int r = 0; r < rows; ++r) {
        const int t = r/32, h = mode(r);
        csv << t << ',' << r%32 << ',' << h << ',' << (r%2?3:0) << ',' << fitting(r) << ',' << cell(r) << ','
            << (h?age(t):t) << ',' << (h?length(t):320) << ',' << speed[r] << ',' << done[r] << ',' << history[r]
            << ',' << reward[r] << ',' << out.raw[r] << ',' << oldlogp[r] << '\n';
    }
    require(bool(csv), "Could not write saved diagnostic rows"); return out;
}

using Keys = std::set<std::pair<int,int>>;
Keys window_keys(const fs::path& path) {
    std::ifstream f(path); require(bool(f), "Missing long-window assessment"); std::string line; std::getline(f,line);
    Keys all, passed; int count = 0;
    while (std::getline(f,line)) {
        std::stringstream row(line); std::string item; std::vector<double> v;
        while (std::getline(row,item,',')) v.push_back(std::stod(item));
        require(v.size() == 10, "Wrong long-window columns"); const int pose = int(v[0]), end = int(v[1]);
        require((pose == 0 || pose == 3) && end >= 4 && end <= 60 && end%4 == 0 && all.emplace(pose,end).second,
                "Wrong/duplicate long-window key"); ++count; if (v[9] == 1) passed.emplace(pose,end);
    }
    require(count == 30, "Wrong long-window count"); return passed;
}
struct Assessment { int short_pass = 0, long_pass = 0, hold_pass = 0; Keys keys; };
Assessment assess(const std::vector<float>& actor, const fs::path& dir, Budget& b) {
    require(fs::create_directory(dir), "Assessment identity exists"); const int before = b.controls;
    const auto swim = evaluate(actor,dir,"frozen",b);
    const auto hold = evaluate_rest(actor,dir/"rest",b);
    require(fs::create_directory(dir/"long"), "Long assessment identity exists");
    const auto longrun = sustained(actor,dir/"long",b);
    require(longrun.safety_fault.empty() && hold.completed == 6 && b.controls-before == 10720, "Frozen assessment failed or count differs");
    const auto keys = window_keys(dir/"long"/"sustained-windows.csv");
    require(keys.size() == size_t(longrun.windows_passed), "Long assessment count/key disagreement");
    std::ifstream f(dir/"rest"/"controls.csv"); std::string line; std::getline(f,line);
    std::ofstream breaches(dir/"first-hold-breach.csv"); breaches << "case,time_s,cause\n";
    std::array<bool,6> seen{}; std::array<double,6> at56{};
    while (std::getline(f,line)) {
        std::stringstream row(line); std::string item; std::vector<double> v;
        while (std::getline(row,item,',')) v.push_back(std::stod(item));
        require(v.size() == 30, "Wrong hold trace columns"); const int id = int(v[0]), k = int(v[3]);
        require(id >= 0 && id < 6, "Wrong hold identity"); if (k == 1120) at56[id] = v[7]; if (seen[id]) continue;
        std::string cause; auto add = [&](const char* x) { if (!cause.empty()) cause += '|'; cause += x; };
        if (v[5] > .5) add("speed"); if (v[13] > 2) add("anchor"); if (std::abs(v[8]) > 1) add("depth");
        if (v[9] > 2) add("lane"); if (v[10] > 10) add("tilt"); if (k > 1120 && std::abs(v[7]-at56[id]) > 2) add("final8_excursion");
        if (!cause.empty()) { breaches << id << ',' << v[4] << ',' << cause << '\n'; seen[id] = true; }
    }
    for (int i = 0; i < 6; ++i) if (!seen[i]) breaches << i << ",,none\n";
    return {swim.passed,longrun.windows_passed,hold.passed,keys};
}
void save_actor(const fs::path& root, const Arm& a, uint64_t id) {
    const auto file = root/("credit-actor-update-"+std::to_string(a.completed)+".bin"); require(!fs::exists(file), "Actor identity exists");
    cp::Writer out, adam; a.actor_opt->save(adam); const auto params = a.actor->get_params(); const int fresh = a.actor_opt->fresh_steps();
    out.value(actor_magic); out.value(uint32_t{1}); out.value(id); out.value(a.completed); out.value(fresh);
    out.value(uint32_t{41}); out.value(uint32_t{4}); out.array(params); out.array(adam.data); cp::write_atomic(file.string(),out.data);
    const auto bytes = cp::read(file.string()); cp::Reader in{bytes};
    require(in.value<uint32_t>() == actor_magic && in.value<uint32_t>() == 1 && in.value<uint64_t>() == id &&
            in.value<int>() == a.completed && in.value<int>() == fresh && in.value<uint32_t>() == 41 && in.value<uint32_t>() == 4 &&
            in.array<float>(2532) == params && in.array<uint8_t>() == adam.data, "Actor/checkpoint round-trip failed"); in.finish();
    cp::Reader ar{adam.data}; const auto state = rl::read_append_adam(ar,2532); ar.finish();
    require(state.steps == 21700+fresh && fresh >= 0 && fresh <= max_actor_steps, "Adam clock round-trip failed");
}
double update_actor(int update, Arm& a, const Baseline& b, const std::vector<float>& raw) {
    auto& d = *a.data; auto& mb = *a.batch; auto& actor = *a.actor; auto& opt = *a.actor_opt; const auto s = a.stream->value;
    std::vector<float> advantage(rows);
    for (int r = 0; r < rows; ++r) { advantage[r] = float((raw[r]-b.mean[cell(r)])/b.rms[mode(r)]); require(std::isfinite(advantage[r]), "Nonfinite fixed residual"); }
    const float identity[2] = {0,1}; d.advantage.copy_from_host(advantage.data(),rows); d.adv_stats.copy_from_host(identity,2);
    require(download(d.advantage.get(),rows,s) == advantage && download(d.adv_stats.get(),2,s) == std::vector<float>({0,1}), "Actor signal transfer changed");
    const int before = opt.steps(); double kl = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        CUDA_CHECK_THROW(cudaMemsetAsync(mb.stats.get(),0,mb.stats.size_bytes(),s));
        const auto permutation = rl::make_permutation(rows,seed+uint32_t(update*epochs+epoch));
        for (int first = 0; first < rows; first += minibatch) {
            wall_guard();
            rl::gather_rows(permutation,first,minibatch,41,d.obs.get(),mb.obs.get(),s);
            rl::gather_rows(permutation,first,minibatch,41,d.previous_obs.get(),mb.obs.get()+minibatch*41,s);
            rl::gather_rows(permutation,first,minibatch,4,d.previous_action.get(),mb.previous_action.get(),s);
            rl::gather_rows(permutation,first,minibatch,1,d.history.get(),mb.history.get(),s);
            rl::gather_rows(permutation,first,minibatch,4,d.action.get(),mb.action.get(),s);
            rl::gather_rows(permutation,first,minibatch,1,d.logp.get(),mb.logp.get(),s);
            rl::gather_rows(permutation,first,minibatch,1,d.advantage.get(),mb.advantage.get(),s);
            actor.forward(mb.obs.get(),2*minibatch,s);
            rl::ar_gaussian_ppo_loss_grad(actor.output(),d.sigma.get(),mb.previous_action.get(),mb.history.get(),d.zeros.get(),
                minibatch,4,correlation,mb.action.get(),mb.logp.get(),mb.advantage.get(),d.zeros.get(),d.adv_stats.get(),{.2f,0,0},
                mb.mean_gradient.get(),mb.value_gradient.get(),mb.row_stats.get(),mb.stats.get(),s);
            if (first == 0) {
                const auto got = download(mb.advantage.get(),minibatch,s), critic = download(mb.value_gradient.get(),minibatch,s);
                for (int j = 0; j < minibatch; ++j) require(got[j] == advantage[rl::permute(permutation,j)] && critic[j] == 0, "Actual actor minibatch lost signal");
            }
            actor.zero_grad(s); actor.backward(mb.mean_gradient.get(),2*minibatch,nullptr,s); opt.step(actor.params(),actor.grads(),s);
        }
        const auto stats = download(mb.stats.get(),5,s); finite(stats); kl = stats[3]/rows; if (kl > .01) break;
    }
    gpu::finite_parameters<<<(actor.num_params()+255)/256,256,0,s>>>(actor.params(),actor.num_params(),d.fault.get());
    require(download(d.fault.get(),1,s)[0] == 0 && opt.steps()-before >= 40 && opt.steps()-before <= 80, "Actor update integrity/count fault"); return kl;
}

void zero_fixtures(const fs::path& dir) {
    require(fs::create_directory(dir), "Zero-step fixture identity exists"); rest_zero_checks(dir);
    const std::set<int> starts{0,20,80,100,160,180,240,260};
    for (int t = 0; t < 320; ++t) {
        auto it = starts.upper_bound(t); const int start = *(--it); auto next = std::next(it);
        require(age(t) == t-start && length(t) == (next == starts.end()?320:*next)-start, "Independent mixed-duration schedule failed");
    }
    std::vector<float> raw(batch_rows);
    for (int r = 0; r < batch_rows; ++r) raw[r] = float(cell(r)) + (mode(r)?.5f:2.f)*(((r%32/8)%2)?1.f:-1.f);
    const auto b = fit_baseline(raw); const auto s = signal(raw,b);
    for (int j = 0; j < cells; ++j) require(b.mean[j] == j, "Duration baseline synthetic mean differs");
    require(b.rms[0] == 2 && b.rms[1] == .5 && s.bias[0] == 0 && s.bias[1] == 0 && s.share == .5, "Duration scale/sign synthetic differs");
    for (int r = 0; r < batch_rows; ++r) if (!fitting(r)) raw[r] += 100;
    const auto other = fit_baseline(raw);
    require(b.mean == other.mean && b.n == other.n && b.rms[0] == other.rms[0] && b.rms[1] == other.rms[1], "Held-out rows entered frozen baseline");
    // Test the NEW GPU boundary at each type of nonzero reset, without stepping.
    for (int slot : {20,80}) {
        Stream stream; Rollout d(stream.value);
        gpu::initialize<<<1,32,0,stream.value>>>(d.worlds.get(),d.tasks.get(),32,d.fault.get(),training_episode);
        auto w = download(d.worlds.get(),32,stream.value); auto tasks = download(d.tasks.get(),32,stream.value);
        const auto initial = w; const int prior = length(slot-1);
        for (int i = 0; i < 32; ++i) {
            w[i].physical.control_steps = prior; w[i].physical.substeps = prior*10; w[i].attempted_substeps = prior*10;
            w[i].physical.body.position.x += .25f; w[i].physical.body.velocity[0] = .1f; tasks[i].controls = prior;
        }
        d.worlds.copy_from_host(w.data(),32); d.tasks.copy_from_host(tasks.data(),32);
        const std::vector<float> x(32*41,.3f), z(32*4,.2f); d.previous_obs.copy_from_host(x.data(),x.size()); d.previous_action.copy_from_host(z.data(),z.size());
        boundary<<<1,32,0,stream.value>>>(d.worlds.get(),d.tasks.get(),d.previous_obs.get(),d.previous_action.get(),d.history.get(),d.pre_action_speed.get(),d.fault.get(),slot,training_episode);
        const auto got = download(d.worlds.get(),32,stream.value);
        const auto t = download(d.tasks.get(),32,stream.value);
        const auto history = download(d.history.get(),32,stream.value), px = download(d.previous_obs.get(),32*41,stream.value), pz = download(d.previous_action.get(),32*4,stream.value);
        require(download(d.fault.get(),1,stream.value)[0] == 0, "New synthetic GPU boundary fault");
        for (int i = 0; i < 32; ++i) {
            const bool h = mode(i) != 0; require(fields(got[i]) == fields(h?initial[i]:w[i]) && t[i].controls == (h?0:prior) && history[i] == (h?0.f:1.f), "New boundary did not isolate hold lanes");
            for (int j = 0; j < 41; ++j) require(px[i*41+j] == (h?0.f:.3f), "New boundary leaked AR observation");
            for (int j = 0; j < 4; ++j) require(pz[i*4+j] == (h?0.f:.2f), "New boundary leaked AR action");
        }
    }
    // Exercise the actual terminal writer at both lengths and at a bad boundary.
    {
        Stream stream; Rollout d(stream.value); std::vector<m::ForwardTask> tasks(32);
        for (int slot : {0,18,19,20,78,79,80,319}) {
            for (int i = 0; i < 32; ++i) tasks[i].controls = (mode(i)?age(slot):slot)+1;
            d.tasks.copy_from_host(tasks.data(),32);
            terminal<<<1,32,0,stream.value>>>(d.tasks.get(),d.done.get(),d.fault.get(),slot,training_episode);
            const auto done = download(d.done.get(),32,stream.value);
            require(download(d.fault.get(),1,stream.value)[0] == 0, "Synthetic terminal writer fault");
            for (int i = 0; i < 32; ++i) require(done[i] == float(mode(i)?age(slot)+1==length(slot):slot+1==320), "Synthetic terminal mask differs");
        }
        tasks[0].controls = 0; d.tasks.copy_from_host(tasks.data(),32);
        terminal<<<1,32,0,stream.value>>>(d.tasks.get(),d.done.get(),d.fault.get(),319,training_episode);
        require(download(d.fault.get(),1,stream.value)[0] == 9, "Wrong terminal boundary was not rejected");
    }
    std::ofstream f(dir/"credit-zero-step.json"); f << "{\"passed\":true,\"baseline_cells\":800,\"fit_only\":true,\"duration_age_schedule\":true,\"mixed_GPU_boundaries\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void result(const fs::path& file, const Budget& b, const std::string& fault, const std::string& gate,
            int updates_done, int steps, double prior, bool qualified) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::ofstream f(file); f << std::setprecision(17) << "{\"controls\":" << b.controls << ",\"substeps\":" << b.substeps
        << ",\"active_seconds\":" << elapsed << ",\"prior_active_seconds\":" << prior << ",\"PPO_updates\":" << updates_done
        << ",\"actor_steps\":" << steps << ",\"critic_steps\":0,\"critic_forward_calls\":0,\"fault\":" << std::quoted(fault)
        << ",\"gate\":" << std::quoted(gate) << ",\"qualified\":" << (qualified?"true":"false")
        << ",\"closed\":true,\"runtime_adopted\":false,\"retries\":0}\n"; f.close(); require(bool(f), "Cannot write closed result");
}
int prepare(const fs::path& root, const fs::path& protocol) {
    require(!fs::exists(root) && fs::is_regular_file(protocol), "New identity or frozen protocol missing");
    require(fs::create_directories(root), "Cannot create isolated identity"); fs::copy_file(protocol,root/"approved-protocol.md");
    configure(root,false); fs::copy_file(parent_checkpoint_path,root/"parent-checkpoint.bin"); fs::copy_file(parent_protocol_path,root/"parent-protocol.bin");
    const auto bytes = protocol_bytes(root); cp::write_atomic((root/"protocol.bin").string(),bytes);
    cp::write_atomic((root/"migrated-initial.bin").string(),rl::write_appended_hold(migrated41,cp::checksum(bytes)));
    const auto restored = rl::read_appended_hold(cp::read((root/"migrated-initial.bin").string()),cp::checksum(bytes));
    require(restored.ppo.actor.parameters == resume_state->actor.parameters && restored.ppo.actor_adam.steps == 21700 && restored.actor_column_age == 0, "Migration checkpoint identity mismatch");
    zero_fixtures(root/"prepare-zero");
    std::ofstream f(root/"prepared.json"); f << "{\"prepared\":true,\"physical_controls\":0,\"optimizer_steps\":0,\"controls_cap\":188640,\"training_controls_cap\":81920,\"PPO_updates_cap\":8}\n"; return 0;
}
int preflight(const fs::path& root) {
    configure(root,true); validate_protocol(root); zero_fixtures(root/"verified-zero");
    std::ofstream f(root/"preflight-passed.txt"); f << "Zero-step CPU/CUDA migration, schedule and fit-only fixtures passed.\n"; return 0;
}
int calibrate(const fs::path& root) {
    configure(root,true); validate_protocol(root); require(fs::exists(root/"preflight-passed.txt"), "Preflight missing");
    require(fs::create_directory(root/"calibration-started"), "Calibration already attempted; no retries");
    failure_directory = root/"calibration-started";
    started = std::chrono::steady_clock::now(); active_wall_cap = 120; Budget budget; budget.maximum_controls = calibration_controls; budget.maximum_substeps = calibration_controls*10; budget.seconds = 120;
    std::string fault, gate;
    try {
        const auto baseline = assess(resume_state->actor.parameters,root/"baseline",budget);
        require(baseline.short_pass == 4 && baseline.long_pass == 28 && baseline.hold_pass == 0 && preserved_long_windows(root/"baseline"/"long"), "Original 336 behavioral reference changed");
        calibration_collect = true; Arm a; require(fs::create_directory(root/"calibration"), "Calibration output exists"); create_actor(a,root/"calibration",false);
        const auto recorded = batch(336,a,budget); measured_actor_parity(a);
        require(a.actor->get_params() == resume_state->actor.parameters && !a.actor_opt, "Calibration actor changed");
        const auto fixed = fit_baseline(recorded.raw); const auto id = cp::checksum(protocol_bytes(root));
        cp::write_atomic((root/"fixed-baseline.bin").string(),baseline_bytes(fixed,id));
        require(baseline_bytes(read_baseline(root,id),id) == baseline_bytes(fixed,id), "Baseline roundtrip mismatch");
        const auto s = signal(recorded.raw,fixed); const auto hs = scale_stats(recorded.raw,true), sw = scale_stats(recorded.raw,false);
        const bool pass = recorded.heldquiet >= 256 && s.share >= .1 && s.share <= .9 && hs.pass && sw.pass &&
            s.duration_offset_fraction[0] <= .25 && s.duration_offset_fraction[1] <= .25;
        std::ofstream f(root/"calibration-report.json"); f << std::setprecision(17) << "{\"passed\":" << (pass?"true":"false")
            << ",\"heldout_hold_quiet\":" << recorded.heldquiet << ",\"heldout_hold_rows\":2560,\"all_hold_quiet\":" << recorded.quiet
            << ",\"hold_share\":" << s.share << ",\"hold_bias\":" << s.bias[1] << ",\"hold_RMSE\":" << s.rmse[1]
            << ",\"hold_fit_residual_RMS\":" << fixed.rms[1] << ",\"swim_fit_residual_RMS\":" << fixed.rms[0]
            << ",\"duration20_offset_energy_fraction\":" << s.duration_offset_fraction[0]
            << ",\"duration60_offset_energy_fraction\":" << s.duration_offset_fraction[1]
            << ",\"neutral_0p03_rows\":" << recorded.neutral_rows[0] << ",\"neutral_0p03_longest_controls\":" << recorded.neutral_longest[0]
            << ",\"neutral_0p1_rows\":" << recorded.neutral_rows[1] << ",\"neutral_0p1_longest_controls\":" << recorded.neutral_longest[1]
            << ",\"max_GAE_error\":" << recorded.gae_error << ",\"max_oldlogp_error\":" << recorded.logp_error
            << ",\"fit_cells\":800,\"hold_fit_per_cell\":16,\"swim_fit_per_cell\":4,\"hold_return_scale\":";
        print_scale(f,hs); f << ",\"swim_return_scale\":"; print_scale(f,sw); f << ",\"actor_steps\":0,\"critic_steps\":0}\n";
        f.close(); require(bool(f), "Could not write calibration report"); if (!pass) gate = "calibration_signal_or_occupancy_or_scale";
        require(budget.controls == calibration_controls && budget.substeps == calibration_controls*10, "Calibration accounting changed"); wall_guard();
    } catch (const std::exception& e) { fault = e.what(); std::cerr << fault << std::endl; }
    result(root/"calibration-result.json",budget,fault,gate,0,0,0,false);
    if (fault.empty() && gate.empty()) {
        cp::Writer ready; ready.value(ready_magic); ready.value(uint32_t{1}); ready.value(cp::checksum(protocol_bytes(root)));
        for (const auto& file : {root/"fixed-baseline.bin",root/"calibration-report.json",root/"calibration-result.json",root/"baseline"/"long"/"sustained-windows.csv"}) ready.value(text_hash(file));
        ready.value(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
        cp::write_atomic((root/"calibration-ready.bin").string(),ready.data);
    }
    return fault.empty()?0:3;
}
double release(const fs::path& root, uint64_t id) {
    const auto bytes = cp::read((root/"calibration-ready.bin").string()); cp::Reader r{bytes};
    require(r.value<uint32_t>() == ready_magic && r.value<uint32_t>() == 1 && r.value<uint64_t>() == id, "Calibration ready identity wrong");
    for (const auto& file : {root/"fixed-baseline.bin",root/"calibration-report.json",root/"calibration-result.json",root/"baseline"/"long"/"sustained-windows.csv"}) require(r.value<uint64_t>() == text_hash(file), "Calibrated evidence changed");
    const double prior = r.value<double>(); r.finish(); require(std::isfinite(prior) && prior > 0 && prior < 120, "Invalid prior active time"); return prior;
}
int train(const fs::path& root) {
    configure(root,true); validate_protocol(root); const auto id = cp::checksum(protocol_bytes(root)); const auto prior = release(root,id);
    const auto fixed = read_baseline(root,id);
    const auto original = window_keys(root/"baseline"/"long"/"sustained-windows.csv");
    require(original.size() == 28, "Original retained-window set changed"); require(fs::create_directory(root/"training-started"), "Training already attempted; no retry");
    failure_directory = root/"training-started";
    started = std::chrono::steady_clock::now(); active_wall_cap = 120-prior;
    Budget budget; budget.maximum_controls = total_controls; budget.maximum_substeps = total_controls*10; budget.seconds = active_wall_cap;
    budget.controls = calibration_controls; budget.substeps = calibration_controls*10;
    std::string fault, gate; Arm a; int completed = 0; bool qualified = false;
    std::ofstream progress(root/"updates.jsonl"); progress << std::setprecision(17);
    try {
        create_actor(a,root/"training-started",true); save_actor(a.root,a,id); calibration_collect = false;
        for (int update = 336; update < 344; ++update) {
            const auto recorded = batch(update,a,budget); const auto s = signal(recorded.raw,fixed);
            if (update == 336 && (recorded.heldquiet < 256 || s.share < .1 || s.share > .9 ||
                s.duration_offset_fraction[0] > .25 || s.duration_offset_fraction[1] > .25)) { gate = "first_fresh_signal_or_occupancy"; break; }
            const double kl = update_actor(update,a,fixed,recorded.raw); a.completed = update+1; ++completed; save_actor(a.root,a,id);
            require(a.actor_opt->fresh_steps() <= max_actor_steps && a.training_controls == uint64_t(completed)*batch_rows, "Training budget/optimizer age exceeded");
            const auto measured = assess(a.actor->get_params(),a.root/("assessment-"+std::to_string(a.completed)),budget);
            const bool retained = measured.short_pass == 4 && std::includes(measured.keys.begin(),measured.keys.end(),original.begin(),original.end());
            const bool pass = retained && measured.hold_pass == 6; qualified = qualified || pass;
            progress << "{\"update\":" << a.completed << ",\"short\":" << measured.short_pass << ",\"long\":" << measured.long_pass
                << ",\"holds\":" << measured.hold_pass << ",\"retained\":" << (retained?"true":"false") << ",\"qualified\":" << (pass?"true":"false")
                << ",\"hold_heldout_share\":" << s.share << ",\"hold_quiet_fraction\":" << double(recorded.quiet)/5120
                << ",\"hold_heldout_bias\":" << s.bias[1] << ",\"KL\":" << kl << ",\"actor_steps\":" << a.actor_opt->fresh_steps()
                << ",\"neutral_0p03_rows\":" << recorded.neutral_rows[0] << ",\"neutral_0p03_longest_controls\":" << recorded.neutral_longest[0]
                << ",\"neutral_0p1_rows\":" << recorded.neutral_rows[1] << ",\"neutral_0p1_longest_controls\":" << recorded.neutral_longest[1]
                << ",\"controls\":" << budget.controls << "}\n"; progress.flush(); require(bool(progress), "Could not write assessment progress");
            if (!retained) { gate = "regressed_swimming"; break; } if (pass) { gate = "qualified_early_stop_no_adoption"; break; }
        }
        wall_guard();
    } catch (const std::exception& e) { fault = e.what(); std::cerr << fault << std::endl; }
    result(root/"training-result.json",budget,fault,gate,completed,a.actor_opt?a.actor_opt->fresh_steps():0,prior,qualified);
    return fault.empty()?0:3;
}
} // namespace credit
int main(int argc, char** argv) {
    try {
        require(argc >= 3, "Use mode run-directory [protocol-file]"); const std::string mode = argv[1]; const auto root = fs::absolute(argv[2]);
        if (mode == "--prepare") { require(argc == 4, "Prepare needs frozen protocol path"); return credit::prepare(root,fs::absolute(argv[3])); }
        require(argc == 3, "Unexpected study argument");
        if (mode == "--preflight") return credit::preflight(root);
        if (mode == "--calibrate") return credit::calibrate(root);
        if (mode == "--train") return credit::train(root);
        throw std::runtime_error("Unknown credit study mode");
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 2; }
}
