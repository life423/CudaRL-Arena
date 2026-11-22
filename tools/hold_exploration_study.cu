// One opt-in, bounded hold-exploration lesson. No runtime controller installation.
// Reuses shared physics, likelihood, GAE, GPU MLP/Adam and frozen assessments.
#include "motor/joint_motor.h"
namespace explore {
void physical_failure(const cudarl::joint_motor::World&, const cudarl::physics::BodyState&,
                      const cudarl::physics::BodyState&, int, int);
void physical_step(const cudarl::joint_motor::World&, const cudarl::physics::BodyState&, int, int);
}
#define CUDARL_PHYSICAL_FAILURE_HOOK explore::physical_failure
#define CUDARL_PHYSICAL_STEP_HOOK explore::physical_step
#define CUDARL_REST_HOLD_MEASURED_PHASE_B 1
#define CUDARL_REST_HOLD_MEASURED_BASELINE 1
#define CUDARL_REST_HOLD_RESET_STUDY 1
#include "rest_hold_study.cu"
#include <set>
#include "hold_exploration_math.h"
#include "core/rng.h"

namespace explore {
static_assert(observations == 41 && actions == 4, "Compile with CUDARL_APPEND_HOLD41");
constexpr int batch_rows = 10240, cells = 1280, calibration_controls = 20960;
constexpr int total_controls = 691680, max_actor_steps = 2560;
constexpr uint32_t protocol_magic = 0x32454850, baseline_magic = 0x32454842;
constexpr uint32_t actor_magic = 0x32454841, ready_magic = 0x32454852;
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
int gain_bin(int slot,int world) { return (world/8+(slot/80))%4; }
int cell(int row) {
    const int t=row/32,pose=row%2;
    return mode(row)?640+pose*320+gain_bin(t,row%32)*80+(length(t)==20?0:20)+age(t):pose*320+t;
}
struct Extra {
    device_ptr<float> gains{4}, gradient{4}, base{batch_rows*4}, previous_base{batch_rows*4};
    device_ptr<float> amplitudes{batch_rows}, bins{batch_rows}, hold{batch_rows}, post{batch_rows*5};
    device_ptr<float> world_amplitude{32}, mb_base{256*4}, mb_previous_base{256*4};
    device_ptr<float> effective_marginal_mean{32*4};
    device_ptr<float> mb_amplitude{256}, mb_hold{256}, row_gradient{256*4};
};
std::unique_ptr<Extra> extra;
std::ofstream all_poses;
int cpu_episode=-1;
std::array<float,4> assessment_gains{1,1,1,1};
void write_pose(std::ostream& f,const char* category,int id,int control,const p::BodyState& body,const float* gain) {
    f<<category<<','<<id<<','<<control<<','<<control*.05<<','<<body.position.x<<','<<body.position.y<<','<<body.position.z
     <<','<<body.orientation.w<<','<<body.orientation.x<<','<<body.orientation.y<<','<<body.orientation.z;
    for(float x:body.angle)f<<','<<x;for(int j=0;j<4;++j)f<<','<<gain[j];f<<'\n';
}
void pose_header(std::ostream& f) {
    f<<"category,case,control,time_s,position_x_model,position_y_model,position_z_model,quat_w,quat_x,quat_y,quat_z,angle0,angle1,angle2,angle3,gain0,gain1,gain2,gain3\n";
}
void physical_step(const m::World& world,const p::BodyState& before,int,int) {
    if(!all_poses.is_open())return;
    const int control=int(world.physical.control_steps);
    if(control==1)++cpu_episode;
    require(cpu_episode>=0&&cpu_episode<12,"CPU pose episode count differs");
    const char* category=cpu_episode<4?"short":cpu_episode<10?"hold":"long";
    const int id=cpu_episode<4?cpu_episode:cpu_episode<10?cpu_episode-4:cpu_episode-10;
    const float ones[4]={1,1,1,1};const float* gains=cpu_episode>=4&&cpu_episode<10?assessment_gains.data():ones;
    if(control==1)write_pose(all_poses,category,id,0,before,gains);
    write_pose(all_poses,category,id,control,world.physical.body,gains);
    require(bool(all_poses),"Could not save CPU physical pose");
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
        require(b.n[j] == (4), "Duration-conditioned fit occupancy changed");
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
    for (int n : {20,60,80,320,32,41,4,cells,calibration_controls,total_controls,total_controls*10,32,max_actor_steps,120}) w.value(n);
    for(float x:{.01f,1.f,.0025f})w.value(x);w.value(uint32_t{0x61a10e11u});
    const std::string method = "one frozen336MLP; four learned holding gains; segment log-uniform amplitude .01..1; 4 gain strata; fit-only pose/duration/age/bin baseline; fixed task RMS; fresh gain Adam .0025; 32 updates max; no adoption";
    w.array(std::vector<uint8_t>(method.begin(), method.end())); return w.data;
}
void configure(const fs::path& root, bool copied) {
    configure_rest(root, copied); control_cap = total_controls; substep_cap = total_controls * 10;
    training_cap = 327680; updates = 32; active_wall_cap = 120;
    require(horizon == 320 && rows == batch_rows && seed == 1247037243u && correlation == .8f,
            "Credit study inherited configuration mismatch");
}
void validate_protocol(const fs::path& root) {
    require(cp::read((root / "protocol.bin").string()) == protocol_bytes(root), "Frozen exploration protocol changed");
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
    for (int j = 0; j < cells; ++j) require(std::isfinite(b.mean[j]) && b.n[j] == (4), "Bad baseline cell");
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
namespace math = cudarl::hold_exploration;
__global__ void sample_policy(const float* base,const float* previous_action,const float* history,const float* sigma,
                              const float* gains,float* world_amp,int slot,int offset,uint32_t action_seed,uint32_t counter,
                              float* action,float* logp,float* amplitudes,float* bins,float* recorded_base,float* recorded_previous,float* marginal) {
    const int i=threadIdx.x;if(i>=32)return;const bool hold=(i/2)%2;
    const int bin=(i/8+slot/80)%4;
    if(history[i]==0){const float u=cudarl::rng::uniform(0x61a10e11u,counter,uint32_t(i));
        world_amp[i]=hold?expf(logf(.01f)+(float(bin)+u)*(logf(100.f)/4)):1.f;}
    const float g=world_amp[i];float mu[4],logs[4];
    math::conditional(base+i*4,base+(32+i)*4,previous_action+i*4,gains,sigma,hold,g,.8f,history[i]!=0,mu,logs);
    for(int j=0;j<4;++j){action[i*4+j]=mu[j]+expf(logs[j])*cudarl::rng::normal(action_seed,counter,uint32_t(i*4+j));
        marginal[i*4+j]=hold?g*gains[j]*base[i*4+j]:base[i*4+j];
        recorded_base[(offset+i)*4+j]=base[i*4+j];recorded_previous[(offset+i)*4+j]=base[(32+i)*4+j];}
    logp[i]=rl::gaussian_logp(mu,logs,action+i*4,4);amplitudes[offset+i]=g;bins[offset+i]=hold?float(bin):-1.f;
}
__global__ void post_metrics(const m::World* worlds,const m::ForwardTask* tasks,float* out,int* fault) {
    const int i=threadIdx.x;if(i>=32||*fault)return;const auto z=m::measure(tasks[i],worlds[i]);
    out[i*5]=p::norm(z.velocity_m_s);out[i*5+1]=p::norm(z.hold_error_m);out[i*5+2]=z.depth_m;
    out[i*5+3]=z.lateral_m;out[i*5+4]=z.tilt_deg;
    for(int j=0;j<5;++j)if(!p::finite_number(out[i*5+j]))atomicExch(fault,10);
}
__global__ void gain_loss(const float* base,const float* prevbase,const float* prevaction,const float* history,
                          const float* gains,const float* logs,const float* holding,const float* amplitude,
                          const float* action,const float* oldp,const float* advantage,int n,float* gradient,float* statistics) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto terms=math::loss(base+i*4,prevbase+i*4,prevaction+i*4,gains,logs,holding[i]!=0,amplitude[i],.8f,
                                history[i]!=0,action+i*4,oldp[i],advantage[i],1.f/n,gradient+i*4);
    statistics[i*5]=terms.pg_loss;statistics[i*5+1]=terms.v_loss;statistics[i*5+2]=terms.entropy;
    statistics[i*5+3]=terms.approx_kl;statistics[i*5+4]=terms.clipped;
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
        sample_policy<<<1,32,0,s>>>(a.actor->output(),oldaction,hist,d.sigma.get(),extra->gains.get(),extra->world_amplitude.get(),
            t,offset,rest_action_seed(),rest_action_counter(update,t),d.action.get()+offset*4,d.logp.get()+offset,
            extra->amplitudes.get(),extra->bins.get(),extra->base.get(),extra->previous_base.get(),extra->effective_marginal_mean.get());
        CUDA_CHECK_THROW(cudaMemsetAsync(d.value.get()+offset,0,32*sizeof(float),s));
        gpu::advance<<<1,32,0,s>>>(d.worlds.get(),d.tasks.get(),d.action.get()+offset*4,d.reward.get()+offset,
            d.done.get()+offset,d.counts.get(),32,d.fault.get(),extra->effective_marginal_mean.get(),training_episode,d.cost_trace.get()+offset*5);
        terminal<<<1,32,0,s>>>(d.tasks.get(),d.done.get()+offset,d.fault.get(),t,training_episode);
        post_metrics<<<1,32,0,s>>>(d.worlds.get(),d.tasks.get(),extra->post.get()+offset*5,d.fault.get());
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
void create_actor(Arm& a,const fs::path& root,bool optimizer) {
    a.root=root;a.stream=std::make_unique<Stream>();a.actor=std::make_unique<nn::GpuMlp>(spec(4),rows,seed);
    a.actor->set_params(resume_state->actor.parameters);a.data=std::make_unique<Rollout>(a.stream->value);a.batch=std::make_unique<Batches>();
    extra=std::make_unique<Extra>();const float ones[4]={1,1,1,1};extra->gains.copy_from_host(ones,4);
    std::vector<float> flags(rows);for(int r=0;r<rows;++r)flags[r]=float(mode(r));extra->hold.copy_from_host(flags.data(),rows);
    if(optimizer){auto config=initial_optimizer_config(true);config.lr=.0025f;
        a.actor_opt=std::make_unique<nn::GpuAdam>(4,config);require(a.actor_opt->steps()==0,"Fresh gain Adam clock differs");}
}
struct Batch { std::vector<float> raw; int quiet = 0, heldquiet = 0;
    int neutral_rows[2]{}, neutral_longest[2]{}, complete_quiet=0, quiet_groups[2][2]{}; double gae_error = 0, logp_error = 0; };
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
            } else require(std::abs(std::sqrt(x[3]*x[3]+x[4]*x[4]+x[5]*x[5])-.4f)<1e-5f,"Swimming request/gain mask mismatch");
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
    const auto base=download(extra->base.get(),rows*4,s),prevbase=download(extra->previous_base.get(),rows*4,s);
    const auto amplitude=download(extra->amplitudes.get(),rows,s),bins=download(extra->bins.get(),rows,s);
    const auto post=download(extra->post.get(),rows*5,s),gains=download(extra->gains.get(),4,s);
    const auto oldlogp=download(d.logp.get(),rows,s);const auto logs=policy_sigma();
    for(int e=0;e<32;++e){for(int start=0;start<320;){const bool h=mode(e)!=0;const int len=h?length(start):320;
        bool quiet=h;const float firstg=amplitude[start*32+e];
        for(int t=start;t<start+len;++t){const int r=t*32+e;
            require(amplitude[r]==firstg&&(!h?amplitude[r]==1.f:amplitude[r]>=.01f&&amplitude[r]<=1.f),"Segment amplitude changed or out of bounds");
            require(bins[r]==float(h?gain_bin(t,e):-1),"Recorded gain stratum differs");
            double bcur[4],bprev[4],pa[4],ga[4],sig[4],act[4],mu[4],stddev[4];
            for(int j=0;j<4;++j){bcur[j]=base[r*4+j];bprev[j]=prevbase[r*4+j];pa[j]=oldaction[r*4+j];ga[j]=gains[j];sig[j]=logs[j];act[j]=latent[r*4+j];}
            math::conditional(bcur,bprev,pa,ga,sig,h,double(amplitude[r]),double(.8f),history[r]!=0,mu,stddev);
            const double lp=rl::gaussian_logp(mu,stddev,act,4);out.logp_error=std::max(out.logp_error,std::abs(lp-oldlogp[r]));
            quiet=quiet&&speed[r]<=.0005f&&post[r*5]<=.0005f&&post[r*5+1]<=.002f&&post[r*5+2]<=.001f&&post[r*5+3]<=.002f&&post[r*5+4]<=10.f;
        }
        if(quiet){++out.complete_quiet;++out.quiet_groups[e%2][len==60];}start+=len;
    }}
    require(out.logp_error<1e-4,"Actual scaled action density rescore failed");
    require(a.actor->get_params()==resume_state->actor.parameters,"Frozen base actor changed");cache_batch(a,update);
    cp::Writer proof;proof.value(uint32_t{0x31504548});proof.value(uint32_t{1});proof.array(base);proof.array(prevbase);
    proof.array(amplitude);proof.array(bins);proof.array(post);proof.array(gains);
    cp::write_atomic((a.root/("exploration-evidence-"+std::to_string(update)+".bin")).string(),proof.data);
    std::ofstream csv(a.root/("rows-"+std::to_string(update)+".csv"));
    csv << std::setprecision(12) << "slot,world,task,pose,fit,cell,local_age,segment_length,speed_m_s,done,history,reward,return,logp,amplitude,gain_bin,post_speed_m_s,post_anchor_m,post_depth_m,post_lane_m,post_tilt_deg\n";
    for (int r = 0; r < rows; ++r) {
        const int t = r/32, h = mode(r);
        csv << t << ',' << r%32 << ',' << h << ',' << (r%2?3:0) << ',' << fitting(r) << ',' << cell(r) << ','
            << (h?age(t):t) << ',' << (h?length(t):320) << ',' << speed[r] << ',' << done[r] << ',' << history[r]
            << ',' << reward[r] << ',' << out.raw[r] << ',' << oldlogp[r] << ',' << amplitude[r] << ',' << bins[r];
        for(int j=0;j<5;++j)csv<<','<<post[r*5+j];csv<<'\n';
    }
    require(bool(csv), "Could not write saved diagnostic rows"); return out;
}

RestAssessment evaluate_gain_rest(const std::vector<float>& params,const fs::path& root,Budget& budget){
    require(fs::create_directory(root),"Rest assessment exists");nn::CpuMlp<float> actor(spec(4));actor.params()=params;RestAssessment result;
    std::ofstream poses(root/"poses.csv");poses<<std::setprecision(12);pose_header(poses);
    std::ofstream trace(root/"controls.csv"),summary(root/"summary.csv");trace<<std::setprecision(12);summary<<std::setprecision(12);
    trace<<"case,pose,offset_mm,control,time_s,speed_mm_s,forward_velocity_mm_s,forward_mm,depth_mm,lane_mm,tilt_deg,roll_deg,pitch_deg,anchor_error_mm";
    for(int j=0;j<4;++j)trace<<",command"<<j<<",angle"<<j<<",rate"<<j<<",last_torque"<<j;trace<<'\n';
    summary<<"case,pose,offset_mm,passed,max_speed_mm_s,RMS_speed_mm_s,max_anchor_error_mm,max_depth_mm,max_lane_mm,max_tilt_deg,final8_forward_excursion_mm,net_forward_mm,final_speed_mm_s,final_depth_mm,positive_work_J\n";
    for(int id=0;id<6;++id){auto w=m::initial_world(id%2?3:0);w.arena=m::long_swim_arena();m::ForwardTask t;const auto c=rest_config();const float offset=id/2==0?0.f:id/2==1?-.0005f:.0005f;
        require(m::begin_forward(w,c,t)&&m::offset_episode_start(w,t,c,offset)&&m::update_velocity_request(w,c,t),"Rest assessment setup failed");
        const auto anchor=t.hold_anchor_model,start=t.start_com_model,heading=t.request.heading;
        m::RestHoldAssessment score;
        require(score.record(0,m::measure(t,w)),"Invalid rest assessment reset sample");
        write_pose(poses,"hold",id,0,w.physical.body,assessment_gains.data());
        for(int k=1;k<=m::RestHoldAssessment::controls;++k){require(m::update_velocity_request(w,c,t)&&p::norm(t.hold_anchor_model-anchor)==0&&p::norm(t.start_com_model-start)==0&&p::norm(t.request.heading-heading)==0,"Rest reference moved");
            float x[41];require(m::observe_appended(t,w,c,x),"Rest assessment observation failed");actor.forward(x,1);float means[4];for(int j=0;j<4;++j)means[j]=assessment_gains[j]*actor.output()[j];m::Action action;require(m::action_from_latent(means,action),"Rest assessment action failed");
            p::BodyState before{},candidate{};try{budget.step(w,action.command,nullptr,&before,&candidate);}catch(...){cp::Writer proof;proof.value(uint32_t{1});proof.value(uint32_t{sizeof(m::World)});proof.value(w);proof.value(before);proof.value(candidate);cp::write_atomic((root/"failure-state.bin").string(),proof.data);throw;}
            const auto interval=m::record_hold_control(t,w,c);require(interval.accepted,"Rest assessment record rejected");const auto& z=interval.metrics;require(score.record(k,z),"Invalid rest assessment control sample");
            write_pose(poses,"hold",id,k,w.physical.body,assessment_gains.data());
            trace<<id<<','<<(id%2?3:0)<<','<<offset*1000<<','<<k<<','<<k*.05<<','<<p::norm(z.velocity_m_s)*1000<<','<<p::dot(z.velocity_m_s,heading)*1000<<','<<z.forward_m*1000<<','<<z.signed_depth_m*1000<<','<<z.lateral_m*1000<<','<<z.tilt_deg<<','<<z.roll_deg<<','<<z.pitch_deg<<','<<p::norm(z.hold_error_m)*1000;
            for(int j=0;j<4;++j)trace<<','<<action.command[j]<<','<<w.physical.body.angle[j]<<','<<w.physical.body.velocity[6+j]<<','<<w.physical.last_actual_torques[j];trace<<'\n';}
        require(score.complete(),"Incomplete rest assessment trajectory");result.passed+=score.passed();++result.completed;const auto last=m::measure(t,w);
        summary<<id<<','<<(id%2?3:0)<<','<<offset*1000<<','<<score.passed()<<','<<score.peak_speed_m_s()*1000<<','<<score.rms_speed_m_s()*1000<<','<<score.peak_anchor_m()*1000<<','<<score.peak_depth_m()*1000<<','<<score.peak_lane_m()*1000<<','<<score.peak_tilt_degrees()<<','<<score.final_excursion_m()*1000<<','<<last.forward_m*1000<<','<<p::norm(last.velocity_m_s)*1000<<','<<last.signed_depth_m*1000<<','<<w.physical.body.work<<'\n';summary.flush();trace.flush();}
    return result;
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
Assessment assess(const std::vector<float>& actor, const fs::path& dir, Budget& b, const std::vector<float>& gains) {
    require(fs::create_directory(dir), "Assessment identity exists"); const int before = b.controls;
    require(gains.size()==4,"Bad deterministic gains");std::copy(gains.begin(),gains.end(),assessment_gains.begin());
    all_poses.open(dir/"all-cpu-poses.csv");all_poses<<std::setprecision(12);pose_header(all_poses);cpu_episode=-1;
    const auto swim = evaluate(actor,dir,"frozen",b);
    const auto hold = evaluate_gain_rest(actor,dir/"rest",b);
    require(fs::create_directory(dir/"long"), "Long assessment identity exists");
    const auto longrun = sustained(actor,dir/"long",b);
    require(longrun.safety_fault.empty() && hold.completed == 6 && b.controls-before == 10720, "Frozen assessment failed or count differs");
    all_poses.close();require(cpu_episode==11,"CPU physical pose capture incomplete");
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
void save_actor(const fs::path& root,const Arm& a,uint64_t id) {
    const auto file=root/("exploration-policy-update-"+std::to_string(a.completed)+".bin");require(!fs::exists(file),"Policy identity exists");
    cp::Writer out,adam;a.actor_opt->save(adam);const auto gains=download(extra->gains.get(),4,a.stream->value);const int steps=a.actor_opt->steps();
    out.value(actor_magic);out.value(uint32_t{1});out.value(id);out.value(cp::checksum(parent_checkpoint_payload));out.value(a.completed);out.value(steps);
    out.value(uint32_t{41});out.value(uint32_t{4});out.array(a.actor->get_params());out.array(gains);out.array(adam.data);cp::write_atomic(file.string(),out.data);
    const auto bytes=cp::read(file.string());cp::Reader in{bytes};
    require(in.value<uint32_t>()==actor_magic&&in.value<uint32_t>()==1&&in.value<uint64_t>()==id&&in.value<uint64_t>()==cp::checksum(parent_checkpoint_payload)&&
        in.value<int>()==a.completed&&in.value<int>()==steps&&in.value<uint32_t>()==41&&in.value<uint32_t>()==4&&
        in.array<float>(2532)==resume_state->actor.parameters&&in.array<float>(4)==gains&&in.array<uint8_t>()==adam.data,"Policy checkpoint roundtrip failed");in.finish();
    cp::Reader ar{adam.data};const auto cfg=ar.value<nn::AdamConfig>();const int age=ar.value<int>();const auto first=ar.array<float>(4),second=ar.array<float>(4);ar.finish();
    require(cfg.lr==.0025f&&age==steps&&steps>=0&&steps<=max_actor_steps&&first.size()==4&&second.size()==4,"Fresh gain Adam checkpoint invalid");
    for(int j=0;j<4;++j)require(std::isfinite(gains[j])&&std::isfinite(first[j])&&std::isfinite(second[j])&&second[j]>=0,"Nonfinite policy/Adam checkpoint");
}
double update_actor(int update,Arm& a,const Baseline& b,const std::vector<float>& raw) {
    auto& d=*a.data;auto& mb=*a.batch;auto& opt=*a.actor_opt;const auto stream=a.stream->value;
    std::vector<float> advantage(rows);for(int r=0;r<rows;++r){advantage[r]=float((raw[r]-b.mean[cell(r)])/b.rms[mode(r)]);require(std::isfinite(advantage[r]),"Invalid fixed residual");}
    d.advantage.copy_from_host(advantage.data(),rows);const int before=opt.steps();double kl=0;
    for(int epoch=0;epoch<epochs;++epoch){CUDA_CHECK_THROW(cudaMemsetAsync(mb.stats.get(),0,5*sizeof(float),stream));
        const auto permutation=rl::make_permutation(rows,seed+uint32_t(update*epochs+epoch));
        for(int first=0;first<rows;first+=256){wall_guard();
            rl::gather_rows(permutation,first,256,4,extra->base.get(),extra->mb_base.get(),stream);
            rl::gather_rows(permutation,first,256,4,extra->previous_base.get(),extra->mb_previous_base.get(),stream);
            rl::gather_rows(permutation,first,256,1,extra->amplitudes.get(),extra->mb_amplitude.get(),stream);
            rl::gather_rows(permutation,first,256,1,extra->hold.get(),extra->mb_hold.get(),stream);
            rl::gather_rows(permutation,first,256,4,d.previous_action.get(),mb.previous_action.get(),stream);
            rl::gather_rows(permutation,first,256,1,d.history.get(),mb.history.get(),stream);
            rl::gather_rows(permutation,first,256,4,d.action.get(),mb.action.get(),stream);
            rl::gather_rows(permutation,first,256,1,d.logp.get(),mb.logp.get(),stream);
            rl::gather_rows(permutation,first,256,1,d.advantage.get(),mb.advantage.get(),stream);
            gain_loss<<<1,256,0,stream>>>(extra->mb_base.get(),extra->mb_previous_base.get(),mb.previous_action.get(),mb.history.get(),
                extra->gains.get(),d.sigma.get(),extra->mb_hold.get(),extra->mb_amplitude.get(),mb.action.get(),mb.logp.get(),mb.advantage.get(),256,
                extra->row_gradient.get(),mb.row_stats.get());
            if(first==0){const auto gradient=download(extra->row_gradient.get(),1024,stream),flags=download(extra->mb_hold.get(),256,stream);
                const auto got=download(mb.advantage.get(),256,stream);int held_count=0,nonzero=0;
                for(int j=0;j<256;++j){require(got[j]==advantage[rl::permute(permutation,j)],"Actual gain minibatch lost residual");
                    for(int k=0;k<4;++k){require(std::isfinite(gradient[j*4+k]),"Nonfinite gain gradient");if(!flags[j])require(gradient[j*4+k]==0,"Swimming produced gain gradient");else nonzero+=gradient[j*4+k]!=0;}held_count+=flags[j]!=0;}
                require(held_count>0&&nonzero>0,"No actual holding gain gradient");}
            CUDA_CHECK_THROW(cudaMemsetAsync(extra->gradient.get(),0,4*sizeof(float),stream));
            rl::sum_columns(extra->row_gradient.get(),256,4,extra->gradient.get(),stream);
            rl::sum_columns(mb.row_stats.get(),256,5,mb.stats.get(),stream);opt.step(extra->gains.get(),extra->gradient.get(),stream);
        }
        const auto stats=download(mb.stats.get(),5,stream);finite(stats);kl=stats[3]/rows;if(kl>.01)break;
    }
    const auto gains=download(extra->gains.get(),4,stream);finite(gains);
    require(opt.steps()-before>=40&&opt.steps()-before<=80&&a.actor->get_params()==resume_state->actor.parameters,"Gain optimizer/frozen actor integrity fault");return kl;
}
__global__ void math_fixture(float* output) {
    const int i=threadIdx.x;if(i>=24)return;const bool held=i%2,history=(i/2)%2;
    const float g=(i/4)%3==0?.01f:(i/4)%3==1?.1f:1.f;
    const float base[4]={.8f,-.5f,.6f,-.9f},previous[4]={.7f,-.4f,.55f,-.7f};
    const float gains[4]={.7f,.9f,1.1f,.8f},logs[4]={-1,-1,-1,-1},noise[4]={-.8f,.2f,.6f,-.4f};
    const float pg=held?g:1.f;float pa[4]={.3f*pg,-.2f*pg,.15f*pg,-.4f*pg},mu[4],sigma[4],action[4],gradient[4];
    math::conditional(base,previous,pa,gains,logs,held,g,.8f,history,mu,sigma);
    for(int j=0;j<4;++j)action[j]=mu[j]+expf(sigma[j])*noise[j];
    const float old=rl::gaussian_logp(mu,sigma,action,4),adv=i<12?.75f:-.75f;
    const auto terms=math::loss(base,previous,pa,gains,logs,held,g,.8f,history,action,old,adv,1.f,gradient);
    output[i*14]=old;output[i*14+1]=terms.pg_loss;
    for(int j=0;j<4;++j){output[i*14+2+j]=action[j];output[i*14+6+j]=gradient[j];output[i*14+10+j]=mu[j];}
}
void numerical_fixtures(const fs::path& dir) {
    Stream stream;device_ptr<float> output(24*14);math_fixture<<<1,32,0,stream.value>>>(output.get());
    const auto got=download(output.get(),24*14,stream.value);double max_lp=0,max_gradient=0,max_fd=0,max_jacobian=0;
    for(int i=0;i<24;++i){const bool held=i%2,history=(i/2)%2;const float gf=(i/4)%3==0?.01f:(i/4)%3==1?.1f:1.f;
        const double g=gf,rho=double(.8f),base[4]={double(.8f),double(-.5f),double(.6f),double(-.9f)};
        const double previous[4]={double(.7f),double(-.4f),double(.55f),double(-.7f)},logs[4]={-1,-1,-1,-1};
        double gains[4]={double(.7f),double(.9f),double(1.1f),double(.8f)},pa[4]={double(.3f)*g,double(-.2f)*g,double(.15f)*g,double(-.4f)*g};
        if(!held){pa[0]=.3f;pa[1]=-.2f;pa[2]=.15f;pa[3]=-.4f;}
        double action[4],mu[4],sigma[4],gradient[4];for(int j=0;j<4;++j)action[j]=got[i*14+2+j];
        math::conditional(base,previous,pa,gains,logs,held,g,rho,history,mu,sigma);
        const double lp=rl::gaussian_logp(mu,sigma,action,4),old=got[i*14],adv=i<12?.75:-.75;
        max_lp=std::max(max_lp,std::abs(lp-old));
        (void)math::loss(base,previous,pa,gains,logs,held,g,rho,history,action,old,adv,1.,gradient);
        for(int j=0;j<4;++j){max_gradient=std::max(max_gradient,std::abs(gradient[j]-got[i*14+6+j]));
            const double original=gains[j],epsilon=1e-6;double unused[4];gains[j]=original+epsilon;
            const auto plus=math::loss(base,previous,pa,gains,logs,held,g,rho,history,action,old,adv,1.,unused);
            gains[j]=original-epsilon;const auto minus=math::loss(base,previous,pa,gains,logs,held,g,rho,history,action,old,adv,1.,unused);gains[j]=original;
            max_fd=std::max(max_fd,std::abs((plus.pg_loss-minus.pg_loss)/(2*epsilon)-gradient[j]));
        }
        if(held){double normalized[4],oldnormalized[4],scaledbase[4],scaledprevious[4];
            for(int j=0;j<4;++j){normalized[j]=action[j]/g;oldnormalized[j]=pa[j]/g;scaledbase[j]=gains[j]*base[j];scaledprevious[j]=gains[j]*previous[j];}
            const double normalized_lp=rl::ar_logp(scaledbase,scaledprevious,oldnormalized,logs,4,rho,history,normalized)-4*std::log(g);
            max_jacobian=std::max(max_jacobian,std::abs(normalized_lp-lp));
        }else{double oldmu[4],oldlogs[4];rl::ar_parameters(base,previous,pa,logs,4,rho,history,oldmu,oldlogs);
            for(int j=0;j<4;++j)require(mu[j]==oldmu[j]&&sigma[j]==oldlogs[j]&&gradient[j]==0,"Swimming policy/gradient changed");}
    }
    require(max_lp<2e-5&&max_gradient<2e-4&&max_fd<2e-6&&max_jacobian<1e-10,"Scaled actual-action numerical fixture failed");
    auto config=initial_optimizer_config(true);config.lr=.0025f;nn::GpuAdam first(4,config),second(4,config);cp::Writer before,after;
    first.save(before);cp::Reader reader{before.data};second.load(reader);reader.finish();second.save(after);
    require(first.steps()==0&&second.steps()==0&&before.data==after.data,"Fresh gain optimizer zero-step roundtrip failed");
    std::ofstream report(dir/"exploration-math-zero.json");report<<std::setprecision(17)
      <<"{\"passed\":true,\"CPU_CUDA_cases\":24,\"max_logp_error\":"<<max_lp<<",\"max_gradient_error\":"<<max_gradient
      <<",\"max_finite_difference_error\":"<<max_fd<<",\"max_scaled_Jacobian_error\":"<<max_jacobian
      <<",\"fresh_gain_Adam_roundtrip\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
}
void zero_fixtures(const fs::path& dir) {
    require(fs::create_directory(dir), "Zero-step fixture identity exists"); rest_zero_checks(dir);
    numerical_fixtures(dir);
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
    std::ofstream f(dir/"exploration-zero-step.json"); f << "{\"passed\":true,\"baseline_cells\":1280,\"fit_only\":true,\"duration_age_schedule\":true,\"mixed_GPU_boundaries\":true,\"physical_controls\":0,\"optimizer_steps\":0}\n";
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
    std::ofstream f(root/"prepared.json"); f << "{\"prepared\":true,\"physical_controls\":0,\"optimizer_steps\":0,\"controls_cap\":691680,\"training_controls_cap\":327680,\"PPO_updates_cap\":32}\n"; return 0;
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
        const auto baseline = assess(resume_state->actor.parameters,root/"baseline",budget,{1,1,1,1});
        require(baseline.short_pass == 4 && baseline.long_pass == 28 && baseline.hold_pass == 0 && preserved_long_windows(root/"baseline"/"long"), "Original 336 behavioral reference changed");
        calibration_collect = true; Arm a; require(fs::create_directory(root/"calibration"), "Calibration output exists"); create_actor(a,root/"calibration",false);
        const auto recorded = batch(336,a,budget); measured_actor_parity(a);
        require(a.actor->get_params() == resume_state->actor.parameters && !a.actor_opt, "Calibration actor changed");
        const auto fixed = fit_baseline(recorded.raw); const auto id = cp::checksum(protocol_bytes(root));
        cp::write_atomic((root/"fixed-baseline.bin").string(),baseline_bytes(fixed,id));
        require(baseline_bytes(read_baseline(root,id),id) == baseline_bytes(fixed,id), "Baseline roundtrip mismatch");
        const auto s = signal(recorded.raw,fixed); const auto hs = scale_stats(recorded.raw,true), sw = scale_stats(recorded.raw,false);
        const bool pass = recorded.heldquiet >= 256 && s.share >= .1 && s.share <= .9 && hs.pass && sw.pass &&
            recorded.complete_quiet>=4 && recorded.quiet_groups[0][0]>0 && recorded.quiet_groups[0][1]>0 && recorded.quiet_groups[1][0]>0 && recorded.quiet_groups[1][1]>0;
        std::ofstream f(root/"calibration-report.json"); f << std::setprecision(17) << "{\"passed\":" << (pass?"true":"false")
            << ",\"heldout_hold_quiet\":" << recorded.heldquiet << ",\"heldout_hold_rows\":2560,\"all_hold_quiet\":" << recorded.quiet
            << ",\"complete_quiet_segments\":" << recorded.complete_quiet
            << ",\"quiet_pose0_len20\":" << recorded.quiet_groups[0][0] << ",\"quiet_pose0_len60\":" << recorded.quiet_groups[0][1]
            << ",\"quiet_pose3_len20\":" << recorded.quiet_groups[1][0] << ",\"quiet_pose3_len60\":" << recorded.quiet_groups[1][1]
            << ",\"hold_share\":" << s.share << ",\"hold_bias\":" << s.bias[1] << ",\"hold_RMSE\":" << s.rmse[1]
            << ",\"hold_fit_residual_RMS\":" << fixed.rms[1] << ",\"swim_fit_residual_RMS\":" << fixed.rms[0]
            << ",\"duration20_offset_energy_fraction\":" << s.duration_offset_fraction[0]
            << ",\"duration60_offset_energy_fraction\":" << s.duration_offset_fraction[1]
            << ",\"neutral_0p03_rows\":" << recorded.neutral_rows[0] << ",\"neutral_0p03_longest_controls\":" << recorded.neutral_longest[0]
            << ",\"neutral_0p1_rows\":" << recorded.neutral_rows[1] << ",\"neutral_0p1_longest_controls\":" << recorded.neutral_longest[1]
            << ",\"max_GAE_error\":" << recorded.gae_error << ",\"max_oldlogp_error\":" << recorded.logp_error
            << ",\"fit_cells\":1280,\"hold_fit_per_cell\":4,\"swim_fit_per_cell\":4,\"hold_return_scale\":";
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
        for (int update = 336; update < 368; ++update) {
            const auto recorded = batch(update,a,budget); const auto s = signal(recorded.raw,fixed);
            if (update == 336 && (!scale_stats(recorded.raw,true).pass || !scale_stats(recorded.raw,false).pass || recorded.heldquiet < 256 || s.share < .1 || s.share > .9 ||
                recorded.complete_quiet<4 || recorded.quiet_groups[0][0]==0 || recorded.quiet_groups[0][1]==0 || recorded.quiet_groups[1][0]==0 || recorded.quiet_groups[1][1]==0)) { gate = "first_fresh_signal_or_occupancy"; break; }
            const double kl = update_actor(update,a,fixed,recorded.raw); a.completed = update+1; ++completed; save_actor(a.root,a,id);
            require(a.actor_opt->steps() <= max_actor_steps && a.training_controls == uint64_t(completed)*batch_rows, "Training budget/optimizer age exceeded");
            const auto learned_gains=download(extra->gains.get(),4,a.stream->value);
            const auto measured = assess(a.actor->get_params(),a.root/("assessment-"+std::to_string(a.completed)),budget,learned_gains);
            const bool retained = measured.short_pass == 4 && std::includes(measured.keys.begin(),measured.keys.end(),original.begin(),original.end());
            const bool pass = retained && measured.hold_pass == 6; qualified = qualified || pass;
            progress << "{\"update\":" << a.completed << ",\"short\":" << measured.short_pass << ",\"long\":" << measured.long_pass
                << ",\"holds\":" << measured.hold_pass << ",\"retained\":" << (retained?"true":"false") << ",\"qualified\":" << (pass?"true":"false")
                << ",\"hold_heldout_share\":" << s.share << ",\"hold_quiet_fraction\":" << double(recorded.quiet)/5120
                << ",\"complete_quiet_segments\":" << recorded.complete_quiet
                << ",\"hold_heldout_bias\":" << s.bias[1] << ",\"KL\":" << kl << ",\"actor_steps\":" << a.actor_opt->steps()
                << ",\"neutral_0p03_rows\":" << recorded.neutral_rows[0] << ",\"neutral_0p03_longest_controls\":" << recorded.neutral_longest[0]
                << ",\"neutral_0p1_rows\":" << recorded.neutral_rows[1] << ",\"neutral_0p1_longest_controls\":" << recorded.neutral_longest[1]
                << ",\"gains\":[" << learned_gains[0] << "," << learned_gains[1] << "," << learned_gains[2] << "," << learned_gains[3] << "]"
                << ",\"controls\":" << budget.controls << "}\n"; progress.flush(); require(bool(progress), "Could not write assessment progress");
            if (!retained) { gate = "regressed_swimming"; break; } if (pass) { gate = "qualified_early_stop_no_adoption"; break; }
        }
        wall_guard();
    } catch (const std::exception& e) { fault = e.what(); std::cerr << fault << std::endl; }
    result(root/"training-result.json",budget,fault,gate,completed,a.actor_opt?a.actor_opt->steps():0,prior,qualified);
    return fault.empty()?0:3;
}
} // namespace explore
int main(int argc, char** argv) {
    try {
        require(argc >= 3, "Use mode run-directory [protocol-file]"); const std::string mode = argv[1]; const auto root = fs::absolute(argv[2]);
        if(mode=="--numerical"){require(argc==3&&!fs::exists(root),"Numerical fixture identity exists");fs::create_directories(root);explore::numerical_fixtures(root);return 0;}
        if (mode == "--prepare") { require(argc == 4, "Prepare needs frozen protocol path"); return explore::prepare(root,fs::absolute(argv[3])); }
        require(argc == 3, "Unexpected study argument");
        if (mode == "--preflight") return explore::preflight(root);
        if (mode == "--calibrate") return explore::calibrate(root);
        if (mode == "--train") return explore::train(root);
        throw std::runtime_error("Unknown exploration study mode");
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 2; }
}
