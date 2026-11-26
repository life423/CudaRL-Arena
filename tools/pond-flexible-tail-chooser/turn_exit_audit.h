#pragma once
#include "rl/ppo_math.h"

// Offline forks of the frozen actor. Overrides below are experimental treatments.
namespace turn_exit {
constexpr int horizon = 50; // 40 simulated seconds per branch
constexpr int stochastic_replicates = 16;
constexpr double substeps_per_s = 800.;
struct Budget {
    Clock::time_point deadline;
    uint64_t attempted = 0, accepted = 0;
};

chooser::Result advance(chooser::World& w, int action, const Models& m, Budget& budget) {
    if (Clock::now() >= budget.deadline || budget.attempted + 640 > 4800000)
        throw std::runtime_error("Turn-exit audit exceeded its finite budget");
    if (!chooser::accept(w, action)) throw std::runtime_error("Fork was not at a decision");
    auto r = chooser::advance(w, m.gait.data(), m.packed_motor.data(), larger_pond, chooser::decision_controls);
    budget.attempted += r.attempted_substeps;
    budget.accepted += r.accepted_substeps;
    if (r.fault && r.fault != chooser::Clearance) throw std::runtime_error("Unexpected native fault");
    if (!r.fault && r.controls != chooser::decision_controls) throw std::runtime_error("Incomplete decision");
    if (!r.fault) {
        const auto movement = tail::fast::feedback_sample(w.motor);
        const float gap = float(tail::submerged_gap_m(larger_pond, w.motor.navigation.motion.body, w.pose));
        r.reward += tail::curved_bank::reward(tail::curved_bank::observe(w.motor, larger_pond),
            gap, physics::norm(movement.velocity_m_s), r.controls * tail::motor_control_s);
    }
    if (!std::isfinite(r.reward)) throw std::runtime_error("Nonfinite collector reward");
    return r;
}

struct Policy {
    cudarl::nn::CpuMlp<float> actor{chooser::actor_spec()};
    cudarl::nn::CpuMlp<float> critic{{{chooser::inputs,32,1},
        {cudarl::nn::Activation::Tanh,cudarl::nn::Activation::Identity}}};
    struct Decision {
        float logp[chooser::actions]{};
        float value;
        int greedy;
    };
    Decision observe(const chooser::World& w) {
        float input[chooser::inputs];
        chooser::observe(w, larger_pond, input, 3);
        actor.forward(input, 1);
        critic.forward(input, 1);
        Decision d{};
        cudarl::rl::log_softmax(actor.output(), chooser::actions, d.logp);
        d.greedy = cudarl::rl::sample_action(d.logp, chooser::actions, 0.f, true);
        d.value = critic.output()[0];
        if (!std::isfinite(d.value) || !std::all_of(std::begin(d.logp), std::end(d.logp), physics::finite_number))
            throw std::runtime_error("Nonfinite policy inference");
        return d;
    }
};

void branch(const chooser::World& saved, int phase, const char* kind, int forced, int holds, int seed,
            Policy& policy, const Models& m, Budget& budget, std::ostream& cases, std::ostream& trace) {
    auto w = saved;
    const int tick = saved.motor.navigation.motion.controls;
    const auto before = budget.accepted;
    const auto initial = policy.observe(w);
    const float initial_gap = float(tail::submerged_gap_m(larger_pond, w.motor.navigation.motion.body, w.pose));
    double discounted = 0, discount = 1;
    int release = -1, macros = 0;
    while (macros < horizon && !w.fault) {
        const auto d = policy.observe(w);
        int action = d.greedy;
        if (seed >= 0) {
            const float u = cudarl::rng::uniform(26112532u, macros, uint32_t((phase * 200 + tick) * 16 + seed));
            action = cudarl::rl::sample_action(d.logp, chooser::actions, u, false);
        }
        if (macros < holds) action = forced;
        if (macros == holds) release = action;
        const auto r = advance(w, action, m, budget);
        discounted += discount * r.reward;
        discount *= learning_signal::discount;
        const double gap = 1000. * tail::submerged_gap_m(larger_pond, w.motor.navigation.motion.body, w.pose);
        trace << phase << ',' << tick << ',' << kind << ',' << forced << ',' << seed << ',' << macros
              << ',' << d.greedy << ',' << action << ',' << d.value << ',' << r.reward << ','
              << (budget.accepted-before)/substeps_per_s << ',' << gap << ',' << w.fault << '\n';
        ++macros;
    }
    const double seconds = (budget.accepted-before)/substeps_per_s;
    const double gap = tail::submerged_gap_m(larger_pond, w.motor.navigation.motion.body, w.pose);
    const double speed = seconds ? (w.transport_m - saved.transport_m) / seconds : 0;
    const bool pass = !w.fault && macros == horizon && w.min_gap_m > 0 && speed >= .006 &&
        w.max_stall_s <= 8.14f && w.peak_tilt_deg <= 10 && w.peak_depth_error_m <= .001f && gap >= .015;
    float p_straight = 0, p_previous_rate = 0;
    for (int a=0; a<chooser::actions; ++a) {
        if (chooser::action_rate_deg_s(a)==0) p_straight += std::exp(initial.logp[a]);
        if (chooser::action_rate_deg_s(a)==chooser::action_rate_deg_s(saved.last_action))
            p_previous_rate += std::exp(initial.logp[a]);
    }
    cases << phase << ',' << tick << ',' << kind << ',' << forced << ',' << holds << ',' << seed
          << ',' << saved.last_action << ',' << initial.greedy << ',' << release << ',' << p_straight << ','
          << p_previous_rate << ',' << initial.value << ',' << discounted << ',' << (w.fault?0:discount*policy.observe(w).value)
          << ',' << seconds << ',' << initial_gap*1000 << ',' << gap*1000 << ',' << w.min_gap_m*1000
          << ',' << speed*1000 << ',' << w.max_stall_s << ',' << w.peak_tilt_deg << ',' << w.peak_depth_error_m*1000
          << ',' << w.fault << ',' << pass << '\n';
}

// The original score uses one reference posture. Also measure clear headings at
// each swimmer's own depth and posture without integrating a single physics step.
void map_geometry(const Models& m, const fs::path& output) {
    std::ofstream table(output/"geometry.csv");
    table.exceptions(std::ios::failbit|std::ios::badbit);
    table << std::setprecision(10) << "case,x_cell,z_cell,depth_mm,reference_cell,clear_headings,min_gap_mm,max_gap_mm\n";
    const auto reference = reachable_cells(m.starts.front());
    const int nx = int(std::ceil(larger_pond.water_radius_x_m / cell_size_m));
    const int nz = int(std::ceil(larger_pond.water_radius_z_m / cell_size_m));
    for (const auto& start : endurance) {
        const auto& archived = m.starts.at(start.archived_row);
        for (int x=-nx; x<=nx; ++x) for (int z=-nz; z<=nz; ++z) {
            int clear = 0;
            double low = 1, high = -1;
            for (int i=0; i<8; ++i) {
                legacy::World w{};
                legacy::initialize(w, archived, {x*cell_size_m,0,z*cell_size_m}, radians(45.f*i));
                if (w.fault) throw std::runtime_error("Invalid placement");
                const auto gap = tail::submerged_gap_m(larger_pond, w.navigation.motion.body, w.pose);
                low = std::min(low,gap);
                high = std::max(high,gap);
                clear += gap > 0;
            }
            const bool scored = reference.count({x,z}) != 0;
            if (clear || scored) table << start.name << ',' << x << ',' << z << ','
                << -1000*archived.legs[archived.active_leg].goal_m.y << ',' << scored << ',' << clear
                << ',' << low*1000 << ',' << high*1000 << '\n';
        }
    }
}

int run(const Models& models, const fs::path& output, const fs::path& full, const fs::path& captures,
        const fs::path& traces, Clock::time_point begin) {
    static_assert(chooser::actions == 15 && chooser::inputs == 39);
    if (models.distant_feedback != 3 || !fs::create_directory(output)) throw std::runtime_error("Require frozen curved actor and unused directory");
    Policy policy;
    policy.actor.params() = models.policy;
    policy.critic.params() = learning_signal::read_critic(full,models,4);
    map_geometry(models,output);
    Budget budget{begin+std::chrono::seconds(40)};
    std::ofstream cases(output/"forks.csv"), detail(output/"decisions.csv"), prefixes(output/"prefix.csv");
    for (auto* f : {&cases,&detail,&prefixes}) { f->exceptions(std::ios::failbit|std::ios::badbit); *f << std::setprecision(10); }
    cases << "phase,tick,branch,forced,hold_decisions,seed,previous_action,policy_action,release_action,p_straight,p_same_rate,V,discounted_reward,endpoint_bootstrap,elapsed_s,initial_gap_mm,final_gap_mm,min_gap_mm,mean_speed_mm_s,max_stall_s,tilt_deg,depth_error_mm,fault,pass\n";
    detail << "phase,tick,branch,forced,seed,macro,greedy,action,V,reward,elapsed_s,post_gap_mm,fault\n";
    prefixes << "phase,tick,COM_error_m,logit_error,action,state_checksum\n";
    int count = 0;
    for (int phase : {8,9}) {
        const std::string name = phase==8?"phase8-bank1-direction-1":"phase9-bank1-direction1";
        const auto raw = cudarl::checkpoint::read((captures/(name+"-40.bin")).u8string());
        cudarl::checkpoint::Reader reader{raw};
        if (reader.value<uint32_t>()!=1) throw std::runtime_error("Unexpected capture format");
        auto w=reader.value<chooser::World>();
        reader.finish();
        std::ifstream source(traces/(name+".csv"));
        std::map<int,std::vector<std::string>> archived;
        std::string line;
        while (std::getline(source,line)) {
            auto f=close_probe::fields(line);
            if (f.size()>=33 && f[12]=="decision") archived.emplace(std::stoi(f[0]),std::move(f));
        }
        for (int tick : {40,80,120,160}) {
            const auto& f=archived.at(tick);
            const auto d=policy.observe(w);
            const auto err=physics::norm(w.pose.com_m - V{std::stof(f[2]),std::stof(f[3]),std::stof(f[4])});
            float logit_error=0;
            for(int a=0; a<chooser::actions; ++a)
                logit_error=std::max(logit_error,std::abs(policy.actor.output()[a]-std::stof(f[17+a])));
            if(err>1.e-7f || logit_error>1.e-5f || d.greedy!=std::stoi(f[13]) ||
               w.motor.navigation.motion.controls!=tick || !chooser::ready(w))
                throw std::runtime_error("Archived decision prefix mismatch");
            prefixes << phase << ',' << tick << ',' << err << ',' << logit_error << ',' << d.greedy << ',' << close_probe::hash(w) << '\n';
            if (tick>=80) {
                const auto fork=[&](const char* label,int a,int holds,int seed) {
                    branch(w,phase,label,a,holds,seed,policy,models,budget,cases,detail);
                    ++count;
                };
                fork("policy",-1,0,-1);
                for(int a=0; a<chooser::actions; ++a) fork("one",a,1,-1);
                const int straight=(w.last_action/chooser::rate_count)*chooser::rate_count+chooser::rate_count/2;
                fork("straight_8s",straight,10,-1);
                if(tick==160) for(int seed=0; seed<stochastic_replicates; ++seed) fork("sampled",-1,0,seed);
            }
            if (tick<160 && advance(w,d.greedy,models,budget).fault) throw std::runtime_error("Prefix stopped before fork");
        }
    }
    if(count!=134) throw std::runtime_error("Missing turn-exit forks");
    std::ofstream summary(output/"run.json");
    summary.exceptions(std::ios::failbit|std::ios::badbit);
    summary << "{\"branches\":" << count << ",\"attempted_substeps\":" << budget.attempted
            << ",\"accepted_substeps\":" << budget.accepted << ",\"wall_s\":" << std::chrono::duration<double>(Clock::now()-begin).count()
            << ",\"optimizer_updates\":0}\n";
    return 0;
}
}
