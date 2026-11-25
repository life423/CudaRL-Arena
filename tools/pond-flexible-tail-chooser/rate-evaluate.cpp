#include "habitat_config.h"
#include "nn/mlp_cpu.h"
#include "rl/flexible_tail_rate_chooser.h"
#include "rl/flexible_tail_chooser_checkpoint.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace tail = cudarl::flexible_tail;
namespace chooser = tail::rate_chooser;
namespace legacy = tail::chooser;
namespace physics = cudarl::physics;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using physics::V;
using Cell = std::pair<int, int>;

constexpr int diagnostic_seconds = 120;
constexpr int endurance_seconds = 300;
constexpr int controls_per_second = 50;
constexpr int samples_per_second = controls_per_second * tail::turn_substeps_per_control;
constexpr int motion_window_controls = controls_per_second / 2;
constexpr int runtime_limit_s = 30;
constexpr float cell_size_m = .05f;
constexpr float visit_displacement_m = .006f;
constexpr float speed_floor_m_s = .006f;
constexpr float shallow_gap_m = .020f;

static_assert(tail::motor_control_s == .02f);
static_assert(samples_per_second == 800);

struct Start {
    const char* name;
    size_t archived_row;
    float x_radius, z_radius;
    float heading_deg;
    bool exact_godot = false;
};

// Normalized starts use the current water radii and archived pose/phase.
// Heading is absolute world yaw from +X. Each row starts one fresh case.
constexpr std::array<Start, 12> diagnostic{{
    {"godot_rest", 0, 0, 0, 0, true},
    {"center_west", 9, -.05f, .01f, 174.f},
    {"center_north", 1, .03f, -.03f, 90.f},
    {"center_south", 8, -.02f, -.07f, -90.f},
    {"east_shelf_tangent", 10, .43f, .09f, 65.f},
    {"west_shelf_tangent", 11, -.41f, -.13f, -113.f},
    {"north_shelf_tangent", 5, .10f, -.42f, 154.f},
    {"south_shelf_tangent", 4, -.07f, .44f, -19.f},
    {"northeast_to_center", 3, .28f, -.27f, -137.f},
    {"southwest_to_center", 2, -.27f, .29f, 45.f},
    {"northwest_diagonal", 8, -.25f, -.24f, 123.f},
    {"southeast_diagonal", 9, .24f, .25f, -57.f}
}};
constexpr std::array<Start, 4> endurance{{
    {"endurance_godot_rest", 0, 0, 0, 0, true},
    {"endurance_northeast", 11, .20f, -.36f, 23.f},
    {"endurance_west", 6, -.35f, -.06f, 128.f},
    {"endurance_southwest", 7, -.16f, .29f, -78.f}
}};
constexpr int substep_cap =
    (diagnostic.size() * diagnostic_seconds + endurance.size() * endurance_seconds) * samples_per_second;
static_assert(substep_cap == 2112000);

struct Models {
    std::vector<float> gait, packed_motor, policy;
    tail::steering::Model motor;
    tail::steering::Config config;
    bool old_loader_rejected = false;
    std::vector<tail::NavigationWorld> starts;



};

struct Case {
    Start start;
    int horizon_s;
    int attempted = 0, accepted = 0, fault = 0;
    int decisions = 0, left = 0, right = 0, keep = 0;
    int repeats = 0, perimeter_controls = 0, complete_controls = 0, measured_windows = 0;
    int current_stall_controls = 0, max_stall_controls = 0;
    double path_m = 0, min_gap_m = std::numeric_limits<double>::infinity();
    double requested_speed_sum_m_s = 0, actual_speed_sum_m_s = 0, perimeter_angle_rad = 0;
    int speed_samples = 0;
    float max_tilt_deg = 0, max_depth_error_m = 0;
    std::set<Cell> visited;
    std::set<Cell> outside_reference;
    std::string outcome;
    bool pass = false;
};

float radians(float deg) { return deg * (tail::pi / 180.f); }

V horizontal(V position) {
    position.y = 0.f;
    return position;
}

Cell cell_at(V position) {
    return {static_cast<int>(std::floor(position.x / cell_size_m + .5f)),
        static_cast<int>(std::floor(position.z / cell_size_m + .5f))};
}

V horizontal_place(const Start& start, const tail::NavigationWorld& archived) {
    if (start.exact_godot) return horizontal(tail::snapshot(archived.motion.body).com_m);
    return {start.x_radius * float(larger_pond.water_radius_x_m), 0.f,
        start.z_radius * float(larger_pond.water_radius_z_m)};
}

float yaw_offset(const Start& start, const tail::NavigationWorld& archived) {
    if (start.exact_godot) return 0.f;
    const V aim = physics::axis_rotation({0.f, 1.f, 0.f}, radians(start.heading_deg)).rotate({1.f, 0.f, 0.f});
    const V incoming = archived.legs[archived.active_leg].heading;
    return std::atan2(physics::cross(incoming, aim).y, physics::dot(incoming, aim));
}

std::set<Cell> reachable_cells(const tail::NavigationWorld& reference) {
    const int radius_x = static_cast<int>(std::ceil(larger_pond.water_radius_x_m / cell_size_m));
    const int radius_z = static_cast<int>(std::ceil(larger_pond.water_radius_z_m / cell_size_m));
    std::set<Cell> reachable;
    for (int x = -radius_x; x <= radius_x; ++x) {
        for (int z = -radius_z; z <= radius_z; ++z) {
            for (int direction = 0; direction < 8; ++direction) {
                legacy::World state{};
                legacy::initialize(state, reference, {x * cell_size_m, 0.f, z * cell_size_m},
                    radians(45.f * direction));
                if (tail::submerged_gap_m(larger_pond, state.navigation.motion.body, state.pose) > 0.) {
                    reachable.emplace(x, z);
                    break;
                }
            }
        }
    }
    if (reachable.empty()) throw std::runtime_error("No certifiable upright coverage cells");
    return reachable;
}

void write_number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
}

struct VisitTracker {
    Cell cell{};
    V entry{};
    int entry_control = 0;
    bool credited = false;

    void record(Case& row, const std::set<Cell>& reachable, V com, int control) {
        const auto current = cell_at(com);
        if (current != cell) {
            cell = current;
            entry = com;
            entry_control = control;
            credited = false;
        }
        if (credited || control - entry_control < controls_per_second ||
            physics::norm(horizontal(com - entry)) < visit_displacement_m) return;
        credited = true;
        if (reachable.count(cell)) {
            if (!row.visited.insert(cell).second) ++row.repeats;
        } else {
            row.outside_reference.insert(cell);
        }
    }
};

Case run_case(Start start, int horizon_s, int cell_floor, const Models& models,
              const std::set<Cell>& reachable, const fs::path& output, Clock::time_point deadline) {
    Case row{start, horizon_s};
    std::ostringstream trace;
    trace << std::setprecision(9)
          << "control,time_s,x_m,y_m,z_m,heading_x,heading_z,rolling_speed_mm_s,clearance_bound_mm,"
          << "tilt_deg,depth_error_mm,unique_cells,event,action,requested_speed_mm_s,actual_COM_speed_mm_s,"
          << "requested_course_rate_deg_s";
    for (int action = 0; action < chooser::actions; ++action) trace << ",logit" << action;
    trace << ",fault\n";
    cudarl::nn::CpuMlp<float> actor(chooser::actor_spec());
    actor.params() = models.policy;
    const auto& archived = models.starts.at(start.archived_row);
    chooser::World world{};
    chooser::initialize(world, archived, horizontal_place(start, archived), yaw_offset(start, archived),
        models.motor.kind, models.config);
    const float depth_y = archived.legs[archived.active_leg].goal_m.y;
    const auto speed = [&]() {
        return physics::norm(tail::turn_course(world.motor.navigation.motion, world.pose.com_m,
            tail::steering_history_controls)) / (tail::steering_history_controls * tail::motor_control_s);
    };
    const auto log = [&](const char* event, int action = -1, const float* logits = nullptr) {
        const auto heading = world.motor.navigation.motion.request.heading;
        const auto com = world.pose.com_m;
        trace << row.complete_controls << ',' << double(row.accepted) / samples_per_second << ','
              << com.x << ',' << com.y << ',' << com.z << ',' << heading.x << ',' << heading.z << ',';
        write_number(trace, 1000. * speed());
        trace << ',';
        write_number(trace, 1000. * tail::submerged_gap_m(larger_pond, world.motor.navigation.motion.body, world.pose));
        trace << ',' << world.pose.tilt_deg << ',' << 1000.f * std::abs(depth_y - com.y) << ','
              << row.visited.size() << ',' << event << ',';
        if (action >= 0) trace << action;
        trace << ',' << 1000.f * world.config.target_speed_m_s << ','
            << 1000.f * physics::norm(world.pose.com_velocity_m_s) << ','
            << world.config.desired_course_rate_rad_s * (180.f / tail::pi);
        for (int i = 0; i < chooser::actions; ++i) {
            trace << ',';
            if (logits) write_number(trace, logits[i]);
        }
        trace << ',' << row.fault << '\n';
    };
    row.min_gap_m = tail::submerged_gap_m(larger_pond, world.motor.navigation.motion.body, world.pose);
    row.max_tilt_deg = world.pose.tilt_deg;
    row.max_depth_error_m = std::abs(depth_y - world.pose.com_m.y);
    if (!(row.min_gap_m > 0.)) {
        row.fault = 2;
        row.outcome = "initial_clearance";
        log("stop");
        std::ofstream file(output / "traces" / (std::string(start.name) + ".csv")); file << trace.str();
        return row;
    }
    log("start");

    constexpr int warmup_controls = 6 * controls_per_second;
    const int limit_control = horizon_s * controls_per_second;
    VisitTracker coverage{cell_at(world.pose.com_m), world.pose.com_m};
    V motion_window_start = world.pose.com_m;
    int last_motion_window_control = 0;
    float last_bearing = std::atan2(world.pose.com_m.z, world.pose.com_m.x);
    const auto historical_com = [&](int control) {
        const auto& motion = world.motor.navigation.motion;
        if(control == motion.controls) return world.pose.com_m;
        const int back = motion.controls - control;
        if(back < 1 || back > tail::turn_history_controls) throw std::runtime_error("Missing exact COM history endpoint");
        return motion.history[(motion.history_next + tail::turn_history_controls - back) % tail::turn_history_controls];
    };
    while (row.complete_controls < limit_control) {
        if (Clock::now() >= deadline) {
            row.outcome = "runtime_limit";
            break;
        }
        if (chooser::ready(world)) {
            float input[chooser::inputs];
            chooser::observe(world, larger_pond, input);
            actor.forward(input, 1);
            const float* logits = actor.output();
            if (!std::all_of(logits, logits + chooser::actions, physics::finite_number)) {
                row.fault = 3;
                row.outcome = "invalid_actor_output";
                break;
            }
            const int choice = static_cast<int>(std::max_element(logits, logits + chooser::actions) - logits);
            if (!chooser::accept(world, choice)) {
                row.fault = world.fault ? world.fault : 6;
                row.outcome = "invalid_action";
                break;
            }
            ++row.decisions;
            if (chooser::action_rate_deg_s(choice) == 0) ++row.keep;
            if (chooser::action_rate_deg_s(choice) > 0) ++row.left;
            if (chooser::action_rate_deg_s(choice) < 0) ++row.right;
            log("decision", choice, logits);
        }
        const int previous_control = row.complete_controls;
        const int requested_controls = std::min(4, limit_control - row.complete_controls);
        const auto step = chooser::advance(world, models.gait.data(), models.packed_motor.data(), larger_pond, requested_controls);
        row.attempted += step.attempted_substeps;
        row.accepted += step.accepted_substeps;
        row.min_gap_m = std::min(row.min_gap_m, double(world.min_gap_m));
        row.max_tilt_deg = std::max(row.max_tilt_deg, world.peak_tilt_deg);
        row.max_depth_error_m = std::max(row.max_depth_error_m, world.peak_depth_error_m);
        row.complete_controls += step.controls;
        if (step.fault != 0 || step.controls != requested_controls) {
            row.fault = step.fault ? step.fault : 6;
            row.outcome = "native_stop";
            break;
        }
        const auto com = world.pose.com_m;
        // Recover the exact original 20 ms coverage and half-second speed
        // endpoints from native history without altering 80 ms motor decisions.
        for(int control=previous_control+1;control<=row.complete_controls;++control)
            coverage.record(row, reachable, historical_com(control), control);
        if (tail::submerged_gap_m(larger_pond, world.motor.navigation.motion.body, world.pose) <= shallow_gap_m) {
            row.perimeter_controls += step.controls;
            const float bearing = std::atan2(com.z, com.x);
            row.perimeter_angle_rad += std::remainder(double(bearing-last_bearing), 2.*tail::pi);
        }
        last_bearing=std::atan2(com.z,com.x);
        while(last_motion_window_control+motion_window_controls<=row.complete_controls) {
            last_motion_window_control += motion_window_controls;
            const auto endpoint = historical_com(last_motion_window_control);
            if (last_motion_window_control > warmup_controls) {
                row.path_m += physics::norm(horizontal(endpoint - motion_window_start));
                ++row.measured_windows;
            }
            motion_window_start = endpoint;
        }
        if (row.complete_controls > warmup_controls) {
            for(int control=std::max(previous_control+1,warmup_controls+1);control<=row.complete_controls;++control){
                const auto then=historical_com(control),before=historical_com(control-motion_window_controls);
                const float rolling=physics::norm(horizontal(then-before)) / .5f;
                row.current_stall_controls=rolling<speed_floor_m_s?row.current_stall_controls+1:0;
                row.max_stall_controls=std::max(row.max_stall_controls,row.current_stall_controls);
            }
        }
        row.requested_speed_sum_m_s += world.config.target_speed_m_s;
        row.actual_speed_sum_m_s += physics::norm(world.pose.com_velocity_m_s);
        ++row.speed_samples;
        log("macro");
    }
    const bool completed = row.complete_controls == limit_control && row.accepted == horizon_s * samples_per_second;
    const double measured_time_s = double(row.measured_windows) * motion_window_controls / controls_per_second;
    const double mean_speed = measured_time_s > 0 ? row.path_m / measured_time_s : 0;
    row.pass = completed && row.fault == 0 && row.min_gap_m > 0 && row.max_tilt_deg <= 10.f &&
        row.max_depth_error_m <= .001f && mean_speed >= speed_floor_m_s &&
        static_cast<int>(row.visited.size()) >= cell_floor && row.left + row.right >= 2;
    if (row.outcome.empty()) row.outcome = row.pass ? "pass" : "metrics";
    log(row.pass ? "pass" : "stop");
    {std::ofstream file(output / "traces" / (std::string(start.name) + ".csv"));
        file.exceptions(std::ios::failbit | std::ios::badbit); file << trace.str();}
    return row;
}

template <size_t N>
std::vector<Case> run_parallel(const std::array<Start, N>& starts, int horizon_s, int cells,
                               const Models& models, const std::set<Cell>& reachable,
                               const fs::path& output, Clock::time_point deadline) {
    std::vector<std::future<Case>> futures;
    futures.reserve(N);
    for (auto start : starts) {
        futures.push_back(std::async(std::launch::async, [&, start] {
            return run_case(start, horizon_s, cells, models, reachable, output, deadline);
        }));
    }
    std::vector<Case> results;
    results.reserve(N);
    for (auto& future : futures) results.push_back(future.get());
    return results;
}

template <size_t N>
void write_starts(std::ostream& out, const char* stage, const std::array<Start, N>& starts,
                  const Models& models) {
    for (const auto& start : starts) {
        const auto& archived = models.starts.at(start.archived_row);
        const auto com = horizontal_place(start, archived);
        out << stage << ',' << start.name << ',' << start.archived_row << ','
            << com.x << ',' << tail::snapshot(archived.motion.body).com_m.y << ',' << com.z << ','
            << yaw_offset(start, archived) << ',' << start.exact_godot << '\n';
    }
}

int evaluate(const Models& models, const fs::path& output, Clock::time_point begin) {
    const auto deadline = begin + std::chrono::seconds(runtime_limit_s);
    if (models.starts.size() != 12 || !fs::create_directory(output))
        throw std::runtime_error("Expected the 12 archived straight starts and a new output directory");
    fs::create_directory(output / "traces");
    {
        chooser::World sensory{};const auto& archived=models.starts.front();
        chooser::initialize(sensory,archived,horizontal_place(diagnostic.front(),archived),0.f,models.motor.kind,models.config);
        const auto before=sensory.motor.navigation.motion.body;
        float observation[chooser::inputs];chooser::observe(sensory,larger_pond,observation);
        if(std::memcmp(&before,&sensory.motor.navigation.motion.body,sizeof(before)))throw std::runtime_error("Sensory query changed native body");
        for(float value:observation)if(!physics::finite_number(value))throw std::runtime_error("Nonfinite chooser sensory input");
        cudarl::nn::CpuMlp<float> actor(chooser::actor_spec());actor.params()=models.policy;actor.forward(observation,1);
        for(int action=0;action<chooser::actions;++action)if(!physics::finite_number(actor.output()[action]))throw std::runtime_error("Invalid nine-action chooser inference");
        std::ofstream metadata(output/"metadata-check.json");
        metadata<<"{\"pass\":true,\"physics_steps\":0,\"optimizer_updates\":0,\"sensory_did_not_change_body\":true,\"old_chooser_loader_rejected\":"
            <<(models.old_loader_rejected?"true":"false")<<",\"chooser_schema\":"<<chooser::schema<<",\"chooser_inputs\":"<<chooser::inputs
            <<",\"chooser_actions\":"<<chooser::actions<<",\"decision_controls\":"<<chooser::decision_controls
            <<",\"motor_kind\":"<<static_cast<uint32_t>(models.motor.kind)<<",\"motor_feature_schema\":"<<tail::steering::feature_schema(models.motor.kind)
            <<",\"motor_material\":"<<static_cast<uint32_t>(models.motor.material)<<",\"material\":"<<static_cast<uint32_t>(models.config.material)
            <<",\"bank_feature_schema\":"<<tail::bank::schema<<",\"transport_time_is_constant_shape_estimate\":true}\n";
    }
    const auto cells = reachable_cells(models.starts.front());
    std::ofstream starts(output / "starts.csv"), cases(output / "cases.csv"),
        reach(output / "coverage.csv"), summary(output / "summary.json");
    for (auto* file : {&starts, &cases, &reach, &summary}) {
        file->exceptions(std::ios::failbit | std::ios::badbit);
        *file << std::setprecision(9);
    }
    starts << "suite,case,archived_row,com_x_m,com_y_m,com_z_m,relative_yaw_rad,exact_godot_start\n";
    write_starts(starts, "diagnostic", diagnostic, models);
    write_starts(starts, "endurance", endurance, models);
    starts.flush();
    cases << "suite,case,outcome,fault,horizon_s,simulated_s,attempted_substeps,accepted_substeps,"
          << "min_clearance_mm,peak_tilt_deg,peak_depth_error_mm,mean_post_launch_speed_mm_s,"
          << "measured_path_mm,unique_cells,qualifying_revisits,outside_reference_cells,"
          << "perimeter_time_s,max_stall_s,decisions,keep,left,right,pass,mean_requested_speed_mm_s,"
          << "mean_instant_COM_speed_mm_s,perimeter_net_revolutions\n";

    int attempted = 0, accepted = 0, clearance_stops = 0, other_faults = 0;
    const auto record = [&](const char* suite, const std::vector<Case>& results, std::set<Cell>& visited) {
        int passes = 0;
        for (const auto& row : results) {
            const double moving_seconds = double(row.measured_windows) * motion_window_controls / controls_per_second;
            cases << suite << ',' << row.start.name << ',' << row.outcome << ',' << row.fault << ','
                  << row.horizon_s << ',' << double(row.accepted) / samples_per_second << ','
                  << row.attempted << ',' << row.accepted << ',';
            write_number(cases, 1000. * row.min_gap_m);
            cases << ',' << row.max_tilt_deg << ',' << 1000.f * row.max_depth_error_m << ',';
            if (moving_seconds > 0) cases << 1000. * row.path_m / moving_seconds;
            cases << ',' << 1000. * row.path_m << ',' << row.visited.size() << ',' << row.repeats << ','
                  << row.outside_reference.size() << ',' << double(row.perimeter_controls) / controls_per_second
                  << ',' << double(row.max_stall_controls) / controls_per_second << ',' << row.decisions
                  << ',' << row.keep << ',' << row.left << ',' << row.right << ',' << row.pass
                  << ',' << (row.speed_samples?1000.*row.requested_speed_sum_m_s/row.speed_samples:0)
                  << ',' << (row.speed_samples?1000.*row.actual_speed_sum_m_s/row.speed_samples:0)
                  << ',' << std::abs(row.perimeter_angle_rad)/(2.*tail::pi) << '\n';
            attempted += row.attempted;
            accepted += row.accepted;
            if (row.fault == 2) ++clearance_stops;
            if (row.fault != 0 && row.fault != 2) ++other_faults;
            if (row.pass) ++passes;
            visited.insert(row.visited.begin(), row.visited.end());
        }
        cases.flush();
        return passes;
    };

    std::set<Cell> visited, endurance_visited;
    const auto outcomes = run_parallel(diagnostic, diagnostic_seconds, 8, models, cells, output, deadline);
    const int diagnostic_passes = record("diagnostic", outcomes, visited);
    const bool diagnostic_passed =
        diagnostic_passes == static_cast<int>(diagnostic.size()) && 5 * visited.size() >= cells.size();
    int endurance_passes = 0, endurance_attempts = 0;
    if (diagnostic_passed) {
        const auto long_outcomes = run_parallel(endurance, endurance_seconds, 14, models, cells, output, deadline);
        endurance_attempts = static_cast<int>(long_outcomes.size());
        endurance_passes = record("endurance", long_outcomes, endurance_visited);
    }
    const bool passed = diagnostic_passed && endurance_passes == static_cast<int>(endurance.size());
    if (attempted > substep_cap) throw std::runtime_error("Held-out physical substep cap exceeded");
    reach << "center_x_m,center_z_m,diagnostic_visited,endurance_visited\n";
    for (const auto& cell : cells)
        reach << cell.first * cell_size_m << ',' << cell.second * cell_size_m
              << ',' << visited.count(cell) << ',' << endurance_visited.count(cell) << '\n';

    summary << "{\n"
            << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
            << "  \"steering_mode\": \"" << chooser::mode_name() << "\",\n"
            << "  \"steering_request_schema\": " << chooser::schema << ",\n"
            << "  \"motor_feedback_mode\": \"whole-beat-program\",\n"
            << "  \"motor_kind\": " << static_cast<uint32_t>(models.motor.kind) << ",\n"
            << "  \"material\": " << static_cast<uint32_t>(models.config.material) << ",\n"
            << "  \"diagnostic_passes\": " << diagnostic_passes << ",\n"
            << "  \"diagnostic_cases\": " << diagnostic.size() << ",\n"
            << "  \"endurance_passes\": " << endurance_passes << ",\n"
            << "  \"endurance_attempts\": " << endurance_attempts << ",\n"
            << "  \"clearance_stops\": " << clearance_stops << ",\n"
            << "  \"clearance_stops_are_guard_stops_not_contacts\": true,\n"
            << "  \"other_native_faults\": " << other_faults << ",\n"
            << "  \"reachable_50mm_cells_estimate\": " << cells.size() << ",\n"
            << "  \"diagnostic_visited_cells\": " << visited.size() << ",\n"
            << "  \"diagnostic_coverage_fraction\": " << double(visited.size()) / cells.size() << ",\n"
            << "  \"endurance_visited_cells\": " << endurance_visited.size() << ",\n"
            << "  \"attempted_substeps\": " << attempted << ",\n"
            << "  \"accepted_substeps\": " << accepted << ",\n"
            << "  \"substep_cap\": " << substep_cap << ",\n"
            << "  \"resets\": 0,\n"
            << "  \"CPU_parallel_workers_maximum\": 12,\n"
            << "  \"motor_action_controls\": 4,\n"
            << "  \"exact_original_20ms_coverage_and_halfsecond_motion_endpoints\": true,\n"
            << "  \"coverage_denominator_is_fixed_posture_estimate\": true,\n"
            << "  \"optimizer_updates\": 0,\n"
            << "  \"internal_elapsed_s\": " << std::chrono::duration<double>(Clock::now() - begin).count()
            << "\n}\n";
    std::cout << "diagnostic=" << diagnostic_passes << '/' << diagnostic.size()
              << " endurance=" << endurance_passes << '/' << endurance_attempts
              << " coverage=" << visited.size() << '/' << cells.size() << " clearance_stops=" << clearance_stops
              << " accepted_substeps=" << accepted << " attempted_substeps=" << attempted << '\n';
    return passed ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto begin = Clock::now();
        if (argc != 7) throw std::runtime_error("Usage: rate_chooser_eval GAIT MOTOR CHOOSER STARTS NEW_OUT MATERIAL");
        Models models;
        models.gait=tail::read_navigation_weights(argv[1],tail::NavigationMode::Gait);
        models.motor=tail::steering::read_model(argv[2]);
        models.policy=chooser::read_weights(argv[3]);
        models.starts=tail::read_navigation_worlds(argv[4]);
        models.config.study=tail::steering::Study::Rate;
        models.config.material=tail::steering::read_material(argv[6]);
        if(models.motor.kind<tail::steering::Kind::ProgramScalar||models.motor.material!=models.config.material)
            throw std::runtime_error("Chooser motor/material binding differs");
        models.packed_motor=models.motor.weights;
        models.packed_motor.insert(models.packed_motor.end(),models.motor.frozen_skip.begin(),models.motor.frozen_skip.end());
        tail::fast::validate_frozen_gait(models.gait);
        try { legacy::read_weights(argv[3]); } catch(const std::exception&) { models.old_loader_rejected=true; }
        if(!models.old_loader_rejected)throw std::runtime_error("Old chooser loader accepted incompatible rate model");
        return evaluate(models, argv[5], begin);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
