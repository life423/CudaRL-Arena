// Recorded CSV analysis only. No simulator, policy, CUDA or optimizer includes.
// bcrypt.h requires the Windows type declarations first.
// clang-format off
#include <windows.h>
#include <bcrypt.h>
// clang-format on
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace fs = std::filesystem;
namespace {
constexpr char half_second_study[] = "pond-motor-ppo-v5-half-second-2026-11-24";
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr int change_control = 40, window_controls = 80, registered_end = 400;
constexpr double seconds_per_control = .05, minimum_projection = .006, maximum_course_error = 5,
                 maximum_depth = .001, maximum_tilt = 10;
void require(bool ok, const std::string &why) {
    if (!ok)
        throw std::runtime_error(why);
}
std::string text(const fs::path &path) {
    std::ifstream in(path);
    require(bool(in), "Missing " + path.string());
    return {(std::istreambuf_iterator<char>(in)), {}};
}
std::string quote(const std::string &s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\')
            o += '\\';
        if (c == '\n')
            o += "\\n";
        else if (c == '\r')
            o += "\\r";
        else
            o += c;
    }
    return o + '"';
}
std::string scalar(const std::string &s, const std::string &key) {
    std::smatch m;
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(\"[^\"]*\"|true|false|[-+0-9.eE]+)");
    require(std::regex_search(s, m, pattern), "Missing result field " + key);
    const auto v = m[1].str();
    return v.front() == '"' ? v.substr(1, v.size() - 2) : v;
}
std::string sha256(const fs::path &path) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_size = 0, read = 0;
    require(BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0,
            "SHA provider failed");
    require(BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size),
                              sizeof(object_size), &read, 0) >= 0,
            "SHA object size failed");
    std::vector<unsigned char> object(object_size);
    std::array<unsigned char, 32> digest{};
    require(BCryptCreateHash(alg, &hash, object.data(), object_size, nullptr, 0, 0) >= 0,
            "SHA creation failed");
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "Missing hash input " + path.string());
    std::array<char, 65536> buffer{};
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto n = in.gcount();
        if (n > 0)
            require(BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()), static_cast<ULONG>(n), 0) >=
                        0,
                    "SHA update failed");
    }
    require(in.eof() && BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) >= 0,
            "SHA finalization failed");
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char x : digest)
        out << std::setw(2) << unsigned(x);
    return out.str();
}
std::vector<std::string> split(const std::string &line) {
    std::vector<std::string> out;
    std::string value;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                value += '"';
                ++i;
            } else
                quoted = !quoted;
        } else if (c == ',' && !quoted) {
            out.push_back(value);
            value.clear();
        } else if (c != '\r')
            value += c;
    }
    require(!quoted, "Unclosed CSV string");
    out.push_back(value);
    return out;
}
struct Csv {
    std::ifstream in;
    std::map<std::string, size_t> columns;
    explicit Csv(const fs::path &path) : in(path) {
        require(bool(in), "Missing CSV " + path.string());
        std::string line;
        require(bool(std::getline(in, line)), "Empty CSV");
        const auto names = split(line);
        for (size_t i = 0; i < names.size(); ++i)
            require(columns.emplace(names[i], i).second, "Duplicate CSV column");
    }
    bool next(std::vector<std::string> &row) {
        std::string line;
        if (!std::getline(in, line))
            return false;
        row = split(line);
        require(row.size() == columns.size(), "CSV field count mismatch");
        return true;
    }
    const std::string &get(const std::vector<std::string> &row, const std::string &key) const {
        const auto found = columns.find(key);
        require(found != columns.end(), "Missing CSV column " + key);
        return row[found->second];
    }
    double number(const std::vector<std::string> &row, const std::string &key) const {
        const double v = std::stod(get(row, key));
        require(std::isfinite(v), "Nonfinite CSV metric");
        return v;
    }
};
double wrap(double degrees) {
    return std::atan2(std::sin(degrees * pi / 180), std::cos(degrees * pi / 180)) * 180 / pi;
}
double course(double x, double z) {
    return std::atan2(-z, x) * 180 / pi;
}
std::string number(double x) {
    std::ostringstream out;
    out << std::setprecision(17) << x;
    return out.str();
}
std::string optional(std::optional<double> value) {
    return value ? number(*value) : "";
}
struct Control {
    double vx = 0, vz = 0, depth = 0, tilt = 0;
    int samples = 0;
};
struct Turn {
    int id = 0, pose = 0, accepted = 0;
    double requested = 0;
    bool completed = false, passed = false, hard = false;
    std::string reason;
    double initial_x = 0, initial_z = 0, target_x = 0, target_z = 0;
    bool initial_set = false, target_set = false;
    double end_lane = 0, max_lane = 0, end_depth = 0, end_tilt = 0;
    double max_depth = 0, max_tilt = 0, min_gap = 0, registered_projection = 0, registered_error = 180;
    std::map<int, Control> controls;
};
struct Window {
    double vx = 0, vz = 0, course_deg = 0, residual = 0, projection = 0, horizontal = 0;
    int samples = 0;
    bool valid = false, qualified = false;
};
Window window(const Turn &t, int end, double prefix_depth, double prefix_tilt) {
    Window w;
    if (!t.target_set || end < change_control + window_controls)
        return w;
    for (int i = end - window_controls + 1; i <= end; ++i) {
        const auto c = t.controls.find(i);
        if (c == t.controls.end() || c->second.samples != 10)
            return w;
        w.vx += c->second.vx;
        w.vz += c->second.vz;
        w.samples += c->second.samples;
    }
    w.vx /= w.samples;
    w.vz /= w.samples;
    w.horizontal = std::hypot(w.vx, w.vz);
    w.projection = w.vx * t.target_x + w.vz * t.target_z;
    if (w.horizontal < 1e-9)
        return w;
    w.course_deg = course(w.vx, w.vz);
    w.residual = wrap(w.course_deg - course(t.target_x, t.target_z));
    w.valid = true;
    w.qualified = w.projection >= minimum_projection && std::abs(w.residual) <= maximum_course_error &&
                  prefix_depth <= maximum_depth && prefix_tilt <= maximum_tilt;
    return w;
}
struct Derived {
    std::optional<double> first_end, first_delay, first_error, first_projection, pre_course, actual_change,
        achieved_turn;
    Window last, registered;
    std::string classification, status;
    int registered_samples = 0;
};
Derived derive(const Turn &t) {
    Derived d;
    double pre_x = 0, pre_z = 0;
    int pre_n = 0;
    for (int i = 21; i <= 40; ++i) {
        const auto c = t.controls.find(i);
        if (c != t.controls.end()) {
            pre_x += c->second.vx;
            pre_z += c->second.vz;
            pre_n += c->second.samples;
        }
    }
    if (pre_n == 200 && std::hypot(pre_x, pre_z) > 1e-9)
        d.pre_course = course(pre_x, pre_z);
    double depth = 0, tilt = 0;
    for (const auto &pair : t.controls) {
        const int end = pair.first;
        const auto &c = pair.second;
        require(c.samples == 10, "Accepted control does not contain ten stored substeps");
        depth = std::max(depth, c.depth);
        tilt = std::max(tilt, c.tilt);
        const auto current = window(t, end, depth, tilt);
        if (current.qualified && !d.first_end) {
            d.first_end = end * seconds_per_control;
            d.first_delay = (end - change_control) * seconds_per_control;
            d.first_error = current.residual;
            d.first_projection = current.projection;
        }
        if (end == t.accepted)
            d.last = current;
        if (end > 320) {
            d.registered.vx += c.vx;
            d.registered.vz += c.vz;
            d.registered_samples += c.samples;
        }
    }
    if (d.last.valid) {
        d.achieved_turn = wrap(d.last.course_deg - course(t.initial_x, t.initial_z));
        if (d.pre_course)
            d.actual_change = wrap(d.last.course_deg - *d.pre_course);
        if (std::abs(d.last.residual) <= 5)
            d.classification = "within_course_tolerance";
        else if (*d.achieved_turn * t.requested < 0)
            d.classification = "wrong_direction";
        else if (d.last.residual * t.requested < 0)
            d.classification = "underturn";
        else
            d.classification = "overshoot";
    } else
        d.classification = "no_full_postchange_four_second_window";
    if (d.registered_samples > 0 && t.target_set) {
        d.registered.vx /= d.registered_samples;
        d.registered.vz /= d.registered_samples;
        d.registered.horizontal = std::hypot(d.registered.vx, d.registered.vz);
        d.registered.projection = d.registered.vx * t.target_x + d.registered.vz * t.target_z;
        d.registered.course_deg = course(d.registered.vx, d.registered.vz);
        d.registered.residual = wrap(d.registered.course_deg - course(t.target_x, t.target_z));
        d.registered.valid = d.registered.horizontal >= 1e-9;
    }
    if (t.completed) {
        require(d.registered_samples == 800, "Completed turn is missing its registered final window");
        const bool reconstructed = d.registered.projection >= .006 && std::abs(d.registered.residual) <= 5 &&
                                   depth <= .001 && tilt <= 10;
        require(reconstructed == t.passed,
                "Recorded final-case pass disagrees with reconstructed unchanged gate");
    }
    d.status = t.passed      ? "final_case_pass"
               : d.first_end ? "transient_qualified_window_only"
                             : "no_qualified_window";
    if (t.reason == "skipped_after_stop")
        d.status = "skipped_after_stop";
    return d;
}
struct Margins {
    double lane = 0, depth = 0, tilt = 0, hold_speed = 0, hold_anchor = 0, long_min = 1e30, long_max = 0;
};
struct Stage {
    std::string name, result_text;
    int update = 0;
    std::map<int, Turn> turns;
    Margins margins;
    std::vector<std::pair<std::string, std::string>> hashes;
};
Stage stage(const fs::path &root, const std::string &name) {
    Stage s;
    s.name = name;
    const auto directory = root / name;
    for (const char *file :
         {"result.json", "cases.csv", "controls.csv", "turn-substeps.csv", "long-windows.csv"})
        s.hashes.emplace_back((directory / file).generic_string(), sha256(directory / file));
    s.result_text = text(directory / "result.json");
    const auto protocol_id = scalar(s.result_text, "protocol_id");
    require((protocol_id == "pond-motor-ppo-v2-2026-11-23" ||
             protocol_id == "pond-motor-ppo-v3-course-2026-11-23" ||
             protocol_id == "pond-motor-ppo-v4-repeated-2026-11-24" || protocol_id == half_second_study) &&
                scalar(s.result_text, "preflight_only") == "false",
            "Wrong recorded assessment protocol");
    s.update = std::stoi(scalar(s.result_text, "update"));
    Csv cases(directory / "cases.csv");
    std::vector<std::string> row;
    while (cases.next(row)) {
        if (cases.get(row, "suite") != "turn")
            continue;
        Turn t;
        t.id = static_cast<int>(cases.number(row, "case"));
        t.pose = static_cast<int>(cases.number(row, "pose"));
        t.requested = cases.number(row, "turn_degrees");
        t.accepted = static_cast<int>(cases.number(row, "accepted_controls"));
        t.completed = cases.number(row, "completed") == 1;
        t.passed = cases.number(row, "passed") == 1;
        t.hard = cases.number(row, "hard_failure") == 1;
        t.reason = cases.get(row, "reason");
        t.max_depth = cases.number(row, "max_depth_m");
        t.max_tilt = cases.number(row, "max_tilt_deg");
        t.min_gap = cases.number(row, "minimum_body_gap_m");
        t.registered_projection = cases.number(row, "late_projected_speed_m_s");
        t.registered_error = cases.number(row, "late_course_error_deg");
        require(s.turns.emplace(t.id, t).second, "Duplicate turn case");
    }
    require(s.turns.size() == 16, "Assessment does not contain all sixteen predeclared turn cases");
    Csv controls(directory / "controls.csv");
    while (controls.next(row)) {
        const auto suite = controls.get(row, "suite");
        const double lane = controls.number(row, "lane_m"), depth = controls.number(row, "depth_m"),
                     tilt = controls.number(row, "tilt_deg");
        if (suite == "turn") {
            auto &t = s.turns.at(static_cast<int>(controls.number(row, "case")));
            const int control = static_cast<int>(controls.number(row, "control"));
            if (control == 1) {
                t.initial_x = controls.number(row, "heading_x");
                t.initial_z = controls.number(row, "heading_z");
                t.initial_set = true;
            }
            if (control == 41) {
                t.target_x = controls.number(row, "heading_x");
                t.target_z = controls.number(row, "heading_z");
                t.target_set = true;
            }
            t.max_lane = std::max(t.max_lane, lane);
            t.end_lane = lane;
            t.end_depth = depth;
            t.end_tilt = tilt;
        } else {
            s.margins.lane = std::max(s.margins.lane, lane);
            s.margins.depth = std::max(s.margins.depth, depth);
            s.margins.tilt = std::max(s.margins.tilt, tilt);
            if (suite.find("hold") != std::string::npos) {
                const double vx = controls.number(row, "COM_vx"), vy = controls.number(row, "COM_vy"),
                             vz = controls.number(row, "COM_vz");
                s.margins.hold_speed = std::max(s.margins.hold_speed, std::hypot(std::hypot(vx, vy), vz));
                s.margins.hold_anchor =
                    std::max(s.margins.hold_anchor, controls.number(row, "anchor_error_m"));
            }
        }
    }
    Csv samples(directory / "turn-substeps.csv");
    while (samples.next(row)) {
        auto &t = s.turns.at(static_cast<int>(samples.number(row, "case")));
        auto &c = t.controls[static_cast<int>(samples.number(row, "control"))];
        ++c.samples;
        c.vx += samples.number(row, "COM_vx");
        c.vz += samples.number(row, "COM_vz");
        c.depth = std::max(c.depth, samples.number(row, "depth_m"));
        c.tilt = std::max(c.tilt, samples.number(row, "tilt_deg"));
    }
    Csv windows(directory / "long-windows.csv");
    while (windows.next(row))
        if (windows.number(row, "end_s") >= 8) {
            const double speed = windows.number(row, "mean_m_s");
            s.margins.long_min = std::min(s.margins.long_min, speed);
            s.margins.long_max = std::max(s.margins.long_max, speed);
        }
    for (const auto &binding : s.hashes)
        require(sha256(fs::path(binding.first)) == binding.second,
                "Raw assessment changed while being analyzed");
    return s;
}
void pure_tests() {
    unsigned tests = 0;
    const auto check = [&](bool yes) {
        require(yes, "Pure recorded-analysis test failed");
        ++tests;
    };
    check(std::abs(wrap(359) + 1) < 1e-12);
    check(std::abs(course(0, -1) - 90) < 1e-12);
    Turn t;
    t.initial_x = t.target_x = 1;
    t.initial_set = t.target_set = true;
    t.accepted = 400;
    t.completed = t.passed = true;
    t.requested = 5;
    for (int i = 1; i <= 400; ++i)
        t.controls[i] = {.08, 0, 0, 0, 10};
    auto d = derive(t);
    check(d.first_end && std::abs(*d.first_end - 6) < 1e-12 && std::abs(*d.first_delay - 4) < 1e-12);
    check(d.status == "final_case_pass");
    t.passed = false;
    for (int i = 321; i <= 400; ++i)
        t.controls[i].vz = .08;
    d = derive(t);
    check(d.first_end && d.status == "transient_qualified_window_only");
    t.controls[50].depth = .0011;
    d = derive(t);
    check(!d.first_end);
    t.controls[50].depth = 0;
    t.completed = false;
    t.accepted = 349;
    t.controls.erase(t.controls.upper_bound(349), t.controls.end());
    d = derive(t);
    check(d.last.valid && d.registered_samples == 290);
    std::cout << "{\"passed\":true,\"assertions\":" << tests
              << ",\"physical_controls\":0,\"optimizer_updates\":0}\n";
}
void write_outputs(const std::vector<Stage> &stages, const fs::path &input, const fs::path &output) {
    fs::create_directories(output);
    std::ofstream cases(output / "turn-cases.csv"), summary(output / "checkpoint-summary.csv"),
        json(output / "analysis.json");
    cases << "stage,update,case,pose,requested_turn_deg,side,accepted_controls,completed,final_case_pass,"
             "reason,initial_requested_course_deg,requested_course_deg,pre_request_COM_course_deg,last_4s_"
             "end_s,last_4s_COM_course_deg,actual_COM_course_change_deg,achieved_turn_from_initial_request_"
             "deg,signed_course_error_deg,absolute_course_error_deg,classification,last_4s_projected_speed_m_"
             "s,last_4s_horizontal_speed_m_s,first_qualified_window_end_s,first_qualified_window_delay_s,"
             "first_qualified_window_error_deg,first_qualified_window_projected_speed_m_s,completion_status,"
             "registered_final_samples,registered_final_signed_error_deg,registered_final_projected_speed_m_"
             "s,max_depth_m,max_tilt_deg,final_depth_m,final_tilt_deg,maximum_lane_m,final_lane_m,minimum_"
             "body_gap_m\n";
    summary << "stage,update,rest_passed,short_swim_passed,long_swim_passed,required_long_keys_retained,turn_"
               "passed,right_negative_passed,left_positive_passed,qualified_window_completions,transient_"
               "window_only,max_retention_lane_m,max_retention_depth_m,max_retention_tilt_deg,max_rest_COM_"
               "speed_m_s,max_rest_anchor_error_m,long_poststartup_window_min_m_s,long_poststartup_window_"
               "max_m_s,attempted_controls,attempted_substeps,hard_failure\n";
    json << "{\n\"schema\":1,\"input_root\":"
         << quote(fs::relative(input, fs::current_path()).generic_string())
         << ",\"path_base\":\"repository root\",\"analysis_source_sha256\":"
         << quote(sha256(fs::path(__FILE__)))
         << ",\"resolution_s\":0.05,\"COM_definition\":\"Stored mass-weighted COM velocity; course "
            "atan2(-vz,vx). Each control averages its ten recorded5ms "
            "samples.\",\"completion_definition\":\"Earliest end of80complete post-change controls with mean "
            "projection>=0.006m/s and abs shortest signed course error<=5deg; maximum depth<=0.001m and "
            "tilt<=10deg through that instant. First possible delay4s.\",\"final_rule\":\"Authoritative "
            "cases.csv pass is unchanged; early qualifying windows remain transient when the final case "
            "fails.\",\"last_window_definition\":\"Last available full4s after the change, including "
            "early-stopped cases; not automatically the registered final16-20s "
            "window.\",\"actual_change_definition\":\"Last4s COM course minus mean COM course over "
            "controls21-40; achieved_turn_from_initial_request instead uses the original requested "
            "heading.\",\"direction_convention\":\"Positive course/request is left; negative is right. "
            "Final-angle classification does not claim unwrapped full revolutions.\",\"lane_note\":\"Turn "
            "lane is distance from the request line through the initial COM; it is descriptive, not a "
            "turning pass criterion.\",\"physical_controls\":0,\"optimizer_updates\":0,\"stages\":[\n";
    bool first_stage = true;
    for (const auto &s : stages) {
        int right = 0, left = 0, completions = 0, transients = 0;
        for (const auto &pair : s.turns) {
            const auto &t = pair.second;
            const auto d = derive(t);
            if (t.passed)
                (t.requested < 0 ? right : left)++;
            if (d.first_end)
                ++completions;
            if (d.status == "transient_qualified_window_only")
                ++transients;
            const auto opt = [&](bool valid, double value) { return valid ? number(value) : ""; };
            const std::vector<std::string> row{s.name,
                                               std::to_string(s.update),
                                               std::to_string(t.id),
                                               std::to_string(t.pose),
                                               number(t.requested),
                                               t.requested > 0 ? "left" : "right",
                                               std::to_string(t.accepted),
                                               t.completed ? "1" : "0",
                                               t.passed ? "1" : "0",
                                               t.reason,
                                               opt(t.initial_set, course(t.initial_x, t.initial_z)),
                                               opt(t.target_set, course(t.target_x, t.target_z)),
                                               optional(d.pre_course),
                                               opt(d.last.valid, t.accepted * .05),
                                               opt(d.last.valid, d.last.course_deg),
                                               optional(d.actual_change),
                                               optional(d.achieved_turn),
                                               opt(d.last.valid, d.last.residual),
                                               opt(d.last.valid, std::abs(d.last.residual)),
                                               d.classification,
                                               opt(d.last.valid, d.last.projection),
                                               opt(d.last.valid, d.last.horizontal),
                                               optional(d.first_end),
                                               optional(d.first_delay),
                                               optional(d.first_error),
                                               optional(d.first_projection),
                                               d.status,
                                               std::to_string(d.registered_samples),
                                               opt(d.registered.valid, d.registered.residual),
                                               opt(d.registered.valid, d.registered.projection),
                                               number(t.max_depth),
                                               number(t.max_tilt),
                                               number(t.end_depth),
                                               number(t.end_tilt),
                                               number(t.max_lane),
                                               number(t.end_lane),
                                               number(t.min_gap)};
            for (size_t i = 0; i < row.size(); ++i) {
                if (i)
                    cases << ',';
                cases << row[i];
            }
            cases << '\n';
        }
        const auto get = [&](const char *key) { return scalar(s.result_text, key); };
        summary << s.name << ',' << s.update << ',' << get("rest_cases_passed") << ','
                << get("short_swim_cases_passed") << ',' << get("long_swim_cases_passed") << ','
                << get("retained_parent_long_window_passes") << ',' << get("turn_cases_passed") << ','
                << right << ',' << left << ',' << completions << ',' << transients << ','
                << std::setprecision(17) << s.margins.lane << ',' << s.margins.depth << ',' << s.margins.tilt
                << ',' << s.margins.hold_speed << ',' << s.margins.hold_anchor << ',' << s.margins.long_min
                << ',' << s.margins.long_max << ',' << get("attempted_controls") << ','
                << get("attempted_substeps") << ',' << get("hard_failure") << '\n';
        if (!first_stage)
            json << ",\n";
        first_stage = false;
        json << "{\"stage\":" << quote(s.name) << ",\"update\":" << s.update
             << ",\"rest_passed\":" << get("rest_cases_passed")
             << ",\"short_passed\":" << get("short_swim_cases_passed")
             << ",\"long_passed\":" << get("long_swim_cases_passed")
             << ",\"required_long_keys_retained\":" << get("retained_parent_long_window_passes")
             << ",\"turn_passed\":" << get("turn_cases_passed") << ",\"right_passed\":" << right
             << ",\"left_passed\":" << left << ",\"qualified_window_completions\":" << completions
             << ",\"transient_window_only\":" << transients << ",\"raw_sha256\":[";
        for (size_t i = 0; i < s.hashes.size(); ++i) {
            if (i)
                json << ',';
            json << "{\"path\":"
                 << quote(fs::relative(fs::path(s.hashes[i].first), fs::current_path()).generic_string())
                 << ",\"sha256\":" << quote(s.hashes[i].second) << '}';
        }
        json << "]}";
    }
    cases.close();
    summary.close();
    require(bool(cases) && bool(summary), "Derived CSV write failed");
    json << "\n],\"derived_files\":[{\"path\":\"turn-cases.csv\",\"sha256\":"
         << quote(sha256(output / "turn-cases.csv")) << "},{\"path\":\"checkpoint-summary.csv\",\"sha256\":"
         << quote(sha256(output / "checkpoint-summary.csv")) << "}]}\n";
    require(bool(json), "Derived JSON write failed");
}
} // namespace
namespace {
uint64_t fnv(const uint8_t *data, size_t size) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}
struct Slice {
    size_t offset = 0, count = 0;
};
struct RecordedBatch {
    std::vector<uint8_t> bytes;
    uint32_t update = 0, steps = 0, stride = 0, version = 0;
    Slice tasks, done, truncated, attempted, substeps, reason, trace;
    template <class T> T at(size_t offset) const {
        require(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset,
                "Recorded batch truncated");
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    }
};
RecordedBatch recorded_batch(const fs::path &path) {
    RecordedBatch b;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(bool(in), "Missing batch");
    const auto length = in.tellg();
    require(length >= 24, "Truncated batch wrapper");
    b.bytes.resize(static_cast<size_t>(length));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(b.bytes.data()), length);
    require(bool(in) && b.at<uint64_t>(0) == 0x31504b434c524443ull &&
                b.at<uint64_t>(8) == b.bytes.size() - 24 &&
                b.at<uint64_t>(16) == fnv(b.bytes.data() + 24, b.bytes.size() - 24),
            "Batch wrapper checksum/length mismatch");
    size_t cursor = 24;
    const auto u32 = [&]() {
        const auto v = b.at<uint32_t>(cursor);
        cursor += 4;
        return v;
    };
    const auto array = [&](size_t width) {
        const auto n = b.at<uint64_t>(cursor);
        cursor += 8;
        require(n <= (b.bytes.size() - cursor) / width, "Corrupt batch array");
        const Slice result{cursor, static_cast<size_t>(n)};
        cursor += static_cast<size_t>(n) * width;
        return result;
    };
    require(u32() == 0x31425450, "Wrong recorded batch magic");
    b.version = u32();
    require(b.version == 1 || b.version == 2, "Unknown batch version");
    b.update = u32();
    b.steps = u32();
    cursor += 16;
    array(1);
    b.tasks = array(1);
    array(4);
    array(4);
    for (int i = 0; i < 9; ++i) {
        const auto a = array(4);
        if (i == 5)
            b.done = a;
        if (i == 6)
            b.truncated = a;
    }
    b.attempted = array(8);
    b.substeps = array(8);
    b.reason = array(4);
    b.stride = u32();
    b.trace = array(1);
    require(b.steps == 64 && b.attempted.count == 32768 && b.substeps.count == 32768 &&
                b.reason.count == 32768 && b.done.count == 32768 && b.truncated.count == 32768 &&
                b.trace.count == 32768 * b.stride && cursor == b.bytes.size(),
            "Recorded batch dimensions changed");
    require((b.version == 1 && b.stride == 104 && b.tasks.count == 512 * 184) ||
                (b.version == 2 && b.stride == 192 &&
                    (b.tasks.count == 512 * 272 || b.tasks.count == 512 * 360)),
            "Unknown recorded task/trace ABI");
    return b;
}
uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
uint32_t random_bits(uint32_t seed, uint32_t stream, uint32_t index) {
    return mix32(seed ^ mix32(stream ^ mix32(index)));
}
int original_angle(uint32_t env, uint32_t episode) {
    const auto draw = random_bits(0x504f4e44u, mix32(env ^ 0xe31f680bu), episode) % 10u;
    constexpr int angle[5] = {5, 15, 30, 60, 90};
    return (draw < 5 ? -1 : 1) * angle[draw % 5];
}
struct RecordedGoal {
    float x = 0, z = 0;
    bool known = false;
};
struct PracticeRequest {
    uint32_t env = 0, episode = 0, serial = 0, onset_update = 0, last_age = 0, streak = 0,
             first_success = 0, late_success = 0;
    int angle = 0;
    float hx = 0, hz = 0;
    bool heading_known = false, native_complete = false, timeout = false, cap = false,
         terminal = false, long_hold = false, long_end = false;
    double start_course = 0, final_course = 0, final_error = 0, final_speed = 0,
           final_projection = 0, success_error = 0, success_speed = 0;
    bool start_course_known = false;
    uint32_t eligible_ticks = 0, qualified_ticks = 0, max_streak = 0, course_fail = 0,
             speed_fail = 0, depth_fail = 0, tilt_fail = 0, terminal_reason = 0;
};
uint32_t practice_streak(uint32_t age, bool qualified, uint32_t previous) {
    return age > 100 && qualified ? previous + 1 : 0;
}
bool course_within_tolerance(float vx, float vz, float hx, float hz) {
    const float speed = std::sqrt(vx * vx + vz * vz);
    return std::isfinite(speed) && speed > 0 && (vx * hx + vz * hz) / speed >= .9961946980917455f;
}
bool practice_qualified(bool good_course, float projected_speed, float depth, float tilt) {
    return good_course && projected_speed >= .006f && depth <= .001f && tilt <= 10.f;
}
void training_pure_tests() {
    uint32_t streak = 0;
    for (uint32_t age = 1; age <= 119; ++age) {
        streak = practice_streak(age, true, streak);
        require(streak < 20, "Training completion before age120");
    }
    require(practice_streak(120, true, streak) == 20, "Earliest completion is not age120");
    require(practice_streak(120, false, streak) == 0, "Failed training tick did not reset streak");
    require(practice_qualified(course_within_tolerance(.008f, 0, 1, 0), .008f, .001f, 10), "Inclusive training limits rejected");
    require(!practice_qualified(true, .0059f, 0, 0), "Slow training tick accepted");
    require(!practice_qualified(course_within_tolerance(.008f, .008f, 1, 0), .008f, 0, 0),
            "Wrong course training tick accepted");
    require(!practice_qualified(true, .008f, .00101f, 0), "Depth training limit ignored");
    require(!practice_qualified(true, .008f, 0, 10.01f), "Tilt training limit ignored");
    std::cout << "Training streak/predicate pure checks passed; new physical "
                 "controls=0 optimizer=0\n";
}
double utc_seconds(const std::string &value) {
    std::tm time{};
    std::istringstream input(value.substr(0, 19));
    input >> std::get_time(&time, "%Y-%m-%dT%H:%M:%S");
    require(!input.fail(), "Malformed runner UTC");
    double fraction = 0;
    if (value.size() > 19 && value[19] == '.') {
        const auto end = value.find_first_not_of("0123456789", 20);
        fraction = std::stod("0" + value.substr(19, end - 19));
    }
    return static_cast<double>(_mkgmtime(&time)) + fraction;
}
void training_analysis(const fs::path &root, const fs::path &output, int maximum_update) {
    require(maximum_update > 0 && maximum_update <= 32, "Training analysis update cap");
    const auto protocol = text(root / "protocol.json");
    const auto id = scalar(protocol, "protocol_id");
    const bool half_second = id == half_second_study;
    const bool repeated = id == "pond-motor-ppo-v4-repeated-2026-11-24" || half_second;
    require(!half_second || maximum_update <= 8, "Half-second exercise ends at update 8");
    require(repeated || id == "pond-motor-ppo-v3-course-2026-11-23", "Unsupported training study");
    std::vector<fs::path> paths;
    std::vector<std::string> hashes;
    std::map<std::pair<uint32_t, uint32_t>, RecordedGoal> goals;
    for (int update = 1; update <= maximum_update; ++update) {
        const int last = ((update - 1) / 8 + 1) * 8;
        const auto directory = "updates-" + std::to_string(last - 7) + "-" + std::to_string(last);
        const auto path = root / directory / ("batch-" + std::to_string(update) + ".bin");
        const auto b = recorded_batch(path);
        require(b.update == static_cast<uint32_t>(update) && repeated == (b.version == 2) &&
                    half_second == (b.tasks.count == 512 * 360),
                "Study/batch source version mismatch");
        paths.push_back(path);
        hashes.push_back(sha256(path));
        if (!repeated)
            for (uint32_t env = 0; env < 512; ++env) {
                const size_t base = b.tasks.offset + env * 184;
                if (b.at<int32_t>(base + 80) != 2)
                    continue;
                const auto episode = b.at<uint32_t>(base + 72), control = b.at<uint32_t>(base + 76);
                if (control >= 40 && b.at<uint8_t>(base + 88) != 0)
                    goals[{env, episode}] = {b.at<float>(base + 24), b.at<float>(base + 32), true};
            }
    }
    using RequestKey = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::map<RequestKey, PracticeRequest> requests;
    std::array<std::pair<float, float>, 512> previous{};
    // Turn episodes settle for 40 ticks. By the first request, every slot
    // holds a position from this episode, including across rollout boundaries.
    std::array<std::array<std::pair<float,float>,10>,512> com_history{};
    std::array<std::array<uint64_t, 6>, 512> counters{};
    uint64_t accepted = 0, native_onsets = 0, native_successes = 0, native_timeouts = 0,
             native_caps = 0, native_terminals = 0, rule_mismatches = 0;
    for (size_t file_index = 0; file_index < paths.size(); ++file_index) {
        const auto b = recorded_batch(paths[file_index]);
        for (size_t row = 0; row < 32768; ++row) {
            require(b.at<uint64_t>(b.attempted.offset + row * 8) == 1 &&
                        b.at<uint64_t>(b.substeps.offset + row * 8) == 10,
                    "Incomplete/rejected physical batch row");
            ++accepted;
            const uint32_t env = static_cast<uint32_t>(row % 512);
            const size_t at = b.trace.offset + row * b.stride;
            const float vx = b.at<float>(at + 8), vz = b.at<float>(at + 16),
                        depth = b.at<float>(at + 20), tilt = b.at<float>(at + 24);
            const auto episode = b.at<uint32_t>(at + 40), control = b.at<uint32_t>(at + 44),
                       kind = b.at<uint32_t>(at + 48);
            float course_vx = vx, course_vz = vz;
            if (half_second && kind == 2) {
                const auto x = b.at<float>(at), z = b.at<float>(at + 4);
                auto& old = com_history[env][control % 10];
                if (control > 10) {
                    course_vx = (x - old.first) / .5f;
                    course_vz = (z - old.second) / .5f;
                }
                old = {x,z};
            }
            if (repeated) {
                const auto events = b.at<uint32_t>(at + 128);
                const uint32_t bits[6] = {1, 2, 4, 8, 16, 1};
                for (int i = 0; i < 6; ++i) {
                    const auto now = b.at<uint64_t>(at + 144 + i * 8);
                    const uint64_t expected_increment =
                        (events & bits[i]) && (i != 5 || b.at<uint32_t>(at + 140) != 0) ? 1 : 0;
                    require(now == counters[env][i] + expected_increment,
                            "Native cumulative counter/event discontinuity");
                    counters[env][i] = now;
                }
            }
            const auto prior = previous[env];
            previous[env] = {course_vx, course_vz};
            if (kind != 2 || control <= 40)
                continue;
            const uint32_t serial = repeated ? b.at<uint32_t>(at + 112) : 1,
                           age = repeated ? b.at<uint32_t>(at + 120) : control - 40;
            if (serial == 0 || age == 0)
                continue;
            const uint32_t events = repeated ? b.at<uint32_t>(at + 128) : 0;
            const auto key = RequestKey{env, episode, serial};
            auto found = requests.find(key);
            if (found == requests.end()) {
                require(age == 1, "Request start missing from analyzed complete prefix");
                PracticeRequest r;
                r.env = env;
                r.episode = episode;
                r.serial = serial;
                r.onset_update = b.update;
                if (repeated) {
                    require(events & 1u, "First repeated request row lacks native onset");
                    r.angle = static_cast<int>(
                        std::lround(b.at<float>(at + 108) * 180 / 3.14159265358979323846f));
                    require(!half_second || (std::abs(r.angle) >= 10 && std::abs(r.angle) <= 15),
                            "Training request outside the declared 10 to 15 degree range");
                } else
                    r.angle = original_angle(env, episode);
                r.start_course_known = std::hypot(prior.first, prior.second) >= .003;
                r.start_course = course(prior.first, prior.second);
                found = requests.emplace(key, r).first;
            }
            auto &r = found->second;
            require(age == r.last_age + 1, "Request age discontinuity/reset across rollout");
            r.last_age = age;
            if (repeated) {
                r.hx = b.at<float>(at + 100);
                r.hz = b.at<float>(at + 104);
                r.heading_known = true;
                r.long_hold = b.at<uint32_t>(at + 140) != 0;
            } else {
                const auto goal = goals.find({env, episode});
                if (goal != goals.end()) {
                    r.hx = goal->second.x;
                    r.hz = goal->second.z;
                    r.heading_known = true;
                }
            }
            const float speed = std::sqrt(vx * vx + vz * vz), projection = vx * r.hx + vz * r.hz;
            r.final_course = course(course_vx, course_vz);
            r.final_speed = speed;
            r.final_projection = projection;
            if (r.heading_known)
                r.final_error = wrap(r.final_course - course(r.hx, r.hz));
            require(r.heading_known || age <= 100,
                    "Eligible v3 request lacks exact saved post-onset world heading");
            const bool good_course = course_within_tolerance(course_vx, course_vz, r.hx, r.hz);
            const bool qualified = r.heading_known && practice_qualified(good_course, projection, depth, tilt);
            r.streak = practice_streak(age, qualified, r.streak);
            if (age > 100 && age <= 400) {
                ++r.eligible_ticks;
                r.qualified_ticks += qualified ? 1 : 0;
                r.max_streak = std::max(r.max_streak, r.streak);
                r.course_fail += !good_course ? 1 : 0;
                r.speed_fail += !(projection >= .006f) ? 1 : 0;
                r.depth_fail += !(depth <= .001f) ? 1 : 0;
                r.tilt_fail += !(tilt <= 10.f) ? 1 : 0;
            }
            if (r.streak >= 20 && r.late_success == 0) {
                r.late_success = age;
                if (age <= 400) {
                    r.first_success = age;
                    r.success_error = r.final_error;
                    r.success_speed = speed;
                }
            }
            if (repeated) {
                if (events & 1u) {
                    require(age == 1, "Native onset after request age1");
                    ++native_onsets;
                }
                if (events & 2u) {
                    require(!r.native_complete && age >= 120 && age <= 400,
                            "Duplicate/too-early/late native completion");
                    r.native_complete = true;
                    ++native_successes;
                    if (r.first_success != age)
                        ++rule_mismatches;
                }
                if (events & 4u) {
                    require(age == 400 && !r.native_complete,
                            "Native timeout violates400tick/one-count rule");
                    r.timeout = true;
                    ++native_timeouts;
                }
                if (events & 8u) {
                    r.cap = true;
                    ++native_caps;
                }
                if (events & 16u) {
                    r.terminal = true;
                    r.terminal_reason =
                        static_cast<uint32_t>(b.at<int32_t>(b.reason.offset + row * 4));
                    ++native_terminals;
                }
                if (events & 32u)
                    r.long_end = true;
            } else if (b.at<float>(b.done.offset + row * 4) == 1) {
                if (b.at<float>(b.truncated.offset + row * 4) == 1)
                    r.cap = true;
                else
                    r.terminal = true;
                r.terminal_reason = static_cast<uint32_t>(b.at<int32_t>(b.reason.offset + row * 4));
            }
        }
        require(sha256(paths[file_index]) == hashes[file_index],
                "Training batch changed during read-only analysis");
    }
    uint64_t comparable = 0, cap = 0, terminal = 0, pending = 0, long_assigned = 0,
             long_pending = 0, repeats = 0, late_only = 0, depth_terminals = 0, tilt_terminals = 0;
    // Starts, completions, eligible, qualified, course/speed/depth/tilt failures,
    // max streak.
    std::map<int, std::array<uint64_t, 9>> angles;
    for (const auto &entry : requests) {
        const auto &r = entry.second;
        ++angles[r.angle][0];
        if (r.first_success) {
            ++comparable;
            ++angles[r.angle][1];
        }
        if (r.serial > 1)
            ++repeats;
        if (r.cap)
            ++cap;
        if (r.terminal)
            ++terminal;
        depth_terminals += r.terminal && r.terminal_reason == 1 ? 1 : 0;
        tilt_terminals += r.terminal && r.terminal_reason == 2 ? 1 : 0;
        auto &a = angles[r.angle];
        a[2] += r.eligible_ticks;
        a[3] += r.qualified_ticks;
        a[4] += r.course_fail;
        a[5] += r.speed_fail;
        a[6] += r.depth_fail;
        a[7] += r.tilt_fail;
        a[8] = std::max(a[8], uint64_t(r.max_streak));
        if (!r.native_complete && !r.timeout && !r.cap && !r.terminal)
            ++pending;
        if (r.long_hold) {
            ++long_assigned;
            if (r.native_complete && !r.long_end && !r.cap && !r.terminal)
                ++long_pending;
        }
        if (r.late_success > 400)
            ++late_only;
    }
    if (repeated) {
        std::array<uint64_t, 6> final_counters{};
        for (const auto &world : counters)
            for (int i = 0; i < 6; ++i)
                final_counters[i] += world[i];
        require(final_counters[0] == native_onsets && final_counters[1] == native_successes &&
                    final_counters[2] == native_timeouts && final_counters[3] == native_caps &&
                    final_counters[4] == native_terminals && final_counters[5] == long_assigned,
                "Final cumulative native counters differ from reconstructed event "
                "totals");
        require(native_onsets == requests.size() &&
                    native_successes + native_timeouts + native_caps + native_terminals + pending ==
                        requests.size(),
                "Native request outcomes do not partition starts");
        require(rule_mismatches == 0 && native_successes == comparable,
                "Native/comparable completion-rule disagreement");
    }
    double training_seconds = 0, first_start = 1e30, last_end = 0;
    for (const char *folder : {"runner", "runner-recovered-baseline"}) {
        const auto directory = root / folder;
        if (fs::exists(directory / "started.json"))
            first_start = std::min(
                first_start, utc_seconds(scalar(text(directory / "started.json"), "started_utc")));
        for (int update = 8; update <= maximum_update; update += 8) {
            const auto command = directory / ("train-" + std::to_string(update) + ".command.json");
            if (fs::exists(command))
                training_seconds += std::stod(scalar(text(command), "process_wall_ms")) / 1000;
        }
        const auto evaluation =
            directory / ("eval" + std::to_string(maximum_update) + ".command.json");
        if (fs::exists(evaluation))
            last_end = std::max(last_end, utc_seconds(scalar(text(evaluation), "ended_utc")));
        // Include final receipt validation/bookkeeping after the last child process.
        if (fs::exists(directory / "result.json")) {
            const auto result = text(directory / "result.json");
            if (scalar(result, "last_passed_training_chunk_update") == std::to_string(maximum_update) &&
                scalar(result, "status") == "COMPLETED_ALLOCATION") {
                const auto start = utc_seconds(scalar(text(directory / "started.json"), "started_utc"));
                last_end = std::max(last_end, start + std::stod(scalar(result, "total_host_wall_ms")) / 1000);
            }
        }
    }
    require(training_seconds > 0 && last_end > first_start,
            "Checkpoint native timing evidence missing");
    const double elapsed = last_end - first_start;
    fs::create_directories(output);
    const std::string prefix = half_second ? "v5" : (repeated ? "v4" : "v3");
    std::ofstream csv(output / (prefix + "-training-requests.csv"));
    csv << std::setprecision(17);
    csv << "environment,episode,request_serial,angle_deg,onset_update,last_age_"
           "ticks,heading_exact,"
           "comparable_success_by400,completion_age_ticks,completion_seconds,"
           "completion_course_error_deg,"
           "completion_COM_speed_m_s,native_success,native_timeout,episode_cap_"
           "interruption,other_terminal_"
           "interruption,end_pending,long_assigned,long_hold_finished,start_COM_"
           "course_deg,final_COM_course_"
           "deg,final_course_error_deg,final_COM_speed_m_s,final_projected_speed_"
           "m_s,eligible_ticks,"
           "qualified_ticks,max_qualifying_streak,course_fail_ticks,speed_fail_"
           "ticks,depth_fail_ticks,"
           "tilt_fail_ticks,terminal_reason\n";
    for (const auto &entry : requests) {
        const auto &r = entry.second;
        const bool open = !r.native_complete && !r.timeout && !r.cap && !r.terminal;
        csv << r.env << ',' << r.episode << ',' << r.serial << ',' << r.angle << ','
            << r.onset_update << ',' << r.last_age << ',' << r.heading_known << ','
            << (r.first_success != 0) << ',';
        if (r.first_success)
            csv << r.first_success;
        csv << ',';
        if (r.first_success)
            csv << r.first_success * .05;
        csv << ',';
        if (r.first_success)
            csv << std::setprecision(17) << r.success_error;
        csv << ',';
        if (r.first_success)
            csv << r.success_speed;
        csv << ',' << r.native_complete << ',' << r.timeout << ',' << r.cap << ',' << r.terminal
            << ',' << open << ',' << r.long_hold << ',' << r.long_end << ',';
        if (r.start_course_known)
            csv << r.start_course;
        csv << ',' << r.final_course << ',';
        if (r.heading_known)
            csv << r.final_error;
        csv << ',' << r.final_speed << ',' << r.final_projection << ',' << r.eligible_ticks << ','
            << r.qualified_ticks << ',' << r.max_streak << ',' << r.course_fail << ','
            << r.speed_fail << ',' << r.depth_fail << ',' << r.tilt_fail << ',' << r.terminal_reason
            << '\n';
    }
    csv.close();
    require(bool(csv), "Training request CSV write failed");
    std::ofstream out(output / (prefix + "-training-summary.json"));
    out << std::setprecision(17) << "{\"schema\":1,\"protocol_id\":" << quote(id)
        << ",\"through_update\":" << maximum_update << ",\"accepted_controls\":" << accepted
        << ",\"training_definition\":\"After100 post-onset ticks, "
           "require20consecutive per-control samples: "
           "COMcourse<=5deg,projection>=6mm/s,depth<=1mm,tilt<=10deg. "
           "Firstpossible120; comparable "
           "completion must be by400. No4s window.\",\"v3_note\":\"V3 did not "
           "issue replacements at400; its "
           "comparable misses are not actual timeouts. Native v4 events are "
           "separately counted.\",\"starts\":"
        << requests.size() << ",\"comparable_stabilized_completions_by400\":" << comparable
        << ",\"v3_first_stabilized_only_after400\":" << late_only
        << ",\"native_completion_events\":" << native_successes
        << ",\"actual_timeouts\":" << native_timeouts << ",\"episode_cap_interruptions\":" << cap
        << ",\"other_terminal_interruptions\":" << terminal
        << ",\"depth_limit_terminal_interruptions\":" << depth_terminals
        << ",\"tilt_limit_terminal_interruptions\":" << tilt_terminals
        << ",\"end_of_recording_pending\":" << pending
        << ",\"actual_repeat_onsets_same_episode\":" << repeats
        << ",\"assigned_long_holds\":" << long_assigned
        << ",\"successful_long_holds_unfinished_at_end\":" << long_pending
        << ",\"native_training_seconds\":" << training_seconds
        << ",\"full_elapsed_seconds\":" << elapsed
        << ",\"starts_per_native_training_minute\":" << requests.size() * 60 / training_seconds
        << ",\"comparable_completions_per_native_training_minute\":"
        << comparable * 60 / training_seconds
        << ",\"starts_per_full_elapsed_minute\":" << requests.size() * 60 / elapsed
        << ",\"comparable_completions_per_full_elapsed_minute\":" << comparable * 60 / elapsed
        << ",\"timing_note\":\"Native training minute sums recorded train "
           "child-process wall only. Full "
           "elapsed runs from first recorded runner start through this "
           "checkpoint evaluation, including any "
           "parser-repair gap; compilation excluded.\",\"heading_note\":\"V3 "
           "exact post-onset target "
           "headings come from saved Task184 snapshots; nominal signed-angle "
           "draws reproduce frozen counter "
           "RNG. V4 headings/angles/events use recorded192byte "
           "traces.\",\"analysis_source_sha256\":"
        << quote(sha256(fs::path(__FILE__)))
        << ",\"protocol_sha256\":" << quote(sha256(root / "protocol.json"))
        << ",\"requests_csv_sha256\":"
        << quote(sha256(output / (prefix + "-training-requests.csv")))
        << ",\"predicate_note\":\"" << (half_second ?
            "Course is horizontal COM displacement between endpoints 0.5s apart. Projected speed, depth and tilt use the control endpoint. " :
            "Training predicates use instantaneous post-control COM velocity. ") <<
           "Failure counters overlap. They differ from deterministic held-out "
           "four-second mean "
           "criteria. For repeated requests, cumulative native counters and every event increment "
           "are checked.\",\"raw_batches\":[";
    for (size_t i = 0; i < paths.size(); ++i) {
        if (i)
            out << ',';
        out << "{\"path\":" << quote(fs::relative(paths[i], fs::current_path()).generic_string())
            << ",\"sha256\":" << quote(hashes[i]) << '}';
    }
    out << "],\"by_angle\":[";
    bool first = true;
    for (const auto &entry : angles) {
        if (!first)
            out << ',';
        first = false;
        out << "{\"angle_deg\":" << entry.first << ",\"starts\":" << entry.second[0]
            << ",\"comparable_completions\":" << entry.second[1]
            << ",\"eligible_ticks\":" << entry.second[2]
            << ",\"qualified_ticks\":" << entry.second[3]
            << ",\"course_fail_ticks\":" << entry.second[4]
            << ",\"speed_fail_ticks\":" << entry.second[5]
            << ",\"depth_fail_ticks\":" << entry.second[6]
            << ",\"tilt_fail_ticks\":" << entry.second[7]
            << ",\"max_qualifying_streak_ticks\":" << entry.second[8] << '}';
    }
    out << "],\"new_physical_controls\":0,\"new_optimizer_updates\":0}\n";
    require(bool(out), "Training summary write failed");
    std::cout << prefix << " training recorded requests=" << requests.size()
              << ", comparable_completions=" << comparable
              << ", native_timeouts=" << native_timeouts << ", cap_interruptions=" << cap
              << ", other_interruptions=" << terminal << ", pending=" << pending
              << ", newphysics=0 optimizer=0\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            pure_tests();
            training_pure_tests();
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--training") {
            training_analysis(fs::absolute(argv[2]), fs::absolute(argv[3]), std::stoi(argv[4]));
            return 0;
        }
        require(argc == 3 || argc == 5, "Use INPUT_ROOT OUTPUT_DIRECTORY [--archived-baseline BASELINE_DIR], "
                           "--self-test, or --training INPUT_ROOT OUTPUT_DIRECTORY THROUGH_UPDATE");
        const auto input = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
        const bool use_archive = argc == 5;
        require(!use_archive || std::string(argv[3]) == "--archived-baseline", "Unexpected analysis option");
        std::vector<Stage> stages;
        for (const char *name : {"baseline0", "eval8", "eval16", "eval24", "eval32"}) {
            const auto receipt = input / name / "result.json";
            if (!fs::exists(receipt)) continue;
            if (use_archive && std::string(name) == "baseline0") {
                const auto archived = fs::absolute(argv[4]);
                require(scalar(text(receipt), "reused_archived_baseline") == "true" &&
                        scalar(text(receipt), "attempted_controls") == "0" &&
                        scalar(text(input / "protocol.json"), "previous_baseline_result_sha256") ==
                            sha256(archived / "result.json"), "Unbound archived baseline");
                auto baseline = stage(archived.parent_path(), archived.filename().string());
                require(baseline.update == 0, "Expected archived update zero");
                baseline.name = "archived-baseline0";
                baseline.hashes.emplace_back(receipt.generic_string(), sha256(receipt));
                stages.push_back(std::move(baseline));
            } else stages.push_back(stage(input, name));
        }
        require(!stages.empty(), "No completed recorded assessments found");
        write_outputs(stages, input, output);
        std::cout << "Analyzed " << stages.size() << " completed checkpoints, " << stages.size() * 16
                  << " recorded turns; no physical controls or optimizer updates.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
