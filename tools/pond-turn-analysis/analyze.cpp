// Recorded CSV analysis only. No simulator, policy, CUDA or optimizer includes.
// bcrypt.h requires the Windows type declarations first.
// clang-format off
#include <windows.h>
#include <bcrypt.h>
// clang-format on
#include <algorithm>
#include <array>
#include <cmath>
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
#include <vector>
namespace fs = std::filesystem;
namespace {
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
             protocol_id == "pond-motor-ppo-v3-course-2026-11-23") &&
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
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            pure_tests();
            return 0;
        }
        require(argc == 3, "Use --self-test or INPUT_ROOT OUTPUT_DIRECTORY");
        const auto input = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
        std::vector<Stage> stages;
        for (const char *name : {"baseline0", "eval8", "eval16", "eval24", "eval32"})
            if (fs::exists(input / name / "result.json"))
                stages.push_back(stage(input, name));
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
