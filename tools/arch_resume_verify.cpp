#include "demos/feeding_session.h"
#include "rl/arch_policy.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace ar=cudarl::arch;
namespace cp=cudarl::checkpoint;
namespace rl=cudarl::rl;
namespace fs=std::filesystem;
namespace {
struct Snapshot {
    uint32_t version=0;
    int worlds=0,adopted_update=0,applied_update=0,learner_update=0;
    rl::ArchSource source;
    ar::State frame;
    std::vector<float> adopted,live_actor,calm_reference;
    ar::Context desired{},practiced{};
    int parent_stage_update=0;bool imported_full_state=false;uint64_t parent_fingerprint=0;
    std::vector<uint8_t> training;
};
Snapshot snapshot(const fs::path& path) {
    const auto bytes=cp::read(path.u8string());cp::Reader in{bytes};Snapshot out;
    if(in.value<uint32_t>()!=0x41524348u) throw std::runtime_error("Expected an ARCH checkpoint");
    out.version=in.value<uint32_t>();
    if((out.version<1 || out.version>3) || in.value<uint32_t>()!=sizeof(ar::State))
        throw std::runtime_error("Unsupported arch checkpoint layout");
    out.worlds=in.value<int>();out.source=rl::ArchSource::read(in);out.frame=in.value<ar::State>();
    const auto count=static_cast<size_t>(rl::swimmer_specification(4,ar::kObs).num_params());
    out.adopted=in.array<float>(count);out.adopted_update=in.value<int>();
    out.live_actor=in.array<float>(count);out.applied_update=in.value<int>();
    if(out.version>=2) {
        if(out.version==2) {
            struct LegacyContext {ar::V current;int version;};static_assert(sizeof(LegacyContext)==16);
            const auto desired=in.value<LegacyContext>(),practiced=in.value<LegacyContext>();
            out.desired.current=desired.current;out.desired.version=desired.version;
            out.practiced.current=practiced.current;out.practiced.version=practiced.version;
        } else {out.desired=in.value<ar::Context>();out.practiced=in.value<ar::Context>();}
        out.calm_reference=in.array<float>(count);
    }
    if(out.version>=3) {out.parent_stage_update=in.value<int>();out.imported_full_state=in.value<bool>();out.parent_fingerprint=in.value<uint64_t>();}
    out.training=in.array<uint8_t>(128000000);in.finish();
    cp::Reader training{out.training};out.learner_update=training.value<int>();
    if(out.worlds!=1024 || out.adopted.size()!=count || out.live_actor.size()!=count || out.learner_update<0)
        throw std::runtime_error("Expected a valid 1,024-world full arch save");
    return out;
}
std::vector<uint8_t> file_bytes(const fs::path& path) {
    if(fs::file_size(path)>256000000u) throw std::runtime_error("Unexpectedly large checkpoint");
    std::ifstream in(path,std::ios::binary);if(!in) throw std::runtime_error("Could not read protected original");
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
std::vector<uint8_t> source_bytes(const rl::ArchSource& source) {cp::Writer out;source.write(out);return out.data;}
// Canonical meaningful fields avoid making promises about C++ padding bytes
// while checking every physical, pellet, counter and observation-history field.
std::vector<uint8_t> frame_bytes(const ar::State& state) {
    cp::Writer out;const auto& s=state.pond;const auto& body=s.body;
    out.value(body.position);out.value(body.orientation);
    for(float x:body.angle) out.value(x);for(float x:body.velocity) out.value(x);
    out.value(body.work);out.value(body.travel);
    for(const auto& p:s.pellets) {
        out.value(p.position);out.value(p.velocity);out.value(p.age);out.value(p.distance);out.value(p.progress);
        out.value(p.id);out.value(p.active);out.value(p.settling);
    }
    for(float x:s.last_action) out.value(x);
    out.value(s.seed);out.value(s.environment);out.value(s.episode);out.value(s.next_id);
    out.value(s.steps);out.value(s.eaten);out.value(s.since_food);out.value(s.target);out.value(s.desired_foods);
    out.value(s.still_steps);out.value(s.max_still);out.value(s.anchor);
    out.value(s.decision);out.value(s.retention);out.value(s.invalid);
    const auto& h=state.habitat;out.value(h.scale);out.value(h.arch);out.value(h.center);
    out.value(h.major);out.value(h.minor);out.value(h.yaw);
    out.value(state.contacts);out.value(state.passages);out.value(state.clean_passages);
    out.value(state.pending_side);out.value(state.last_full_side);out.value(state.contacted_in_passage);
    out.value(state.crossed_opening);out.value(state.passage_valid);return out.data;
}
bool same_context(const ar::Context& a,const ar::Context& b) {
    return a.current.x==b.current.x && a.current.y==b.current.y && a.current.z==b.current.z && a.version==b.version && a.basin.layout==b.basin.layout;
}
uint64_t fingerprint(const std::vector<uint8_t>& bytes) {
    uint64_t hash=14695981039346656037ull;for(uint8_t x:bytes) {hash^=x;hash*=1099511628211ull;}return hash;
}
}
int main(int argc,char** argv) {
    int checks=0;
    auto check=[&](bool success,const char* message) {++checks;if(!success) throw std::runtime_error(message);};
    try {
        if(argc!=3 && argc!=4) throw std::runtime_error("Usage: arch_resume_verify <protected-save> <fresh-output-checkpoint> [cove]");
        const bool cove=argc==4 && std::string(argv[3])=="cove";
        if(argc==4 && !cove) throw std::runtime_error("Unknown import mode");
        const auto source=fs::absolute(fs::u8path(argv[1])).lexically_normal();
        const auto output=fs::absolute(fs::u8path(argv[2])).lexically_normal();
        check(source!=output,"Output must differ from the protected original");
        check(!fs::exists(output),"Output must be a fresh checkpoint path");
        check(fs::is_regular_file(source),"Protected source checkpoint is missing");
        const auto original_file=file_bytes(source);const auto before=snapshot(source);
        check(before.version>=1 && before.version<=3,"Unsupported source version");
        fs::create_directories(output.parent_path());if(!cove) fs::copy_file(source,output,fs::copy_options::none);
        const auto missing_source=output.parent_path()/(output.filename().u8string()+".missing-feeding-source");
        check(!fs::exists(missing_source),"Expected deliberately absent external feeding source");
        auto session=cove?cudarl::demos::create_cove_session(source.u8string(),output.u8string(),before.worlds):
            cudarl::demos::create_arch_session(missing_source.u8string(),output.u8string(),before.worlds);
        // Initial references run before PPO. Pause immediately, then join the
        // worker and serialize its restored state without a learner update.
        session->set_paused(true);session->stop();const auto status=session->stats();
        check(status.error.empty(),status.error.empty()?"Arch worker failed":status.error.c_str());
        check(status.save_error.empty(),status.save_error.empty()?"Migration save failed":status.save_error.c_str());
        check(status.resumed && status.loaded_update==before.learner_update,"Learner did not restore at its existing update");
        check(status.saved_update==before.learner_update,"Migration performed an unexpected PPO update");
        check(status.adopted_update==before.adopted_update && status.applied_update==before.applied_update,"Validated/live actor versions changed during migration");
        session.reset();const auto after=snapshot(output);
        check(after.version==3,"Migration did not write the new version");
        check(after.worlds==before.worlds,"Practice population changed during migration");
        check(source_bytes(after.source)==source_bytes(before.source),"Embedded source provenance or parent weights changed");
        check(frame_bytes(after.frame)==frame_bytes(before.frame),"Visible physical state, food or passage history changed");
        check(after.adopted==before.adopted && after.adopted_update==before.adopted_update,"Approved actor changed");
        check(after.live_actor==before.live_actor && after.applied_update==before.applied_update,"Applied actor changed");
        check(after.calm_reference==(before.version>=2?before.calm_reference:before.adopted),"Calm skill reference changed");
        auto desired=before.desired;if(cove && desired.basin.layout!=1) {desired.basin.layout=1;++desired.version;}
        check(same_context(after.desired,desired) && same_context(after.practiced,before.practiced),"Requested/practiced context was not preserved or deliberately versioned");
        if(cove) check(after.parent_stage_update==before.learner_update && after.imported_full_state,"Full parent-state import provenance missing");
        cp::Writer expected;expected.data=before.training;
        if(before.version==1) expected.value(ar::Context{});
        else if(before.version==2) expected.value(int{0});
        check(after.training==expected.data,"PPO actor, critic, deviations, Adam or practice-world bytes changed");
        check(after.learner_update==before.learner_update,"Learner counter changed");
        check(file_bytes(source)==original_file,"Protected original checkpoint file was modified");
        std::printf("ARCH_RESUME_VERIFY checks=%d learner_update=%d adopted=%d applied=%d worlds=%d original_trainer_bytes=%zu migrated_trainer_bytes=%zu original_file_fingerprint=%llu PASS\n",
            checks,after.learner_update,after.adopted_update,after.applied_update,after.worlds,before.training.size(),after.training.size(),
            static_cast<unsigned long long>(fingerprint(original_file)));
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"ARCH_RESUME_VERIFY checks=%d FAIL: %s\n",checks,error.what());return 1;
    }
}
