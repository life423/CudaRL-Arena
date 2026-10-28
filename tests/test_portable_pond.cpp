#include <cstdio>
#include <fstream>
#include <sstream>
#include "demos/pond_playback.h"
#if CUDARL_WITH_CUDA
#error Portable regression tests must compile without CUDA.
#endif
namespace {int checks=0;void check(bool value,const char* message) {++checks;if(!value) throw std::runtime_error(message);}}
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::invalid_argument("Supply the saved policy");const auto policy=cudarl::rl::SelectorPolicy::load(argv[1]);
        const auto random=cudarl::rl::initial_pond_policy();
        const auto repeat=cudarl::rl::initial_pond_policy();
        const auto other=cudarl::rl::initial_pond_policy(8);
        check(random.motor==repeat.motor && random.selector==repeat.selector,"Untrained initialization is not repeatable");
        check(random.motor!=policy.motor && random.selector!=policy.selector,"Untrained mode retained learned weights");
        check(random.motor!=other.motor && random.selector!=other.selector,"Untrained seed does not affect both networks");
        check(random.motor==cudarl::nn::orthogonal_params(cudarl::rl::swimmer_specification(4,33),7,{std::sqrt(2.f),std::sqrt(2.f),.01f}),"Untrained motor does not use PPO initialization");
        auto untrained=cudarl::demos::create_pond_playback(random,"cpu");
        auto repeated_untrained=cudarl::demos::create_pond_playback(repeat,"cpu");
        for(int tick=0;tick<80;++tick) {
            const auto& a=untrained->step();const auto& b=repeated_untrained->step();
            check(cudarl::deep::norm(a.state.body.position-b.state.body.position)==0 && a.selected==b.selected,"Untrained physics or neural decisions are not repeatable");
        }
        auto cpu=cudarl::demos::create_pond_playback(policy,"cpu");auto fallback=cudarl::demos::create_pond_playback(policy,"auto","missing-cuda-module");
        check(std::string(cpu->backend())=="CPU","Forced CPU did not select CPU");check(std::string(fallback->backend())=="CPU","Missing CUDA did not fall back");
        check(!fallback->detail().empty(),"Fallback reason is missing");
        auto invalid_module=cudarl::demos::create_pond_playback(policy,"auto",argv[1]);
        check(std::string(invalid_module->backend())=="CPU","An unloadable CUDA module did not fall back safely");
        for(int tick=0;tick<1200;++tick) {
            const auto& a=cpu->step();const auto& b=fallback->step();
            check(cudarl::deep::norm(a.state.body.position-b.state.body.position)==0,"Fallback changed physical trajectory");
            check(a.selected==b.selected && a.state.eaten==b.state.eaten,"Fallback changed controller decisions or food");
        }
        check(cpu->frame().state.eaten>10,"The retained swimmer did not collect food");check(cpu->frame().state.steps==1200,"Playback did not advance at fixed steps");
        bool rejected=false;try {cudarl::demos::create_pond_playback(policy,"cuda","missing-cuda-module");} catch(const std::runtime_error&) {rejected=true;}
        check(rejected,"Strict CUDA silently changed backend");rejected=false;
        try {cudarl::demos::create_pond_playback(policy,"invalid");} catch(const std::invalid_argument&) {rejected=true;}check(rejected,"Invalid backend accepted");
        const auto after=cudarl::rl::SelectorPolicy::load(argv[1]);check(policy.motor==after.motor && policy.selector==after.selector,"Playback modified saved weights");
        if(argc>2) {
            auto reference_player=cudarl::demos::create_pond_playback(policy,"cpu");std::ifstream reference(argv[2]);check(bool(reference),"Pre-portability reference is missing");
            std::string line;std::getline(reference,line);int rows=0;
            while(std::getline(reference,line)) {
                std::stringstream row(line);std::string item;std::vector<double> values;
                while(std::getline(row,item,',')) values.push_back(std::stod(item));check(values.size()==10,"Invalid reference row");
                const auto& snapshot=reference_player->step();const auto& p=snapshot.state.body.position;const auto& q=snapshot.state.body.orientation;
                const float actual[]{p.x,p.y,p.z,q.x,q.y,q.z,q.w};
                for(int j=0;j<7;++j) check(std::fabs(actual[j]-values[j+1])<2e-6,"Portable runtime changed legacy CPU trajectory");
                check(snapshot.state.eaten==int(values[8]) && snapshot.selected==int(values[9]),"Portable runtime changed legacy food decisions");++rows;
            }
            check(rows==80,"Incomplete pre-portability reference");check(cpu->frame().state.eaten==18,"The one-minute legacy collection result changed");
        }
        if(argc>3) {
            auto optional=cudarl::demos::create_pond_playback(policy,"auto",argv[3]);
            const bool expect_cuda=argc<5 || std::string(argv[4])!="cpu";
            check(std::string(optional->backend())==(expect_cuda?"CUDA":"CPU"),"Optional module selected the wrong backend");
            auto comparison=cudarl::demos::create_pond_playback(policy,"cpu");
            for(int tick=0;tick<40;++tick) {
                const auto& a=optional->step();const auto& b=comparison->step();
                check(cudarl::deep::norm(a.state.body.position-b.state.body.position)<.003f,"CPU/CUDA playback physics disagree");
                check(a.selected==b.selected,"CPU/CUDA selector disagrees on the short reference trace");
            }
            for(int tick=40;tick<1200;++tick) optional->step();
            check(optional->frame().state.eaten>10,"Optional module did not retain food collection");
            const auto before=optional->frame();optional->set_current({.005f,0,0});
            check(optional->frame().state.steps==before.state.steps && optional->frame().selected==before.selected,"Current control reset the portable session");
            check(cudarl::deep::norm(optional->frame().state.body.position-before.state.body.position)==0,"Current control moved the portable body instantly");
            check(std::fabs(optional->current().x-.005f)<1e-8f,"Portable current was not retained");
            optional->step();check(optional->frame().state.steps==before.state.steps+1,"Portable current stopped the simulation");
            optional->set_current({});check(cudarl::deep::norm(optional->current())==0,"Portable calm control failed");
            std::printf("PORTABLE_CURRENT backend=%s state_continuity=true\n",optional->backend());
        }
        std::printf("PORTABLE_POND checks=%d failures=0 foods=%d\n",checks,cpu->frame().state.eaten);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"PORTABLE_POND failure: %s\n",e.what());return 1;}
}
