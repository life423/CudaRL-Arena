#include "motor/frozen_controller.h"
#include "core/checkpoint.h"
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>
#include <chrono>
#include <iostream>
namespace fs=std::filesystem;namespace p=cudarl::physics;
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
std::vector<std::string> split(const std::string& line){std::vector<std::string> out;std::stringstream stream(line);std::string value;while(std::getline(stream,value,','))out.push_back(value);return out;}
int main(int argc,char** argv){uint64_t attempts=0;try{require(argc==4,"Use live_demo_check ASSET EXPECTED_CSV OUTPUT_DIR");const fs::path output=fs::absolute(argv[3]);require(!fs::exists(output),"Verification identity exists; no overwrite");fs::create_directories(output);
    const auto start=std::chrono::steady_clock::now();cudarl::motor::FrozenController controller;require(controller.load(argv[1]),controller.error().c_str());require(controller.paused(),"Must load paused");require(!controller.step()&&controller.world().attempted_substeps==0,"Paused load advanced");
    const auto good=cudarl::checkpoint::read(argv[1]);auto corrupt=good;corrupt[0]^=1;const auto invalid=output/"invalid-asset.bin";cudarl::checkpoint::write_atomic(invalid.string(),corrupt);
    require(!controller.load(invalid.string())&&!controller.start()&&!controller.step(),"Corrupt asset did not fail closed");require(controller.load(argv[1]),"Reload approved actor failed");
    std::ifstream input(argv[2]);require(bool(input),"Missing expected frozen trace");std::string line;std::getline(input,line);const auto columns=split(line);std::map<std::string,size_t> index;for(size_t i=0;i<columns.size();++i)index[columns[i]]=i;
    float maximum=0;int rows=0,passed=0;
    for(int id=0;id<4;++id){require(controller.reset(id)&&controller.paused(),"Reset failed/autoruns");require(controller.start(),"Start failed");controller.pause();require(!controller.step()&&controller.world().attempted_substeps==0,"Pause advanced");require(controller.start(),"Resume failed");
        for(int tick=1;tick<=80;++tick){require(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10,"Native check wall cap");require(attempts+10<=3200,"Native check budget cap");const auto before=controller.world().attempted_substeps;const bool advanced=controller.step();attempts+=controller.world().attempted_substeps-before;require(advanced,controller.error().c_str());
            require(bool(std::getline(input,line)),"Expected CSV ended early");const auto values=split(line);require(values.size()==columns.size(),"Malformed expected row");auto expected=[&](const char* name){return std::stof(values[index.at(name)]);};require(int(expected("case"))==id&&int(expected("tick"))==tick,"Case/clock mismatch");
            const auto& w=controller.world();const auto& b=w.physical.body;const auto metric=controller.metrics();
            const char* fields[]={"head_x","head_y","head_z","qx","qy","qz","qw","j0","j1","j2","j3","phase","amplitude","frequency","positive_work_J","forward_m","lateral_m","depth_m","speed_error_m_s"};
            const float actual[]={b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w,b.angle[0],b.angle[1],b.angle[2],b.angle[3],w.phase,w.parameter[0],w.parameter[1],b.work,metric.forward_m,metric.lateral_m,metric.depth_m,metric.speed_error_m_s};
            for(size_t j=0;j<std::size(fields);++j){const float ref=expected(fields[j]);require(std::isfinite(ref)&&std::isfinite(actual[j]),"Nonfinite comparison");const float score=std::fabs(actual[j]-ref)/(1e-5f+1e-5f*std::fabs(ref));maximum=std::fmax(maximum,score);require(score<=1,"Frozen actor native trajectory mismatch");}++rows;
        }require(controller.finished()&&controller.paused(),"Must stop after4s");require(!controller.step(),"Completed controller advanced");require(controller.task().outcome==cudarl::straight_motor::Outcome::Success,"Frozen task gate failed");++passed;
    }
    std::ofstream result(output/"result.json");result<<"{\"native_passed\":"<<passed<<",\"matched_rows\":"<<rows<<",\"physical_substeps\":"<<attempts<<",\"max_normalized_difference\":"<<maximum<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"corrupt_load_rejected\":true,\"pause_reset_bounds_passed\":true}\n";
    std::cout<<"Native live path: "<<passed<<"/4, "<<rows<<" frozen rows, "<<attempts<<" substeps, max score="<<maximum<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"; actual attempts="<<attempts<<'\n';return 3;}}
