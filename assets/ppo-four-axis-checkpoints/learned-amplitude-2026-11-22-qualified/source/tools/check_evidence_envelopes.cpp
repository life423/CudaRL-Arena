// Read-only envelope validation for large recorded batches. No simulation/learning.
#include "core/checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc,char** argv) {
    if(argc!=2) {std::cerr<<"Use check_evidence_envelopes PATHS_FILE\n";return 2;}
    try {
        std::ifstream paths(std::filesystem::u8path(argv[1]));
        if(!paths) throw std::runtime_error("Cannot read evidence path list");
        std::string path;unsigned long long files=0,bytes=0;
        while(std::getline(paths,path)) {
            if(!path.empty()&&path.back()=='\r')path.pop_back();
            if(path.empty())continue;
            const auto payload=cudarl::checkpoint::read(path);
            ++files;bytes+=payload.size();
        }
        if(!paths.eof()||files==0)throw std::runtime_error("Empty or invalid evidence path list");
        std::cout<<"{\"passed\":true,\"files\":"<<files<<",\"payload_bytes\":"<<bytes
                 <<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
