// Final isolated profile; historical fused/unfused evaluator sources and
// executables remain unchanged and bound to their original evidence.
#include "frozen_interface_arithmetic_profile_v2.h"
#define main historical_arithmetic_entry_v2
#include "frozen_interface_arithmetic_v2.cu"
#undef main
#undef cbrtf
int main(int argc,char** argv){try{
    cudarl::frozen_interface_arithmetic_v2::validate();
    const int result=historical_arithmetic_entry_v2(argc,argv);
    if(result==0){std::ofstream out(std::filesystem::absolute(argv[2])/"canonical-radius.json");
        out<<"{\"schema\":1,\"arithmetic_profile\":\"cpu_compatible_no_implicit_fma_v1\",";
        cudarl::frozen_interface_arithmetic_v2::metadata(out);out<<"}\n";out.close();
        if(!out)throw std::runtime_error("Canonical radius metadata write failed");}
    return result;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
