#include <chrono>
#include <filesystem>
#include <fstream>
#include "check.h"
#include "core/checkpoint.h"

namespace {
namespace cp=cudarl::checkpoint;
std::string temporary(const char* tag) {
    const auto directory=std::filesystem::path(__FILE__).parent_path().parent_path()/"build"/"pond-learning-checks";
    std::filesystem::create_directories(directory);
    return (directory/(std::string(tag)+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".training")).u8string();
}
}

TEST_CASE("training resume: checksum rejects corruption and failed replacement preserves the prior save") {
    const auto path=temporary("checkpoint-io-");const std::vector<uint8_t> first{1,2,3,4,5};cp::write_atomic(path,first);CHECK(cp::read(path)==first);
    std::filesystem::create_directory(std::filesystem::u8path(path+".tmp"));
    bool failed=false;try {cp::write_atomic(path,{9,8,7});} catch(const std::exception&) {failed=true;}CHECK(failed && cp::read(path)==first);
    std::filesystem::remove(std::filesystem::u8path(path+".tmp"));
    {std::fstream file(std::filesystem::u8path(path),std::ios::binary|std::ios::in|std::ios::out);file.seekp(24);const char bad=99;file.write(&bad,1);}
    bool corrupt=false;try {cp::read(path);} catch(const std::exception&) {corrupt=true;}CHECK(corrupt);
    std::filesystem::remove(std::filesystem::u8path(path));
}
