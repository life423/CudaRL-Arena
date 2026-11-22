#include "checkpoint.h"
#include <filesystem>
#include <fstream>
#include <cerrno>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif
namespace cudarl::checkpoint {
uint64_t checksum(const std::vector<uint8_t>& bytes) {
    uint64_t h=14695981039346656037ull;for(uint8_t b:bytes) {h^=b;h*=1099511628211ull;}return h;
}
namespace {
constexpr uint64_t magic=0x31504B434C524443ull;
constexpr uint64_t maximum_bytes=512ull*1024*1024;
std::filesystem::path filename(const std::string& path) {return std::filesystem::u8path(path);}
void parent(const std::filesystem::path& path) {if(!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());}
}
void write_atomic(const std::string& path,const std::vector<uint8_t>& bytes) {
    if(bytes.size()>maximum_bytes) throw std::runtime_error("Checkpoint exceeds size limit");
    const Lease writer(path);
    const auto target=filename(path),temporary=filename(path+".tmp");parent(target);
    Writer header;header.value(magic);header.value(static_cast<uint64_t>(bytes.size()));header.value(checksum(bytes));
    {
        std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
        out.write(reinterpret_cast<const char*>(header.data.data()),static_cast<std::streamsize>(header.data.size()));
        if(!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
        out.flush();if(!out) throw std::runtime_error("Could not write full checkpoint");
    }
#ifdef _WIN32
    HANDLE file=CreateFileW(temporary.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE) throw std::runtime_error("Could not flush checkpoint");
    const bool flushed=FlushFileBuffers(file)!=0;CloseHandle(file);
    if(!flushed || !MoveFileExW(temporary.c_str(),target.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Could not replace checkpoint; prior save retained");
#else
    const int fd=::open(temporary.c_str(),O_WRONLY);
    if(fd<0) throw std::runtime_error("Could not flush checkpoint");
    const int result=::fsync(fd);::close(fd);
    if(result!=0) throw std::runtime_error("Could not flush checkpoint");
    std::filesystem::rename(temporary,target);
#endif
}
std::vector<uint8_t> read(const std::string& path) {
    std::ifstream in(filename(path),std::ios::binary|std::ios::ate);
    const auto length=in.tellg();
    if(!in || length<24 || static_cast<uint64_t>(length)>maximum_bytes+24) throw std::runtime_error("Invalid checkpoint length");
    in.seekg(0);std::vector<uint8_t> header(24);in.read(reinterpret_cast<char*>(header.data()),24);
    Reader r{header};if(r.value<uint64_t>()!=magic) throw std::runtime_error("Unknown checkpoint format");
    const auto count=r.value<uint64_t>(),expected=r.value<uint64_t>();
    if(count!=static_cast<uint64_t>(length)-24) throw std::runtime_error("Truncated checkpoint");
    std::vector<uint8_t> bytes(static_cast<size_t>(count));
    if(count) in.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(count));
    if(!in || checksum(bytes)!=expected) throw std::runtime_error("Checkpoint checksum failed");return bytes;
}
Lease::Lease(const std::string& path) {
    const auto lock=filename(path+".lock");parent(lock);
#ifdef _WIN32
    HANDLE file=CreateFileW(lock.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
    if(file==INVALID_HANDLE_VALUE) {
        if(GetLastError()==ERROR_SHARING_VIOLATION || GetLastError()==ERROR_LOCK_VIOLATION)
            throw Busy("This checkpoint is already being written by another writer");
        throw std::runtime_error("Checkpoint folder is not writable");
    }
    handle_=file;
#else
    descriptor_=::open(lock.c_str(),O_CREAT|O_RDWR,0600);
    if(descriptor_<0) throw std::runtime_error("Checkpoint folder is not writable");
    if(::flock(descriptor_,LOCK_EX|LOCK_NB)!=0) {
        const int error=errno;::close(descriptor_);descriptor_=-1;
        if(error==EWOULDBLOCK||error==EAGAIN) throw Busy("This checkpoint is already being written by another writer");
        throw std::runtime_error("Could not lock checkpoint");
    }
#endif
}
Lease::~Lease() {
#ifdef _WIN32
    if(handle_) CloseHandle(handle_);
#else
    if(descriptor_>=0) {::flock(descriptor_,LOCK_UN);::close(descriptor_);}
#endif
}
}
