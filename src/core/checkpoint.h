#ifndef CUDARL_CORE_CHECKPOINT_H
#define CUDARL_CORE_CHECKPOINT_H
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace cudarl::checkpoint {
struct Writer {
    std::vector<uint8_t> data;
    template<class T> void value(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p=reinterpret_cast<const uint8_t*>(&v);data.insert(data.end(),p,p+sizeof(T));
    }
    template<class T> void array(const std::vector<T>& v) {
        value(static_cast<uint64_t>(v.size()));
        if(!v.empty()) {const auto* p=reinterpret_cast<const uint8_t*>(v.data());data.insert(data.end(),p,p+v.size()*sizeof(T));}
    }
};
struct Reader {
    const std::vector<uint8_t>& data;size_t position=0;
    template<class T> T value() {
        static_assert(std::is_trivially_copyable_v<T>);
        if(position>data.size() || sizeof(T)>data.size()-position) throw std::runtime_error("Truncated checkpoint");
        T v;std::memcpy(&v,data.data()+position,sizeof(T));position+=sizeof(T);return v;
    }
    template<class T> std::vector<T> array(size_t maximum=10000000) {
        const auto n=value<uint64_t>();
        if(n>maximum || n>(data.size()-position)/sizeof(T)) throw std::runtime_error("Invalid checkpoint array size");
        std::vector<T> v(static_cast<size_t>(n));
        if(n) std::memcpy(v.data(),data.data()+position,static_cast<size_t>(n)*sizeof(T));
        position+=static_cast<size_t>(n)*sizeof(T);return v;
    }
    void finish() const {if(position!=data.size()) throw std::runtime_error("Unexpected checkpoint data");}
};
// Checksummed, bounded binary payload; write/flush a sibling temporary file,
// then atomically replace the destination. A failed replacement keeps the old file.
// Concurrent writers use an exclusive, nonblocking lease on the target path.
uint64_t checksum(const std::vector<uint8_t>& data);
void write_atomic(const std::string& path,const std::vector<uint8_t>& data);
std::vector<uint8_t> read(const std::string& path);
class Lease {
public:
    explicit Lease(const std::string& path);
    ~Lease();
    Lease(const Lease&)=delete;Lease& operator=(const Lease&)=delete;
private:
    void* handle_=nullptr;
    int descriptor_=-1;
};
struct Busy:std::runtime_error {using std::runtime_error::runtime_error;};
}
#endif
