#include <iostream>
#include <vector>
// #include "kernels.cuh"
#include "kernels.cuh"

using namespace cudarl;

int main() {
    // Create test vectors
    std::vector<float> a, b;
    for (int i = 0; i < 10; ++i) {
        a.push_back(i);
        b.push_back(i * 2);
    }
    
    // Do GPU computation
    std::vector<float> result = addVectors(a, b);
    
    // Print results
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < result.size(); ++i) {
        std::cout << a[i] << " + " << b[i] << " = " << result[i] << std::endl;
    }
    
    return 0;
}