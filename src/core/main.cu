#include <iostream>
#include "environment.h"

using namespace cudarl;

int main() {
    try {
        std::cout << "Creating environment..." << std::endl;
        Environment env(0);

        std::cout << "Resetting environment (should launch GPU kernel)..." << std::endl;
        env.reset();

        std::cout << "Stepping environment with action=1..." << std::endl;
        env.step(1);

        std::cout << "Back on the CPU, PoC complete!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}