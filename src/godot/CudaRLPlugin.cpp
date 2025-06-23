#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include "../core/vectorized_environment.h"

using namespace godot;

class CudaRLPlugin : public Node {
    GDCLASS(CudaRLPlugin, Node)

private:
    cudarl::VectorizedEnvironment *environment;
    const int MAX_VISIBLE_ENVS = 16;

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("initialize"), &CudaRLPlugin::initialize);
        ClassDB::bind_method(D_METHOD("get_visualization_data"), &CudaRLPlugin::get_visualization_data);
    }

public:
    void initialize(int width, int height, int batch_size) {
        environment = new cudarl::VectorizedEnvironment(width, height, batch_size);
        environment->reset();
    }

    TypedArray<Array> get_visualization_data() {
        TypedArray<Array> visual_data;
        auto grids = environment->getGrids();
        
        for (int i = 0; i < std::min(MAX_VISIBLE_ENVS, (int)grids.size()); ++i) {
            Array grid_data;
            for (auto value : grids[i]) {
                grid_data.push_back(value);
            }
            visual_data.push_back(grid_data);
        }
        return visual_data;
    }
};
