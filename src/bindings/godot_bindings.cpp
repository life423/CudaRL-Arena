#include <godot_cpp/godot.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include "../core/environment.h"

namespace godot {

class CudaRLEnvironment : public Node {
    GDCLASS(CudaRLEnvironment, Node)

private:
    std::unique_ptr<cudarl::Environment> env;
    int width;
    int height;
    
protected:
    static void _bind_methods() {
        // Register methods
        ClassDB::bind_method(D_METHOD("initialize", "width", "height"), &CudaRLEnvironment::initialize);
        ClassDB::bind_method(D_METHOD("reset"), &CudaRLEnvironment::reset);
        ClassDB::bind_method(D_METHOD("step", "action"), &CudaRLEnvironment::step);
        
        // Register getters
        ClassDB::bind_method(D_METHOD("get_width"), &CudaRLEnvironment::get_width);
        ClassDB::bind_method(D_METHOD("get_height"), &CudaRLEnvironment::get_height);
        ClassDB::bind_method(D_METHOD("get_agent_x"), &CudaRLEnvironment::get_agent_x);
        ClassDB::bind_method(D_METHOD("get_agent_y"), &CudaRLEnvironment::get_agent_y);
        ClassDB::bind_method(D_METHOD("get_reward"), &CudaRLEnvironment::get_reward);
        ClassDB::bind_method(D_METHOD("is_done"), &CudaRLEnvironment::is_done);
        ClassDB::bind_method(D_METHOD("get_grid_data"), &CudaRLEnvironment::get_grid_data);
        
        // Add signals
        ADD_SIGNAL(MethodInfo("environment_reset"));
        ADD_SIGNAL(MethodInfo("environment_stepped", PropertyInfo(Variant::INT, "action"), PropertyInfo(Variant::FLOAT, "reward")));
        ADD_SIGNAL(MethodInfo("environment_done"));
    }

public:
    CudaRLEnvironment() : width(10), height(10) {
        // Constructor
    }

    ~CudaRLEnvironment() {
        // Destructor (env will be cleaned up by unique_ptr)
    }

    void _ready() {
        UtilityFunctions::print("CudaRLEnvironment ready!");
    }

    void _process(double delta) {
        // Process logic (empty for now)
    }

    void initialize(int p_width, int p_height) {
        width = p_width;
        height = p_height;
        
        // Create new environment
        env = std::make_unique<cudarl::Environment>(0, width, height);
        UtilityFunctions::print("CudaRL Environment initialized with size: ", width, "x", height);
    }

    void reset() {
        if (env) {
            env->reset();
            emit_signal("environment_reset");
            UtilityFunctions::print("Environment reset");
        } else {
            UtilityFunctions::printerr("Environment not initialized!");
        }
    }

    void step(int action) {
        if (env) {
            env->step(action);
            emit_signal("environment_stepped", action, env->getReward());
            
            if (env->isDone()) {
                emit_signal("environment_done");
            }
        } else {
            UtilityFunctions::printerr("Environment not initialized!");
        }
    }

    int get_width() const {
        return env ? env->getWidth() : width;
    }

    int get_height() const {
        return env ? env->getHeight() : height;
    }

    int get_agent_x() const {
        return env ? env->getAgentX() : -1;
    }

    int get_agent_y() const {
        return env ? env->getAgentY() : -1;
    }

    float get_reward() const {
        return env ? env->getReward() : 0.0f;
    }

    bool is_done() const {
        return env ? env->isDone() : false;
    }

    Array get_grid_data() const {
        Array data;
        
        if (env) {
            std::vector<float> grid = env->getGrid();
            
            for (size_t i = 0; i < grid.size(); i++) {
                data.push_back(grid[i]);
            }
        }
        
        return data;
    }
};

// Module initialization
void initialize_cudarl_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    
    ClassDB::register_class<CudaRLEnvironment>();
    UtilityFunctions::print("CudaRL module initialized");
}

void uninitialize_cudarl_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    
    UtilityFunctions::print("CudaRL module uninitialized");
}

extern "C" {
    GDExtensionBool GDE_EXPORT cudarl_library_init(
        GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library,
        GDExtensionInitialization *r_initialization
    ) {
        godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

        init_obj.register_initializer(initialize_cudarl_module);
        init_obj.register_terminator(uninitialize_cudarl_module);
        init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

        return init_obj.init();
    }
}

} // namespace godot