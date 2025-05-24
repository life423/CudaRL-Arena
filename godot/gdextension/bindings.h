#pragma once

#include <godot_cpp/godot.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/node.hpp>
#include "../../src/environment.h"

namespace godot {

class CudaRLEnvironment : public Node {
    GDCLASS(CudaRLEnvironment, Node)

private:
    Environment* env;
    int width;
    int height;
    
protected:
    static void _bind_methods();

public:
    CudaRLEnvironment();
    ~CudaRLEnvironment();

    void _ready();
    void _process(double delta);
    
    // Methods exposed to Godot
    void initialize(int width, int height);
    void reset();
    void step(int action);
    
    // Getters
    int get_width() const;
    int get_height() const;
    int get_agent_x() const;
    int get_agent_y() const;
    float get_reward() const;
    bool is_done() const;
    
    // Get grid data as Array for Godot
    Array get_grid_data() const;
};

}