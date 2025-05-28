extends Node
class_name CudaRLEnvironment

signal environment_reset
signal environment_stepped(action: int, reward: float)
signal environment_done

var width: int = 10
var height: int = 10
var agent_x: int = 5
var agent_y: int = 5
var grid_data: Array = []

func _init():
    set_name("CudaRLEnvironment")

func initialize(w: int, h: int):
    width = w
    height = h
    grid_data.resize(width * height)
    for i in range(grid_data.size()):
        grid_data[i] = randf() * 0.5

func reset():
    agent_x = width / 2
    agent_y = height / 2
    # Reset grid
    for i in range(grid_data.size()):
        grid_data[i] = randf() * 0.5
    # Set goal position (top-right)
    grid_data[width - 1] = 1.0
    environment_reset.emit()

func step(action: int):
    match action:
        0: agent_y = max(0, agent_y - 1)
        1: agent_x = min(width - 1, agent_x + 1)
        2: agent_y = min(height - 1, agent_y + 1)
        3: agent_x = max(0, agent_x - 1)
    
    var reward = -0.01
    if agent_x == width - 1 and agent_y == 0:
        reward = 1.0
        environment_done.emit()
    
    environment_stepped.emit(action, reward)

func get_agent_x() -> int:
    return agent_x

func get_agent_y() -> int:
    return agent_y

func get_grid_data() -> Array:
    return grid_data

func get_width() -> int:
    return width

func get_height() -> int:
    return height

func get_reward() -> float:
    if agent_x == width - 1 and agent_y == 0:
        return 1.0
    return -0.01

func is_done() -> bool:
    return agent_x == width - 1 and agent_y == 0
