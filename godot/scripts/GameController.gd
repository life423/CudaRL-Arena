extends Node2D

var cuda_env : CudaRLEnvironment
var grid_size := Vector2i(10, 10)
var cell_size := 50

func _ready():
    # Create CUDA environment
    cuda_env = CudaRLEnvironment.new()
    cuda_env.initialize(grid_size.x, grid_size.y)
    cuda_env.reset()
    
    # Connect signals
    cuda_env.environment_reset.connect(_on_environment_reset)
    cuda_env.environment_stepped.connect(_on_environment_stepped)
    cuda_env.environment_done.connect(_on_environment_done)

func _input(event):
    if event.is_action_pressed("ui_accept"):  # Space
        cuda_env.reset()
    elif event.is_action_pressed("ui_up"):
        cuda_env.step(0)
    elif event.is_action_pressed("ui_right"):
        cuda_env.step(1)
    elif event.is_action_pressed("ui_down"):
        cuda_env.step(2)
    elif event.is_action_pressed("ui_left"):
        cuda_env.step(3)

func _on_environment_reset():
    queue_redraw()

func _on_environment_stepped(action: int, reward: float):
    print("Action: ", action, " Reward: ", reward)
    queue_redraw()

func _on_environment_done():
    print("Episode complete!")

func _draw():
    # Draw grid
    var grid_data = cuda_env.get_grid_data()
    
    for y in range(grid_size.y):
        for x in range(grid_size.x):
            var idx = y * grid_size.x + x
            var value = grid_data[idx] if idx < grid_data.size() else 0.0
            
            # Draw cell
            var rect = Rect2(x * cell_size, y * cell_size, cell_size, cell_size)
            draw_rect(rect, Color(value, value, value))
            draw_rect(rect, Color.WHITE, false, 2.0)
    
    # Draw agent
    var agent_pos = Vector2(cuda_env.get_agent_x(), cuda_env.get_agent_y())
    var agent_rect = Rect2(agent_pos * cell_size, Vector2(cell_size, cell_size))
    draw_rect(agent_rect, Color.RED)
