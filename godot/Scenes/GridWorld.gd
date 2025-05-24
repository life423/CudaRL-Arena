extends Node2D

@onready var env = $CudaRLEnvironment
@onready var grid_container = $GridContainer
@onready var agent_sprite = $AgentSprite
@onready var reward_label = $UI/RewardLabel
@onready var status_label = $UI/StatusLabel

var cell_size = 50
var grid_width = 10
var grid_height = 10
var cells = []

func _ready():
	# Initialize the CUDA environment
	env.initialize(grid_width, grid_height)
	
	# Connect signals
	env.connect("environment_reset", _on_environment_reset)
	env.connect("environment_stepped", _on_environment_stepped)
	env.connect("environment_done", _on_environment_done)
	
	# Create grid cells
	_create_grid()
	
	# Reset environment
	env.reset()
	
	status_label.text = "Environment ready"

func _create_grid():
	# Clear existing cells
	for cell in cells:
		cell.queue_free()
	cells.clear()
	
	# Create new grid
	for y in range(grid_height):
		for x in range(grid_width):
			var cell = ColorRect.new()
			cell.size = Vector2(cell_size, cell_size)
			cell.position = Vector2(x * cell_size, y * cell_size)
			cell.color = Color(0.2, 0.2, 0.2, 1.0)
			add_child(cell)
			cells.append(cell)
	
	# Update agent sprite
	agent_sprite.position = Vector2(
		env.get_agent_x() * cell_size + cell_size/2,
		env.get_agent_y() * cell_size + cell_size/2
	)

func _update_grid():
	# Get grid data from environment
	var grid_data = env.get_grid_data()
	
	# Update cell colors based on grid data
	for i in range(min(cells.size(), grid_data.size())):
		var value = grid_data[i]
		cells[i].color = Color(value, value, 0.5, 1.0)
	
	# Update agent position
	agent_sprite.position = Vector2(
		env.get_agent_x() * cell_size + cell_size/2,
		env.get_agent_y() * cell_size + cell_size/2
	)
	
	# Update reward display
	reward_label.text = "Reward: %.3f" % env.get_reward()

func _on_environment_reset():
	_update_grid()
	status_label.text = "Environment reset"

func _on_environment_stepped(action, reward):
	_update_grid()
	status_label.text = "Action: %d, Reward: %.3f" % [action, reward]

func _on_environment_done():
	status_label.text = "Episode complete!"

func _input(event):
	if event is InputEventKey and event.pressed:
		var action = -1
		
		# Map arrow keys to actions
		match event.keycode:
			KEY_UP:
				action = 0  # up
			KEY_RIGHT:
				action = 1  # right
			KEY_DOWN:
				action = 2  # down
			KEY_LEFT:
				action = 3  # left
			KEY_R:
				env.reset()
				return
		
		# Take action if valid
		if action >= 0:
			env.step(action)