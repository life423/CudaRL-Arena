extends Node2D

# preload your C++ extension class
@onready var cuda_ext := ArenaGDExtension.new()
@onready var human_rect = $HumanRect
@onready var ai_rect    = $AIRect

const TILE_SIZE = 32
const STEP_HZ = 5.0
var _accum = 0.0

func _ready():
	# just to verify CUDA is loaded
	var info = cuda_ext.hello_cuda()
	print(info)
	
	# Define maze layout (walls)
	var maze_cells = PackedVector2Array()
	# Border walls
	for x in range(10):
		maze_cells.append(Vector2(x, 0))  # top
		maze_cells.append(Vector2(x, 9))  # bottom
	for y in range(10):
		maze_cells.append(Vector2(0, y))  # left
		maze_cells.append(Vector2(9, y))  # right
	
	# Internal walls for simple maze
	maze_cells.append(Vector2(2, 2))
	maze_cells.append(Vector2(2, 3))
	maze_cells.append(Vector2(2, 4))
	maze_cells.append(Vector2(4, 2))
	maze_cells.append(Vector2(6, 4))
	maze_cells.append(Vector2(6, 5))
	maze_cells.append(Vector2(6, 6))
	maze_cells.append(Vector2(4, 6))
	maze_cells.append(Vector2(5, 6))
	
	cuda_ext.set_maze(maze_cells)

func _physics_process(delta):
	_accum += delta
	if _accum < 1.0 / STEP_HZ:
		return
	_accum = 0.0
	
	# for now, pick random actions 0–3
	var human_action = randi() % 4
	var ai_action    = randi() % 4

	# call into your two-agent API
	var results = cuda_ext.step_environment([ human_action, ai_action ])

	# unpack and move the ColorRects
	var h = results[0]
	var a = results[1]

	human_rect.position = Vector2(h["state"].x, h["state"].y) * TILE_SIZE
	ai_rect.position    = Vector2(a["state"].x, a["state"].y) * TILE_SIZE

func _input(event):
	if Input.is_action_just_pressed("ui_select"):
		cuda_ext.reset()
