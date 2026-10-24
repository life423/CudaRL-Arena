extends Node2D

# preload your C++ extension class
@onready var cuda_ext := ArenaGDExtension.new()
@onready var human_rect = $HumanRect
@onready var ai_rect    = $AIRect
@onready var goal_rect  = $GoalRect
@onready var walls_node = $Walls

const TILE_SIZE = 32
const STEP_HZ = 5.0
const WALL_COLOR = Color(0.3, 0.3, 0.3, 1)
var _accum = 0.0

func _ready():
	# just to verify CUDA is loaded (safe without a GPU; reports why)
	var info = cuda_ext.hello_cuda()
	print(info)

	# Define maze layout (walls). This list is the single source of truth:
	# it is sent to the extension for collision AND drawn below.
	var maze_cells = PackedVector2Array()
	# Border walls
	for x in range(10):
		maze_cells.append(Vector2(x, 0))  # top
		maze_cells.append(Vector2(x, 9))  # bottom
	for y in range(1, 9):
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
	_draw_walls(maze_cells)
	goal_rect.position = Vector2(cuda_ext.get_goal()) * TILE_SIZE

func _draw_walls(cells: PackedVector2Array):
	for child in walls_node.get_children():
		child.queue_free()
	for cell in cells:
		var wall = ColorRect.new()
		wall.color = WALL_COLOR
		wall.position = cell * TILE_SIZE
		wall.size = Vector2(TILE_SIZE, TILE_SIZE)
		walls_node.add_child(wall)

func _physics_process(delta):
	_accum += delta
	if _accum < 1.0 / STEP_HZ:
		return
	_accum = 0.0

	# for now, pick random actions 0–3 (Up, Right, Down, Left; see src/gpu/actions.h)
	var human_action = randi() % 4
	var ai_action    = randi() % 4

	# call into your two-agent API
	var results = cuda_ext.step_environment([ human_action, ai_action ])

	# unpack and move the ColorRects
	var h = results[0]
	var a = results[1]

	human_rect.position = Vector2(h["state"].x, h["state"].y) * TILE_SIZE
	ai_rect.position    = Vector2(a["state"].x, a["state"].y) * TILE_SIZE

	if h["done"]:
		var winner = "human" if h["reward"] > a["reward"] else ("AI" if a["reward"] > h["reward"] else "draw")
		print("Episode over: ", winner)
		cuda_ext.reset()

func _input(event):
	if Input.is_action_just_pressed("ui_select"):
		cuda_ext.reset()
