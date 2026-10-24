extends Node2D

# Two agents (both random for now) race to the goal. Rules live in C++
# (src/envs/maze); this script only builds the walls and draws the board.

@onready var human_rect = $HumanRect
@onready var ai_rect    = $AIRect
@onready var goal_rect  = $GoalRect
@onready var walls_node = $Walls

const TILE_SIZE = 32
const STEP_HZ = 5.0
const WALL_COLOR = Color(0.3, 0.3, 0.3, 1)
const NUM_ACTIONS = 4  # Up, Right, Down, Left

var arena := MazeArena.new()
var _accum = 0.0

func _ready():
	print(CudaDevice.new().describe())

	# Single source of truth for the walls: sent to C++ for collision and drawn below.
	var wall_cells = PackedVector2Array()
	for x in range(10):
		wall_cells.append(Vector2(x, 0))
		wall_cells.append(Vector2(x, 9))
	for y in range(1, 9):
		wall_cells.append(Vector2(0, y))
		wall_cells.append(Vector2(9, y))
	for cell in [Vector2(2, 2), Vector2(2, 3), Vector2(2, 4), Vector2(4, 2),
			Vector2(6, 4), Vector2(6, 5), Vector2(6, 6), Vector2(4, 6), Vector2(5, 6)]:
		wall_cells.append(cell)

	arena.set_walls(wall_cells)
	_draw_walls(wall_cells)
	goal_rect.position = Vector2(arena.get_goal()) * TILE_SIZE

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

	var results = arena.step([randi() % NUM_ACTIONS, randi() % NUM_ACTIONS])
	var h = results[0]
	var a = results[1]
	human_rect.position = h["state"] * TILE_SIZE
	ai_rect.position    = a["state"] * TILE_SIZE

	if h["done"]:
		var winner = "human" if h["reward"] > a["reward"] else ("AI" if a["reward"] > h["reward"] else "draw")
		print("Episode over: ", winner)
		arena.reset()

func _input(_event):
	if Input.is_action_just_pressed("ui_select"):
		arena.reset()
