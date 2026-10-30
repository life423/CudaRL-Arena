extends Node2D

const NativeLabFactory = preload("res://shared/native_lab_factory.gd")

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

var arena
var _accum = 0.0

func _ready():
	var native := NativeLabFactory.create("MazeArena", ["set_walls", "get_goal", "step", "reset"])
	arena = native.instance
	if arena == null:
		_show_native_error(native.error)
		set_physics_process(false)
		set_process_input(false)
		return
	print(NativeLabFactory.device_description())

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

func _show_native_error(message: String) -> void:
	human_rect.hide()
	ai_rect.hide()
	goal_rect.hide()
	var layer := CanvasLayer.new()
	add_child(layer)
	var panel := PanelContainer.new()
	panel.position = Vector2(16, 16)
	panel.custom_minimum_size.x = 300
	layer.add_child(panel)
	var content := VBoxContainer.new()
	panel.add_child(content)
	var explanation := Label.new()
	explanation.text = message
	explanation.custom_minimum_size.x = 280
	explanation.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	content.add_child(explanation)
	var back := Button.new()
	back.name = "BackToDemos"
	back.text = "Pond menu"
	back.custom_minimum_size.y = 44
	back.pressed.connect(DemoLaunch.show_menu)
	content.add_child(back)

func _exit_tree() -> void:
	arena = null

func _unhandled_key_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		get_viewport().set_input_as_handled()
		DemoLaunch.show_menu()

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
	if arena == null:
		return
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
	if arena == null:
		return
	if Input.is_action_just_pressed("ui_select"):
		arena.reset()
