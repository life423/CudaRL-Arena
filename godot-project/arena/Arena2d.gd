extends Node2D

# preload your C++ extension class
@onready var cuda_ext := ArenaGDExtension.new()
@onready var human_rect = $HumanRect
@onready var ai_rect    = $AIRect

const TILE_SIZE = 32

func _ready():
	# just to verify CUDA is loaded
	var info = cuda_ext.hello_cuda()
	print(info)

func _process(delta):
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
