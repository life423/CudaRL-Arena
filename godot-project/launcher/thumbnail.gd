extends Control

# Preserve the approved CartPole/Racing illustrations with atlas regions.
# Tadpole Pond uses a render captured from the actual scene. Text/buttons are native UI.
const SOURCE: Texture2D = preload("res://launcher/assets/demo-preview-source.png")
var demo := "cartpole"
var texture: Texture2D

func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	if demo == "pond_boulder":
		var path := "res://launcher/assets/boulder-pond.png"
		texture = load(path) if ResourceLoader.exists(path) else preload("res://launcher/assets/tadpole-pond.png")
	elif demo == "pond":
		texture = preload("res://launcher/assets/tadpole-pond.png")
	else:
		var atlas := AtlasTexture.new()
		atlas.atlas = SOURCE
		atlas.region = Rect2(45, 194, 667, 185) if demo == "cartpole" else Rect2(45, 413, 667, 204)
		atlas.filter_clip = true
		texture = atlas
	resized.connect(queue_redraw)

func _draw() -> void:
	var surface := StyleBoxFlat.new()
	surface.bg_color = Color("12171c")
	surface.set_corner_radius_all(6)
	surface.set_border_width_all(1)
	surface.border_color = Color("38404a")
	draw_style_box(surface, Rect2(Vector2.ZERO, size))
	if texture == null:
		return
	var ratio := minf((size.x - 2) / texture.get_width(), (size.y - 2) / texture.get_height())
	var image_size := texture.get_size() * ratio
	draw_texture_rect(texture, Rect2((size - image_size) * 0.5, image_size), false)
