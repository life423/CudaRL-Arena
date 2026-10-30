extends SceneTree

var checks := 0
var failures := 0
var output := "res://../build/boulder-menu-checks"
var source := "res://../build/boulder-menu-checks/personal-before.bin"
var cove_source := "res://../build/boulder-navigation/source-cove.bin"
var preview_path := ""
var checkpoint := ""
var protected_source := ""
var protected_cove := ""

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func settle() -> void:
	for tick in 8:
		await process_frame

func activate(control: Control) -> void:
	control.grab_focus()
	for pressed in [true, false]:
		var event := InputEventKey.new()
		event.keycode = KEY_SPACE
		event.pressed = pressed
		root.push_input(event)
	await settle()

func capture(name: String) -> void:
	if DisplayServer.get_name() == "headless":
		return
	await RenderingServer.frame_post_draw
	check(root.get_texture().get_image().save_png(output.path_join(name)) == OK, "Menu screenshot failed: " + name)

func isolate(node: Node) -> void:
	# This hook runs before _ready. Even a regression to the old feeding route
	# receives a private save rather than touching the user's running learner.
	if node.name == "BoulderPond":
		node.source_cove_checkpoint = protected_cove
		node.boulder_checkpoint = checkpoint
		node.population = 1024
		node.set_physics_process(false)
		node.ready.connect(func(): node.set_physics_process(false), CONNECT_ONE_SHOT)
	elif node.name == "FeedingPond":
		node.feeding_checkpoint_path = checkpoint + ".feeding.bin"
		node.population = 128
		node.set_physics_process(false)

func _initialize() -> void:
	Engine.max_fps = 60
	run.call_deferred()

func ready_and_paused(pond) -> bool:
	pond.set_physics_process(false)
	pond.lab.set_paused(true)
	pond.paused = true
	var started := Time.get_ticks_msec()
	pond._poll()
	while not pond.stats.get("learning_ready", false) and str(pond.stats.get("learning_error", "")).is_empty() and Time.get_ticks_msec() - started < 45000:
		await create_timer(0.1).timeout
		pond._poll()
	# A just-started rollout may finish after the pause request. Drain it before
	# comparing the save produced by leaving the scene with its reopened state.
	await create_timer(0.7).timeout
	pond._poll()
	var error: String = pond.lab.get_error()
	if error.is_empty():
		error = str(pond.stats.get("learning_error", ""))
	check(error.is_empty() and pond.stats.get("learning_ready", false), "Boulder navigation learner failed: " + error)
	return error.is_empty() and pond.stats.get("learning_ready", false)

func run() -> void:
	create_timer(180).timeout.connect(func(): push_error("Boulder menu smoke timed out"); quit(1))
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--menu-output="):
			output = argument.trim_prefix("--menu-output=")
		elif argument.begins_with("--menu-source="):
			source = argument.trim_prefix("--menu-source=")
		elif argument.begins_with("--menu-cove-source="):
			cove_source = argument.trim_prefix("--menu-cove-source=")
		elif argument.begins_with("--capture-menu-preview="):
			preview_path = ProjectSettings.globalize_path(argument.trim_prefix("--capture-menu-preview="))
	output = ProjectSettings.globalize_path(output)
	check(DirAccess.dir_exists_absolute(output) or DirAccess.make_dir_recursive_absolute(output) == OK, "Could not create the isolated menu output directory")
	check(FileAccess.file_exists(source) and FileAccess.file_exists(cove_source), "A real saved boulder learner and protected cove source are required")
	if failures:
		quit(1)
		return
	var unique := "%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()]
	protected_source = output.path_join("source-boulder-%s.bin" % unique)
	protected_cove = output.path_join("source-cove-%s.bin" % unique)
	checkpoint = output.path_join("menu-boulder-%s.bin" % unique)
	check(DirAccess.copy_absolute(ProjectSettings.globalize_path(source), protected_source) == OK, "Could not snapshot the live boulder save")
	check(DirAccess.copy_absolute(ProjectSettings.globalize_path(cove_source), protected_cove) == OK, "Could not protect the cove source")
	check(DirAccess.copy_absolute(protected_source, checkpoint) == OK, "Could not create the private menu learner")
	if failures:
		quit(1)
		return
	var source_hash := FileAccess.get_sha256(protected_source)
	var cove_hash := FileAccess.get_sha256(protected_cove)
	node_added.connect(isolate)
	var router = root.get_node("DemoLaunch")
	for release in [false, true]:
		for cuda_ready in [false, true]:
			for has_save in [false, true]:
				for cpu_requested in [false, true]:
					var expected := "pond_boulder" if not release and cuda_ready and has_save and not cpu_requested else "pond_feeding"
					check(router.select_primary_pond(release, cuda_ready, has_save, cpu_requested) == expected, "Primary pond fallback policy is wrong")
	check(router.primary_pond() == "pond_boulder", "This local CUDA project does not choose the latest saved boulder pond")
	check(router.SCENES.get("pond_boulder", "") == "res://pond/boulder_pond.tscn", "Latest pond route points to an older environment")
	change_scene_to_file("res://launcher/launcher.tscn")
	await settle()
	check(current_scene.rows[0].preview.texture != null and current_scene.rows[0].preview.texture.resource_path == "res://launcher/assets/boulder-pond.png", "Menu preview still shows an older pond")
	var dimensions_list := [Vector2i(1280,720), Vector2i(768,1024), Vector2i(390,844), Vector2i(844,390)]
	for dimensions in dimensions_list:
		root.size = dimensions
		await settle()
		var menu = current_scene
		check(menu.buttons.size() == 2, "Local pond menu must offer learning and the secondary rock ledge view")
		var ledge: Button = menu.find_child("pond_overhang_watch", true, false)
		check(ledge != null and ledge.text == "View rock ledge" and ledge.size.y >= 44, "The new physical shelter cannot be opened from the local menu")
		var action: Button = menu.find_child("pond_boulder_watch", true, false)
		check(action != null and action.text == "Open pond", "Open pond still points to the older feeding environment")
		if action == null:
			quit(1)
			return
		if dimensions.x >= 840 and dimensions.y <= 400:
			check(action.get_global_rect().end.y <= dimensions.y + 1, "Compact landscape Open pond is below the fold before scrolling")
		var scroll: ScrollContainer = menu.find_child("DemoScroll", true, false)
		scroll.ensure_control_visible(action)
		await settle()
		check(action.size.y >= 44 and action.get_global_rect().position.x >= 0 and action.get_global_rect().end.x <= dimensions.x + 1, "Primary menu action clips or is too small")
		check(action.get_global_rect().end.y <= dimensions.y + 1, "Primary menu action cannot be reached by scrolling")
		check(root.gui_get_focus_owner() == action, "Open pond lost its initial keyboard focus")
		for removed in ["pond_experiments_train", "guide_train", "cartpole_watch", "racing_watch"]:
			check(menu.find_child(removed, true, false) == null, "Removed menu action remains: " + removed)
		for data in menu.rows:
			check(data.row.get_global_rect().end.x <= dimensions.x + 1, "Menu row overflows horizontally")
		await capture("menu-%dx%d.png" % [dimensions.x, dimensions.y])
	root.size = Vector2i(1280,720)
	await settle()
	await activate(current_scene.find_child("pond_boulder_watch", true, false))
	var pond = current_scene
	check(pond.name == "BoulderPond", "Keyboard Open pond did not launch the boulder environment")
	if pond.name != "BoulderPond":
		quit(1)
		return
	var pond_ready: bool = await ready_and_paused(pond)
	if not pond_ready:
		pond.lab.stop()
		quit(1)
		return
	check(pond.boulder_checkpoint == checkpoint and pond.stats.layout_id == 2 and pond.stats.policy_inputs == 55, "Menu opened the wrong habitat or save")
	check(pond.stats.resumed and pond.stats.loaded_update > 0 and pond.stats.background_swimmers == 1024, "Menu restarted or substituted the saved learner")
	check(not pond.find_child("DropFoodAtCenter", true, false).visible and not pond.feed_surface.visible, "Automatic-food pond still offers manual feeding")
	check(pond.find_child("PondExperiments", true, false) == null and pond.current_switch.visible and pond.pause_button.visible, "Latest pond controls are misleading")
	for tick in 20:
		pond.state = pond.lab.step()
	pond.previous = pond.state.duplicate(true)
	pond._poll()
	for dimensions in dimensions_list:
		root.size = dimensions
		await settle()
		pond.sidebar_scroll.scroll_vertical = 0
		check(pond.viewport_container.size.y >= 100 and pond.current_strength.get_global_rect().end.x <= dimensions.x + 1, "Latest pond viewport or current controls clip")
		check(pond.camera_button.size.y >= 44 and pond.pause_button.size.y >= 44, "Latest pond actions are too small")
		check(pond.curve.status_caption.contains("live actor") and pond.curve.plot_caption.contains("This visit"), "Latest GUI still labels a frozen or older policy")
		await capture("pond-%dx%d.png" % [dimensions.x, dimensions.y])
	root.size = Vector2i(1280,720)
	await settle()
	if not preview_path.is_empty() and DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		check(pond.viewport.get_texture().get_image().save_png(preview_path) == OK, "Actual boulder viewport preview could not be captured")
	await activate(pond.details_button)
	check(pond.details_group.visible and pond.training_stats.text.to_lower().contains("boulder"), "AI details do not describe the latest environment")
	await capture("pond-details.png")
	await activate(pond.details_button)
	var previous_state: Dictionary = pond.lab.get_state().duplicate(true)
	var previous_update: int = pond.stats.update
	var previous_context: int = pond.stats.context_version
	var previous_actor: int = pond.stats.adopted_update
	var previous_lab: RefCounted = pond.lab
	pond.set_process(false)
	pond.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu" and not previous_lab.get_stats().get("training_ready", false), "Pond menu did not save and stop the old session")
	check(root.title == "Tadpole Pond · CudaRL", "Menu retains an old environment window title")
	await activate(current_scene.find_child("pond_boulder_watch", true, false))
	pond = current_scene
	check(pond.name == "BoulderPond", "Reopening the menu returned to an older environment")
	if pond.name != "BoulderPond":
		quit(1)
		return
	pond_ready = await ready_and_paused(pond)
	if not pond_ready:
		quit(1)
		return
	check(pond.stats.resumed and pond.stats.loaded_update >= previous_update and pond.stats.context_version == previous_context, "Menu round trip lost learner progress or water settings")
	check(pond.state.position == previous_state.position and pond.state.orientation == previous_state.orientation and pond.state.foods == previous_state.foods and pond.state.elapsed == previous_state.elapsed, "Menu round trip changed the controlled physical snapshot: elapsed %.3f → %.3f" % [previous_state.elapsed, pond.state.elapsed])
	check(pond.stats.adopted_update == previous_actor, "Menu round trip substituted a different visible actor")
	check(FileAccess.get_sha256(protected_source) == source_hash and FileAccess.get_sha256(protected_cove) == cove_hash, "Navigation modified a protected source")
	pond.set_process(false)
	pond.lab.stop()
	var summary := {"checks":checks,"failures":failures,"source_sha256":source_hash,"loaded_update":pond.stats.loaded_update,"checkpoint":checkpoint,"preview":preview_path,"sizes":dimensions_list}
	var report := FileAccess.open(output.path_join("summary.json"), FileAccess.WRITE)
	report.store_string(JSON.stringify(summary,"\t"))
	report.close()
	print("BOULDER_MENU_SMOKE " + JSON.stringify(summary))
	quit(0 if failures == 0 else 1)
