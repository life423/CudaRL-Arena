extends SceneTree
## Camera and interaction validation only; no native physical integration.
var view: Node
var checks: Array[Dictionary] = []
var failures: Array[String] = []
var output := ""
var requested_size := Vector2i(1280,800)

func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--test-output="): output=arg.trim_prefix("--test-output=")
		if arg.begins_with("--test-size="):
			var parts:=arg.trim_prefix("--test-size=").split("x")
			requested_size=Vector2i(int(parts[0]),int(parts[1]))
	root.size=requested_size
	view=load("res://main.tscn").instantiate()
	root.add_child.call_deferred(view)
	_run.call_deferred()

func check(message: String, good: bool) -> void:
	checks.append({"check":message,"passed":good})
	if not good: failures.append(message)

func settle() -> void:
	await process_frame
	await process_frame

func contrast(first: Color, second: Color) -> float:
	var a:=first.srgb_to_linear()
	var b:=second.srgb_to_linear()
	var x:=.2126*a.r+.7152*a.g+.0722*a.b
	var y:=.2126*b.r+.7152*b.g+.0722*b.b
	return (maxf(x,y)+.05)/(minf(x,y)+.05)

func capture(suffix: String) -> void:
	if output.is_empty(): return
	await RenderingServer.frame_post_draw
	check("Screenshot saved: "+suffix,root.get_texture().get_image().save_png(output.get_basename()+suffix+".png")==OK)

func _run() -> void:
	await settle()
	view.set_process(false)
	check("Actual requested viewport",root.size==requested_size)
	check("Saved motor loaded and paused",view.loaded and not view.running)
	check("Startup uses no physical controls",view.physical_substeps==0 and int(view.snapshot.get("controls",-1))==0)
	check("Readable aerial creature view is the default",view.pond_overview and view.world.overview and view.view_button.text=="Side view")
	check("Pond respects native horizontal dimensions",is_equal_approx(view.world.half_width,float(view.world.geometry.arena_width_m)/(2*float(view.world.geometry.scale_m))) and is_equal_approx(view.world.half_depth,float(view.world.geometry.arena_depth_m)/(2*float(view.world.geometry.scale_m))))
	for control in [view.play_button,view.restart_button,view.task,view.details_button,view.view_button]:
		var box: Rect2=control.get_global_rect()
		check("Usable visible control: "+control.text,box.size.x>=44 and box.size.y>=44 and box.position.x>=0 and box.end.x<=root.size.x+1 and box.position.y>=0 and box.end.y<=root.size.y+1)
	var area: Rect2=view.world.viewport_area
	var head_at: Vector2=view.world.camera.unproject_position(view.world.links[0].position)
	var tail_at: Vector2=view.world.camera.unproject_position(view.world.links[2].position)
	check("Starting creature has readable screen size",head_at.distance_to(tail_at)>=100)
	var projected_bounds_ok:=true
	for i in range(3):
		var link_geometry: Dictionary=view.world.geometry.links[i]
		var half_size:=Vector3(float(link_geometry.length)/2,float(link_geometry.half_height),float(link_geometry.half_width))/float(view.world.geometry.scale_m)
		for x in [-1,1]:
			for y in [-1,1]:
				for z in [-1,1]:
					var point: Vector3=view.world.links[i].position+view.world.links[i].quaternion*(half_size*Vector3(x,y,z))
					projected_bounds_ok=projected_bounds_ok and area.has_point(view.world.camera.unproject_position(point))
	check("Full physical creature fits aerial view",projected_bounds_ok)
	check("View control does not overlap details",not view.view_button.get_global_rect().intersects(view.details_button.get_global_rect()))
	var header_box: Rect2=view.identity_panel.get_global_rect()
	var background: Color=view.identity_panel.get_theme_stylebox("panel").bg_color
	var title_contrast:=contrast(view.brand.get_theme_color("font_color"),background)
	var status_contrast:=contrast(view.status.get_theme_color("font_color"),background)
	check("Header background is opaque",background.a==1)
	check("Title text has at least 4.5 to 1 contrast",title_contrast>=4.5)
	check("Status text has at least 4.5 to 1 contrast",status_contrast>=4.5)
	check("Header text fits its surface",header_box.encloses(view.brand.get_global_rect()) and header_box.encloses(view.status.get_global_rect()))
	check("Header does not overlap camera or Details",not header_box.intersects(view.view_button.get_global_rect()) and not header_box.intersects(view.details_button.get_global_rect()))
	var poses: Array=view.snapshot.links.duplicate(true)
	await capture("-aerial")
	var old_size: float=view.world.camera.size
	var zoom_event:=InputEventKey.new()
	zoom_event.pressed=true
	zoom_event.keycode=KEY_PLUS
	view._unhandled_key_input(zoom_event)
	check("Keyboard zooms in without stepping",view.world.camera.size<old_size and view.physical_substeps==0)
	zoom_event.keycode=KEY_MINUS
	view._unhandled_key_input(zoom_event)
	check("Keyboard zooms back out",is_equal_approx(view.world.camera.size,old_size))
	var wheel:=InputEventMouseButton.new()
	wheel.pressed=true
	wheel.button_index=MOUSE_BUTTON_WHEEL_UP
	wheel.position=area.get_center()
	view._unhandled_input(wheel)
	check("Wheel zooms in",view.world.camera.size<old_size)
	var pinch:=InputEventMagnifyGesture.new()
	pinch.position=area.get_center()
	pinch.factor=1.0/1.16
	view._unhandled_input(pinch)
	check("Pinch zoom returns to initial size",is_equal_approx(view.world.camera.size,old_size))
	view._toggle_pond_view()
	await settle()
	check("Side view keeps the creature visible",not view.world.overview and not view.pond_overview and view.view_button.text=="Aerial view")
	check("Creature fits close view",area.has_point(view.world.camera.unproject_position(view.world.links[0].position)))
	for i in range(3):
		check("Rendered link stays native: "+str(i),view.world.links[i].position.is_equal_approx(poses[i].position) and view.world.links[i].quaternion.is_equal_approx(poses[i].orientation))
	await capture("-close")
	view._open_details()
	await settle()
	check("Drawer isolates camera control focus",view.drawer.visible and view.view_button.focus_mode==Control.FOCUS_NONE and root.gui_get_focus_owner()==view.close_button)
	var drawer_size: float=view.world.camera.size
	zoom_event.keycode=KEY_PLUS
	view._unhandled_key_input(zoom_event)
	view._unhandled_input(wheel)
	check("Details blocks camera zoom",is_equal_approx(view.world.camera.size,drawer_size))
	check("Pond and saved learning scope are explained",view.drawer_text.get_parsed_text().contains("Your pond") and view.drawer_text.get_parsed_text().contains("does not run learning updates"))
	view._close_details()
	check("Closing restores keyboard access",view.view_button.focus_mode==Control.FOCUS_ALL and root.gui_get_focus_owner()==view.details_button)
	view._toggle_pond_view()
	check("Overview returns without resetting body",view.world.overview and int(view.snapshot.get("controls",-1))==0)
	check("All validation used zero physical substeps",view.physical_substeps==0 and int(view.preview.call("get_state").get("substeps",-1))==0)
	var result: Dictionary={"passed":failures.is_empty(),"size":[root.size.x,root.size.y],"checks":checks,"failures":failures,"title_contrast":title_contrast,"status_contrast":status_contrast,"native_head_tail_pixels":head_at.distance_to(tail_at),"physical_controls":0,"physical_substeps":view.physical_substeps}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(result,"  "))
	print(JSON.stringify(result))
	quit(0 if failures.is_empty() else 3)
