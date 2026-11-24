extends SceneTree
## Geometry and presentation checks only. Never calls a native physics step.
var view: Node
var output := ""
var requested_size := Vector2i(1280,800)
var failures: Array[String]=[]
var checks: Array[Dictionary]=[]
var maximum_height_error_m := 0.0
var maximum_normal_error := 0.0
var maximum_packed_normal_reference_error := 0.0
var maximum_rim_error_model := 0.0

func check(label: String, good: bool) -> void:
	checks.append({"check":label,"passed":good})
	if not good: failures.append(label)

func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--test-output="): output=arg.trim_prefix("--test-output=")
		if arg.begins_with("--test-size="):
			var parts:=arg.trim_prefix("--test-size=").split("x")
			requested_size=Vector2i(int(parts[0]),int(parts[1]))
	view=load("res://main.tscn").instantiate()
	view.set_process(false)
	root.add_child.call_deferred(view)
	_run.call_deferred()

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
	root.title="Tadpole Pond native geometry QA"
	root.size=requested_size
	await settle()
	view._layout()
	await settle()
	check("Requested viewport rendered",root.size==requested_size)
	check("View process remains disabled",not view.is_processing())
	check("Saved motor loads paused with zero integration",view.loaded and not view.running and view.physical_substeps==0 and int(view.snapshot.get("controls",-1))==0)
	if not view.loaded or view.world.landscape==null:
		_finish()
		return
	var landscape: Node=view.world.landscape
	var descriptor: Dictionary=landscape.geometry_descriptor
	check("Shared native geometry is ready",landscape.geometry_ready and str(landscape.geometry_issue).is_empty())
	check("Geometry version and SI units are explicit",int(descriptor.get("version",0))==1 and descriptor.get("units","")=="metres")
	var water: Vector2=descriptor.get("water_radii_m",Vector2.ZERO)
	check("Accepted visible pond dimensions remain unchanged",absf(water.x-.52)<.000001 and absf(water.y-.34)<.000001)
	check("Shared terrain contact is separate from learned bank sensing",not descriptor.get("geometry_only",true) and descriptor.get("used_by_contact",false) and not descriptor.get("used_by_policy",true))
	check("Smaller legacy guard is absent",descriptor.get("active_motion_domain","")=="shared_pond_terrain" and not descriptor.has("active_guard_radii_m"))
	check("Terrain and water are batch generated",int(landscape.native_query_batches)<20769 and int(landscape.native_geometry_points)>=20769*2)
	_validate_meshes(landscape)
	var initial_poses: Array=view.snapshot.links.duplicate(true)
	check("Aerial creature view remains the default",view.pond_overview and view.world.overview and view.view_button.text=="Side view")
	var head_at: Vector2=view.world.camera.unproject_position(view.world.links[0].position)
	var tail_at: Vector2=view.world.camera.unproject_position(view.world.links[2].position)
	check("Default creature remains readable",head_at.distance_to(tail_at)>=100)
	var before_zoom: float=view.world.camera.size
	view.world.zoom_by(1.16)
	check("Manual zoom preserves body and native controls",view.world.camera.size>before_zoom and view.snapshot.links==initial_poses and int(view.preview.call("get_state").get("controls",-1))==0)
	view.world.zoom_by(1.0/1.16)
	check("Manual zoom returns to original scale",is_equal_approx(view.world.camera.size,before_zoom))
	for control in [view.play_button,view.restart_button,view.task,view.details_button,view.view_button]:
		var box: Rect2=control.get_global_rect()
		check("44px visible control: "+control.text,box.size.x>=44 and box.size.y>=44 and box.position.x>=0 and box.end.x<=root.size.x+1 and box.position.y>=0 and box.end.y<=root.size.y+1)
	var panel_color: Color=view.identity_panel.get_theme_stylebox("panel").bg_color
	check("Opaque header retains readable contrast",panel_color.a==1 and contrast(view.brand.get_theme_color("font_color"),panel_color)>=4.5 and contrast(view.status.get_theme_color("font_color"),panel_color)>=4.5)
	check("Native DLL and CPU shoreline evidence are verified",view.shoreline_qualified and not view.shoreline_binding.is_empty() and not view.shoreline_proof.is_empty())
	check("Actual CUDA agreement receipt is verified",view.cuda_qualified)
	if not view.shoreline_qualified:
		_finish()
		return
	var binding: Dictionary=view.shoreline_binding.duplicate(true)
	var proof: Dictionary=view.shoreline_proof.duplicate(true)
	var dll_hash:=FileAccess.get_sha256(view.NATIVE_DLL_PATH)
	var cpu_hash:=FileAccess.get_sha256(view.SHORELINE_CPU_PATH)
	var bad_binding: Dictionary=binding.duplicate(true)
	bad_binding.dll_sha256="0".repeat(64)
	check("Wrong native DLL binding is rejected",not view._shoreline_validation_issue(bad_binding,proof,dll_hash,cpu_hash,descriptor).is_empty())
	var gpu_proof: Dictionary=view._dictionary_file(view.SHORELINE_GPU_PATH)
	var gpu_hash: String=FileAccess.get_sha256(view.SHORELINE_GPU_PATH)
	check("Wrong CUDA receipt binding is rejected",not view._cuda_verification_valid(binding,gpu_proof,"0".repeat(64)))
	var failed_gpu: Dictionary=gpu_proof.duplicate(true)
	failed_gpu.passed=false
	check("Failed CUDA receipt cannot claim agreement",not view._cuda_verification_valid(binding,failed_gpu,gpu_hash))
	bad_binding=binding.duplicate(true)
	bad_binding.cpu_source.sha256="0".repeat(64)
	check("Wrong CPU evidence binding is rejected",not view._shoreline_validation_issue(bad_binding,proof,dll_hash,cpu_hash,descriptor).is_empty())
	var bad_proof: Dictionary=proof.duplicate(true)
	bad_proof.passed=false
	check("A failed CPU check cannot qualify the app",not view._shoreline_validation_issue(binding,bad_proof,dll_hash,cpu_hash,descriptor).is_empty())
	bad_proof=proof.duplicate(true)
	bad_proof.completed_cases=35
	check("Incomplete case counts cannot qualify the app",not view._shoreline_validation_issue(binding,bad_proof,dll_hash,cpu_hash,descriptor).is_empty())
	bad_proof=proof.duplicate(true)
	bad_proof.protocol_id="shoreline-product-v3-2026-11-23"
	check("A previous shoreline protocol cannot qualify the app",not view._shoreline_validation_issue(binding,bad_proof,dll_hash,cpu_hash,descriptor).is_empty())
	var old_descriptor: Dictionary=descriptor.duplicate(true)
	old_descriptor.runtime_version=3
	check("A previous installed runtime cannot inherit qualification",not view._shoreline_validation_issue(binding,proof,dll_hash,cpu_hash,old_descriptor).is_empty())
	var old_arithmetic: Dictionary=descriptor.duplicate(true)
	old_arithmetic.query_arithmetic_version=1
	check("Previous query arithmetic cannot inherit qualification",not view._shoreline_validation_issue(binding,proof,dll_hash,cpu_hash,old_arithmetic).is_empty())
	check("Failed corrected qualification remains explicit",view._shoreline_update_issue({"schema":1,"runtime_version":4,"status":"failed","completed_cases":28}).contains("did not pass (28 of 36 cases completed)"))
	check("Verification errors retain readable contrast",contrast(view.VERIFICATION_ERROR,view.SURFACE)>=4.5)
	await capture("-default")
	view._open_details()
	var details: String=view.drawer_text.get_parsed_text()
	check("Details explains verified shared native terrain",details.contains("Shared shoreline verified") and details.contains("same terrain") and details.contains("legacy limit has been removed"))
	check("Details distinguishes actual CPU and CUDA verdicts",details.contains("CPU physical checks: 36 / 36 passed") and details.contains("CUDA agreement: 8 / 8 matched cases"))
	check("Details distinguishes contact from learned turning",details.contains("not a learned turn") and details.contains("Water exit is outside"))
	check("Details retains frozen-policy and navigation limits",details.contains("does not train") and details.contains("Learned pond navigation is still unqualified") and details.contains("Original saved motor"))
	var candidate: Dictionary=view._dictionary_file(view.CANDIDATE_STUDY_PATH)
	check("Details distinguishes original playing motor from recorded candidate",details.contains("Playing original motor 336") and details.contains("Recorded candidate") and details.contains("%d PPO updates" % int(candidate.completed_updates)) and not details.contains("No new candidate is installed by this status update"))
	var bad_candidate_binding: Dictionary=view.shoreline_binding.duplicate(true)
	bad_candidate_binding.candidate_source.sha256="0".repeat(64)
	check("Changed candidate receipt shows an honest unavailable fallback",view._candidate_study_status(bad_candidate_binding).contains("Candidate assessment is unavailable"))
	check("Latest shoreline status precedes reference assessment",details.find("Shared shoreline verified")>=0 and details.find("Shared shoreline verified")<details.find("Saved motor reference assessment"))
	check("Details exposes current clearance and whole-body movement",details.contains("Right now") and details.contains("Clear of the bed now") and details.contains("Certified whole-body clearance") and details.contains("Horizontal COM speed") and details.contains("Requested:") and details.contains("Tilt:"))
	var live_reading: Dictionary=view.snapshot.duplicate(true)
	live_reading.whole_body_clearance_lower_m=.005968
	live_reading.whole_body_clearance_upper_m=.005978
	live_reading.minimum_lifetime_clearance_m=-.000118
	live_reading.contact_damping_loss_J=7.5e-8
	live_reading.horizontal_speed_m_s=.000568
	live_reading.terminal_velocity=Vector3(.008,0,0)
	live_reading.tilt_deg=117.3
	live_reading.depth_m=.03863
	var current_text: String=view._current_physics_text(live_reading)
	check("Current clear status ignores past contact and lifetime minimum",current_text.contains("Clear of the bed now") and current_text.contains("5.968 to 5.978") and current_text.contains("0.57 mm/s") and current_text.contains("8.00 mm/s") and current_text.contains("117.3°"))
	live_reading.whole_body_clearance_lower_m=-.000118
	live_reading.whole_body_clearance_upper_m=-.000108
	check("Current compressed interval reports bed contact",view._current_physics_text(live_reading).contains("Bed contact now"))
	live_reading.whole_body_clearance_upper_m=.000001
	check("A zero-straddling interval reports uncertainty",view._current_physics_text(live_reading).contains("Possible bed contact now"))
	live_reading.whole_body_query_certified=false
	check("An uncertified query cannot claim current clearance",view._current_physics_text(live_reading).contains("Current bed clearance is unavailable"))
	live_reading.whole_body_query_certified=true
	live_reading.query_arithmetic_version=1
	check("Old query arithmetic cannot claim certified current clearance",view._current_physics_text(live_reading).contains("Current bed clearance is unavailable"))
	check("Details focus remains modal",root.gui_get_focus_owner()==view.close_button and view.play_button.focus_mode==Control.FOCUS_NONE)
	await settle()
	await capture("-details-verified")
	view.shoreline_qualified=false
	view.shoreline_issue="The shoreline evidence has changed or is missing. Verification cannot be confirmed."
	view._append_shoreline_details(true)
	var unavailable_text: String=view.drawer_text.get_parsed_text()
	check("Unavailable evidence shows a clear error without claiming success",unavailable_text.contains("Shoreline verification unavailable") and unavailable_text.contains("cannot be confirmed") and not unavailable_text.contains("Shared shoreline verified"))
	check("Repeated Details rendering preserves its section title",unavailable_text.begins_with("Pond world"))
	await settle()
	await capture("-details-unavailable")
	view._read_shoreline_status()
	view._close_details()
	check("Closing Details returns keyboard focus",root.gui_get_focus_owner()==view.details_button)
	view.show_measurements=true
	view.measurement_toggle.set_pressed_no_signal(true)
	view._layout()
	view._refresh()
	await settle()
	# A deliberate manual overview framing verifies the unchanged full pond.
	# This does not alter the production default camera or any physical state.
	var scale_m: float=view.world.geometry.scale_m
	var span_y:=maxf(water.y*2/scale_m*1.5,water.x*2/scale_m*1.32*float(root.size.y)/float(root.size.x))
	view.world.set_zoom(span_y/view.world.camera.size)
	var overview_zoom: float=view.world.zoom_factor
	await settle()
	# Default framing is captured above; this overview is used only to check
	# that a physical-stop message preserves the user's chosen manual zoom.
	# Synthetic presentation-only physical-stop message; no historical actor,
	# reset, request or native integration is injected into the product.
	var initial_native: Dictionary=view.preview.call("get_state")
	view.snapshot["error"]="Paused: terrain penetration exceeded the physical tolerance"
	view.error=str(view.snapshot.error)
	view._refresh()
	await settle()
	check("Physical-stop message remains specific",view.notice.text==view.error and view.status.text.begins_with("Stopped"))
	check("Compact stop notice reserves space",view.notice.size.y>=48 and view.world.viewport_area.end.y<=view.notice.position.y-15)
	check("Stopped world retains manual zoom",view.world.body_ready and is_equal_approx(view.world.zoom_factor,overview_zoom))
	await capture("-physical-stop-ui")
	var native: Dictionary=view.preview.call("get_state")
	check("UI fixture never changes the native world",native==initial_native and int(native.get("controls",-1))==0 and int(native.get("substeps",-1))==0 and view.physical_substeps==0 and not view.is_processing())

	_finish()

func _validate_meshes(landscape: Node) -> void:
	var scale_m: float=landscape.scale_m
	var terrain: MeshInstance3D=landscape.get_node("OrganicPondBasin")
	var arrays: Array=terrain.mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array=arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array=arrays[Mesh.ARRAY_NORMAL]
	check("Terrain keeps its original 20769 vertices",vertices.size()==20769 and normals.size()==20769)
	var probes:=PackedVector2Array()
	for vertex in vertices: probes.append(Vector2(vertex.x*scale_m,vertex.z*scale_m))
	var samples: Dictionary=view.preview.call("sample_pond_surface",probes)
	check("Native surface validates every rendered terrain point",bool(samples.get("valid",false)) and samples.get("heights_m",PackedFloat64Array()).size()==vertices.size())
	if bool(samples.get("valid",false)):
		var heights: PackedFloat64Array=samples.heights_m
		var expected_normals: PackedVector3Array=samples.normals
		for i in vertices.size():
			maximum_height_error_m=maxf(maximum_height_error_m,absf(vertices[i].y*scale_m-heights[i]))
			maximum_normal_error=maxf(maximum_normal_error,normals[i].distance_to(expected_normals[i]))
	check("Rendered heights agree with native geometry within 1 micron",maximum_height_error_m<=.000001)
	# ArrayMesh stores normals in the renderer's packed format. Verify the
	# analytic normals against a separate packing round-trip without weakening
	# the geometry threshold or hiding the raw native-to-storage discrepancy.
	var original_samples:=PackedVector2Array()
	for ring in range(129):
		var radius:=float(ring)/72*1.2 if ring<=72 else 1.2+float(ring-72)/56*10.8
		for spoke in range(161): original_samples.append(Vector2(float(spoke)/160*TAU,radius))
	var original_outline: Dictionary=view.preview.call("sample_pond_outline",original_samples)
	var original_probes:=PackedVector2Array()
	for at in original_outline.positions_m: original_probes.append(Vector2(at.x,at.z))
	var original_surface: Dictionary=view.preview.call("sample_pond_surface",original_probes)
	var native_normals: PackedVector3Array=original_surface.normals
	var codec:=SurfaceTool.new()
	codec.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in vertices.size():
		codec.set_normal(native_normals[i])
		codec.add_vertex(vertices[i])
	var source_arrays: Array=codec.commit_to_arrays()
	var source_normals: PackedVector3Array=source_arrays[Mesh.ARRAY_NORMAL]
	var source_error:=0.0
	for i in vertices.size(): source_error=maxf(source_error,source_normals[i].distance_to(native_normals[i]))
	check("Unpacked terrain source uses native analytic normals",source_error<=.00001)
	var codec_mesh:=codec.commit()
	var codec_arrays: Array=codec_mesh.surface_get_arrays(0)
	var packed_reference: PackedVector3Array=codec_arrays[Mesh.ARRAY_NORMAL]
	for i in vertices.size(): maximum_packed_normal_reference_error=maxf(maximum_packed_normal_reference_error,normals[i].distance_to(packed_reference[i]))
	check("Rendered normals match native analytic normals through renderer packing",maximum_packed_normal_reference_error<=.00001)
	var rim_queries:=PackedVector2Array()
	for i in range(161): rim_queries.append(Vector2(float(i)/160*TAU,1))
	var outline: Dictionary=view.preview.call("sample_pond_outline",rim_queries)
	var rim_m: PackedVector3Array=outline.get("positions_m",PackedVector3Array())
	var water_arrays: Array=landscape.surface.mesh.surface_get_arrays(0)
	var water_vertices: PackedVector3Array=water_arrays[Mesh.ARRAY_VERTEX]
	check("Water uses the native 161-point rim",bool(outline.get("valid",false)) and rim_m.size()==161 and water_vertices.size()==480)
	if rim_m.size()==161 and water_vertices.size()==480:
		for spoke in range(160):
			maximum_rim_error_model=maxf(maximum_rim_error_model,water_vertices[spoke*3].length())
			maximum_rim_error_model=maxf(maximum_rim_error_model,water_vertices[spoke*3+1].distance_to(rim_m[spoke+1]/scale_m))
			maximum_rim_error_model=maxf(maximum_rim_error_model,water_vertices[spoke*3+2].distance_to(rim_m[spoke]/scale_m))
	check("Rendered water rim agrees with native outline",maximum_rim_error_model<=.00001)

func _finish() -> void:
	var native: Dictionary={} if view.preview==null else view.preview.call("get_state")
	var report: Dictionary={"passed":failures.is_empty(),"checks":checks,"failures":failures,"size":[root.size.x,root.size.y],"maximum_height_error_m":maximum_height_error_m,"maximum_normal_storage_error":maximum_normal_error,"maximum_packed_normal_reference_error":maximum_packed_normal_reference_error,"maximum_rim_error_model":maximum_rim_error_model,"shoreline_qualified":view.shoreline_qualified,"native_dll_sha256":FileAccess.get_sha256(view.NATIVE_DLL_PATH),"shoreline_cpu_sha256":FileAccess.get_sha256(view.SHORELINE_CPU_PATH),"synthetic_stop_message_only":true,"physical_controls":int(native.get("controls",0)),"physical_substeps":int(native.get("substeps",0)),"CUDA_kernels":0,"optimizer_steps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify({"passed":report.passed,"failures":failures,"checks":checks.size(),"size":report.size,"physical_controls":report.physical_controls,"physical_substeps":report.physical_substeps,"maximum_height_error_m":maximum_height_error_m,"maximum_normal_error":maximum_normal_error}))
	quit(0 if failures.is_empty() else 3)
