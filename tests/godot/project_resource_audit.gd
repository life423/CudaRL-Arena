extends SceneTree

# Loads resources in the actual project/autoload context. It never instantiates
# an audited script, scene, native lab, learner or physics world.
const NATIVE_APIS := {
	"DeepPondLab": ["get_state", "get_stats", "get_error", "stop", "start_feeding_learning", "start_boulder_learning", "watch_overhang"],
	"DeepWaterLab": ["get_state", "get_stats", "get_error", "stop", "start_training"],
	"TadpoleLab": ["get_state", "get_stats", "get_error", "stop"],
	"PpoLab": ["get_stats", "get_error", "stop", "watch_policy"],
	"RacingLab": ["get_stats", "get_error", "stop", "watch_policy"],
	"CudaDevice": ["is_available", "describe"],
	"MazeArena": [],
	"NetworkTrainer": [],
	"MotorControlLab": [],
	"MotorV2Lab": [],
}
var failures: Array[String] = []
var resources: Array[Dictionary] = []
var graph: Dictionary = {}
var source_records: Dictionary = {}
var colors: Dictionary = {}
var cycles: Array = []
var include_tests := false
var output_path := ""
var native_report: Dictionary = {}

func _initialize() -> void:
	_run.call_deferred()

func _fail(message: String) -> void:
	failures.append(message)
	push_error("PROJECT_RESOURCE_AUDIT: " + message)

func _paths(directory: String, result: Array[String]) -> void:
	var folder := DirAccess.open(directory)
	if folder == null:
		_fail("Cannot read directory: " + directory)
		return
	var files := folder.get_files()
	files.sort()
	for filename: String in files:
		if filename.get_extension() in ["gd", "tscn", "gdshader", "tres"]:
			result.append(directory.path_join(filename))
	var directories := folder.get_directories()
	directories.sort()
	for child: String in directories:
		if not child.begins_with(".") and child != "bin":
			_paths(directory.path_join(child), result)

func _matches(pattern: String, source: String) -> Array[String]:
	var values: Array[String] = []
	var expression := RegEx.new()
	if expression.compile(pattern) != OK:
		_fail("Audit regular expression could not compile")
		return values
	for match_value: RegExMatch in expression.search_all(source):
		values.append(match_value.get_string(1))
	return values

func _read_source(path: String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		_fail("Cannot read resource source: " + path)
		return
	var source := file.get_as_text()
	var dependencies: Array[String] = []
	var base_paths := _matches("(?m)^extends\\s+[\"']([^\"']+)[\"']", source)
	dependencies.append_array(base_paths)
	dependencies.append_array(_matches("preload\\s*\\(\\s*[\"']([^\"']+)[\"']", source))
	if path.get_extension() in ["tscn", "tres"]:
		dependencies.append_array(_matches("\\[ext_resource[^\\]]*path=\"([^\"]+)\"", source))
	var unique: Array[String] = []
	for dependency: String in dependencies:
		if not dependency.begins_with("res://"):
			dependency = path.get_base_dir().path_join(dependency).simplify_path()
		if not unique.has(dependency):
			unique.append(dependency)
		if not FileAccess.file_exists(dependency):
			_fail("Missing static dependency: %s -> %s" % [path, dependency])
	graph[path] = unique
	var static_native: Array[String] = []
	var dynamic_native: Array[String] = []
	for native_name: String in NATIVE_APIS:
		if source.contains(native_name + ".new("):
			static_native.append(native_name)
		if source.contains("\"" + native_name + "\"") and source.contains("ClassDB."):
			dynamic_native.append(native_name)
	source_records[path] = {
		"declared_base_script": base_paths[0] if not base_paths.is_empty() else "",
		"declares_ready": not _matches("(?m)^func\\s+(_ready)\\s*\\(", source).is_empty(),
		"static_native_constructors": static_native,
		"dynamic_native_names": dynamic_native,
		"references_autoload": source.contains("DemoLaunch"),
	}

func _visit(path: String, trail: Array[String]) -> void:
	var color := int(colors.get(path, 0))
	if color == 1:
		var cycle: Array[String] = trail.duplicate()
		cycle.append(path)
		cycles.append(cycle)
		_fail("Static preload/inheritance cycle: " + " -> ".join(cycle))
		return
	if color == 2:
		return
	colors[path] = 1
	var next_trail: Array[String] = trail.duplicate()
	next_trail.append(path)
	for dependency: String in graph.get(path, []):
		if graph.has(dependency):
			_visit(dependency, next_trail)
	colors[path] = 2

func _load_resource(path: String) -> void:
	var record: Dictionary = {"path": path, "extension": path.get_extension(), "loaded": false}
	if source_records.has(path):
		record.merge(source_records[path])
	var resource: Resource = ResourceLoader.load(path, "", ResourceLoader.CACHE_MODE_IGNORE)
	if resource == null:
		_fail("ResourceLoader failed: " + path)
		resources.append(record)
		return
	record.loaded = true
	record.resource_type = resource.get_class()
	if resource is Script:
		var script: Script = resource
		record.compiled = script.can_instantiate()
		if not script.can_instantiate():
			_fail("Script cannot instantiate after compilation: " + path)
		var chain: Array[String] = []
		var cursor: Script = script
		var seen: Array[String] = []
		var ready_layers := 0
		while cursor != null:
			var script_path := cursor.resource_path
			if seen.has(script_path):
				_fail("Loaded base-script cycle: " + path)
				break
			seen.append(script_path)
			chain.append(script_path)
			if source_records.get(script_path, {}).get("declares_ready", false):
				ready_layers += 1
			cursor = cursor.get_base_script()
		record.script_chain = chain
		record.script_inheritance_edges = maxi(0, chain.size() - 1)
		record.ready_definition_layers = ready_layers
		record.native_base_type = script.get_instance_base_type()
	elif resource is PackedScene:
		var scene: PackedScene = resource
		record.can_instantiate = scene.can_instantiate()
		record.node_count = scene.get_state().get_node_count()
		if not scene.can_instantiate():
			_fail("PackedScene has no valid root: " + path)
	resources.append(record)

func _run() -> void:
	for argument: String in OS.get_cmdline_user_args():
		if argument == "--include-tests":
			include_tests = true
		elif argument.begins_with("--audit-output="):
			output_path = argument.trim_prefix("--audit-output=")
	await process_frame
	await process_frame
	var router := root.get_node_or_null("DemoLaunch")
	var context_ok := router != null and ProjectSettings.has_setting("autoload/DemoLaunch")
	if not context_ok:
		_fail("DemoLaunch autoload is unavailable; this is not a valid project-context audit")
	if current_scene != null:
		_fail("Audit must start without a playable current scene")
	for native_name: String in NATIVE_APIS:
		var available := ClassDB.class_exists(native_name)
		var methods: Dictionary = {}
		for method_name: String in NATIVE_APIS[native_name]:
			methods[method_name] = available and ClassDB.class_has_method(native_name, method_name)
		native_report[native_name] = {"available": available, "methods": methods}
		if native_name == "DeepPondLab" and not available:
			_fail("Shared player native class DeepPondLab is unavailable")
	var paths: Array[String] = []
	_paths("res://", paths)
	if include_tests:
		_paths(ProjectSettings.globalize_path("res://../tests/godot"), paths)
	paths.sort()
	for path: String in paths:
		_read_source(path)
	for path: String in paths:
		_visit(path, [])
	for path: String in paths:
		# This harness is already compiled/executing, so do not reload itself.
		if path.get_file() != "project_resource_audit.gd":
			_load_resource(path)
	var maximum_depth := 0
	var maximum_ready_layers := 0
	var script_count := 0
	var scene_count := 0
	for record: Dictionary in resources:
		maximum_depth = maxi(maximum_depth, int(record.get("script_inheritance_edges", 0)))
		maximum_ready_layers = maxi(maximum_ready_layers, int(record.get("ready_definition_layers", 0)))
		script_count += 1 if record.extension == "gd" else 0
		scene_count += 1 if record.extension == "tscn" else 0
	var report := {
		"version": 1, "engine": Engine.get_version_info(), "project": ProjectSettings.globalize_path("res://"),
		"real_autoload_context": context_ok, "autoload_script": "res://launcher/demo_launch.gd",
		"scene_instantiations": 0, "native_instantiations": 0, "physics_steps": 0, "training_updates": 0, "save_operations": 0,
		"tests_included": include_tests, "scripts": script_count, "scenes": scene_count,
		"maximum_script_inheritance_edges": maximum_depth, "maximum_ready_definition_layers": maximum_ready_layers,
		"native_apis": native_report, "resources": resources, "static_dependencies": graph, "cycles": cycles,
		"failures": failures, "passed": failures.is_empty(),
		"limits": "Resource loading only. No _ready execution, backend probe, shader rendering, lifecycle or behavioral acceptance is implied.",
	}
	if not output_path.is_empty():
		var absolute := ProjectSettings.globalize_path(output_path) if output_path.begins_with("res://") else output_path
		if FileAccess.file_exists(absolute):
			_fail("Refusing to replace previous audit report: " + absolute)
		else:
			var directory_error := DirAccess.make_dir_recursive_absolute(absolute.get_base_dir())
			if directory_error != OK:
				_fail("Cannot create audit report directory")
			else:
				var output := FileAccess.open(absolute, FileAccess.WRITE)
				if output == null:
					_fail("Cannot create audit report")
				else:
					output.store_string(JSON.stringify(report, "\t"))
	print("PROJECT_RESOURCE_AUDIT scripts=%d scenes=%d resources=%d depth=%d ready_layers=%d failures=%d" % [script_count, scene_count, resources.size(), maximum_depth, maximum_ready_layers, failures.size()])
	quit(0 if failures.is_empty() else 1)
