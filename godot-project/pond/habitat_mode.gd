extends RefCounted

# Explicit native session configuration. Scene/save identities remain stable;
# habitats are siblings rather than historical stages of a view inheritance tree.
static func describe(kind: String) -> Dictionary:
	match kind:
		"arch":
			return {"id": 0, "kind": kind, "title": "Rock arch pond", "window": "Tadpole Pond · Rock arch learning · CudaRL", "geometry": "get_arch_geometry", "start": "start_arch_learning", "source": "source_feeding_checkpoint", "destination": "arch_checkpoint", "parent": "feeding", "frozen": false}
		"cove":
			return {"id": 1, "kind": kind, "title": "Natural pond", "window": "Tadpole Pond · Cove learning · CudaRL", "geometry": "get_cove_geometry", "start": "start_cove_learning", "source": "source_arch_checkpoint", "destination": "cove_checkpoint", "parent": "arch", "frozen": false}
		"boulder":
			return {"id": 2, "kind": kind, "title": "Boulder pond", "window": "Tadpole Pond · Boulder learning · CudaRL", "geometry": "get_boulder_geometry", "start": "start_boulder_learning", "source": "source_cove_checkpoint", "destination": "boulder_checkpoint", "parent": "cove", "frozen": false}
		"overhang":
			return {"id": 3, "kind": kind, "title": "Rock ledge pond", "window": "Tadpole Pond · Rock ledge · live · CudaRL", "geometry": "get_overhang_geometry", "start": "watch_overhang", "source": "source_boulder_checkpoint", "destination": "", "parent": "boulder", "frozen": true}
	return {}

static func backend() -> String:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			return argument.get_slice("=", 1)
	return "auto"

static func start_session(host: Control, lab: RefCounted, mode: Dictionary) -> bool:
	if not lab or mode.is_empty() or not lab.has_method(mode.start):
		return false
	var source: String = str(host.get(mode.source))
	if mode.frozen:
		return bool(lab.call(mode.start, source, backend()))
	return bool(lab.call(mode.start, source, str(host.get(mode.destination)), int(host.get("population")), backend()))
