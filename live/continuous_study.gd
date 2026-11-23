extends RefCounted
## Portable disclosure only. This module cannot load a motor or start physics.

static func dictionary_file(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var parsed: Variant=JSON.parse_string(FileAccess.get_file_as_string(path))
	return parsed if parsed is Dictionary else {}

static func bound_file(record: Variant) -> Dictionary:
	if not record is Dictionary: return {}
	var path: Variant=record.get("path",null)
	var digest: Variant=record.get("sha256",null)
	if not path is String or not digest is String or digest.length()!=64: return {}
	if not FileAccess.file_exists(path) or FileAccess.get_sha256(path).to_lower()!=digest.to_lower(): return {}
	return dictionary_file(path)

static func count(value: Variant, maximum: float) -> bool:
	if not (value is int or value is float): return false
	var number := float(value)
	return is_finite(number) and number>=0 and number<=maximum and floorf(number)==number

static func read(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var invalid := {"issue":"The latest free-swimming result could not be verified."}
	var wrapper := dictionary_file(path)
	if not count(wrapper.get("schema",null),1) or int(wrapper.schema)!=1: return invalid
	var result := bound_file(wrapper.get("source",null))
	var proof := bound_file(wrapper.get("verification",null))
	if result.get("study","")!="continuous_pond_v1": return invalid
	if not result.get("closed",null) is bool or not bool(result.closed): return invalid
	if not result.get("qualified",null) is bool or not proof.get("passed",null) is bool or not bool(proof.passed): return invalid
	if not proof.get("qualified",null) is bool or bool(proof.qualified)!=bool(result.qualified): return invalid
	if str(proof.get("assessment_sha256","")).to_lower()!=str(wrapper.source.get("sha256","")).to_lower(): return invalid
	var limits := {"PPO_updates":64,"actor_steps":1024,"controls":86811440,"substeps":868114400,"selection_passes":16}
	for field in limits:
		if not count(result.get(field,null),float(limits[field])): return invalid
	if float(result.substeps)>10*float(result.controls): return invalid
	if float(result.actor_steps)<4*float(result.PPO_updates) or float(result.actor_steps)>16*float(result.PPO_updates): return invalid
	var optional_limits := {"withheld_continuous_passes":8,"original_hold_passes":6,"withheld_hold_passes":6,"hold300_passes":2,"short_swim_passes":4,"recovery_passes":3,"long60_retained_windows":28,"long300_retained_windows":148}
	for field in optional_limits:
		var value: Variant=result.get(field,null)
		if value!=null and not count(value,float(optional_limits[field])): return invalid
	if not result.get("fault",null) is String: return invalid
	if bool(result.qualified) and (int(result.selection_passes)!=16 or result.get("withheld_continuous_passes",null)!=8 or not str(result.fault).is_empty()): return invalid
	return {"result":result,"verified":true}

static func text(path: String) -> String:
	var checked := read(path)
	if checked.is_empty(): return ""
	if checked.has("issue"): return "\n[b]Free swimming[/b]\n"+str(checked.issue)+"\n"
	var result: Dictionary=checked.result
	var message := "\n[b]Free swimming study[/b]\n"
	if bool(result.qualified):
		message+="The saved candidate passed its finite continuous-swimming validation.\n"
	else:
		message+="Continuous swimming is not ready. The candidate remains out of the pond.\n"
	message+="Native PPO updates: %d · Starting cases: %d / 16\n" % [int(result.PPO_updates),int(result.selection_passes)]
	if result.get("withheld_continuous_passes",null)==null:
		message+="Thirty-minute validation was not run.\n"
	else:
		message+="Thirty-minute cases: %d / 8\n" % int(result.withheld_continuous_passes)
	if not bool(result.qualified) and str(result.get("fault","")).is_empty() and int(result.PPO_updates)==64:
		message+="The candidate slowed and stalled instead of learning dependable moving turns.\n"
	if result.get("short_swim_passes",null)==4 and result.get("recovery_passes",null)==3 and result.get("long60_retained_windows",null)==28 and result.get("long300_retained_windows",null)==148 and result.get("original_hold_passes",null)==6 and result.get("withheld_hold_passes",null)==6 and result.get("hold300_passes",null)==2:
		message+="Earlier swimming, recovery and all 14 holding cases passed again.\n"
	return message
