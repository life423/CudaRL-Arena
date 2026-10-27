shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_disabled, shadows_disabled;

uniform float ripple_clock = 0.0;
uniform vec2 pond_radii = vec2(4.6, 3.5);
uniform float lifetime = 1.35;
varying float amplitude;
varying float wave_radius;
varying float tail_wave;
varying vec2 pond_position;

void vertex() {
	float age = ripple_clock - INSTANCE_CUSTOM.r;
	float progress = clamp(age / lifetime, 0.0, 1.0);
	float alive = step(0.0, age) * (1.0 - step(lifetime, age));
	wave_radius = 0.08 + INSTANCE_CUSTOM.b * progress;
	VERTEX.xz *= wave_radius * alive;
	amplitude = INSTANCE_CUSTOM.g * pow(1.0 - progress, 1.3) * alive;
	tail_wave = INSTANCE_CUSTOM.a;
	pond_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xz;
}

void fragment() {
	vec2 offset = UV * 2.0 - 1.0;
	float radius = length(offset);
	vec2 direction = offset / max(radius, 0.001);
	float d = (radius - 0.76) * wave_radius;
	float crest = exp(-pow(d / 0.032, 2.0));
	float trough = exp(-pow((d + 0.045) / 0.042, 2.0));
	float edge = 1.0 - smoothstep(0.86, 1.0, radius);
	// Body motion leaves a trailing arc; a moving tail emits a smaller ring.
	float wake_mask = mix(1.0 - smoothstep(-0.15, 0.45, direction.x), 1.0, tail_wave);
	float pond_edge = 1.0 - smoothstep(0.97, 1.0, length(pond_position / pond_radii));
	float wave = (crest + trough * 0.35) * edge * wake_mask * pond_edge;
	if (amplitude * wave < 0.004) {
		discard;
	}
	float slope = (crest - trough) * amplitude * 0.70;
	NORMAL_MAP = normalize(vec3(direction * slope, 1.0)) * 0.5 + 0.5;
	ALBEDO = mix(vec3(0.05, 0.23, 0.22), vec3(0.60, 0.77, 0.70), crest);
	ROUGHNESS = 0.20;
	SPECULAR = 0.65;
	ALPHA = amplitude * wave * 0.40;
}
