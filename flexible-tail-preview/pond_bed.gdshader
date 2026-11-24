shader_type spatial;
render_mode cull_disabled, specular_disabled;

uniform float metres_per_unit;
uniform float floor_y_m;
varying vec3 point_m;

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
	vec2 cell = floor(p);
	vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(cell), hash(cell + vec2(1, 0)), u.x),
		mix(hash(cell + vec2(0, 1)), hash(cell + vec2(1, 1)), u.x), u.y);
}

void vertex() {
	point_m = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz * metres_per_unit;
}

void fragment() {
	vec2 p = point_m.xz;
	float patches = noise(p * 15.0) * .72 + noise(p * 61.0) * .28;
	float fine = noise(p * 810.0);
	float depth = clamp(point_m.y / floor_y_m, 0.0, 1.0);
	float sand = 1.0 - smoothstep(.05, .72, depth);
	vec3 deep = mix(vec3(.025, .088, .081), vec3(.055, .150, .126), patches);
	vec3 shore = mix(vec3(.24, .24, .145), vec3(.38, .33, .197), patches);
	float caustic = 1.0 - smoothstep(.04, .15, abs(sin(p.x * 165.0 + sin(p.y * 145.0))
		* sin(p.y * 164.0 + sin(p.x * 133.0))));
	vec3 bed = mix(deep, shore, sand);
	bed += caustic * mix(vec3(.011, .029, .024), vec3(.022, .037, .025), sand) * (.35 + .65 * patches);
	vec3 mud = mix(vec3(.22, .174, .097), vec3(.31, .26, .149), patches);
	vec3 grass = mix(vec3(.093, .159, .059), vec3(.229, .294, .106), patches);
	float meadow = smoothstep(.008, .029, point_m.y + (patches - .5) * .017);
	vec3 land = mix(mud, grass, meadow);
	ALBEDO = mix(bed, land, smoothstep(-.008, .006, point_m.y)) + (fine - .5) * .018;
	ROUGHNESS = .95;
}
