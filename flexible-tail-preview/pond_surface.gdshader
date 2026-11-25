shader_type spatial;
render_mode cull_disabled, depth_draw_never, specular_disabled;

uniform vec2 water_radii;
varying vec2 point_xz;

void vertex() {
	point_xz = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xz;
}

void fragment() {
	vec2 p = point_xz / water_radii;
	float angle = atan(p.y, p.x);
	float edge = 1.0 + .022 * sin(3.0 * angle + .8) + .014 * cos(5.0 * angle - .4) + .009 * sin(7.0 * angle);
	float radial = length(p) / edge;
	float fringe = smoothstep(.968, .999, radial) * (.68 + .32 * sin(angle * 54.0 + sin(angle * 13.0)));
	float fresnel = pow(1.0 - abs(dot(NORMAL, VIEW)), 3.0);
	float dapple = .5 + .5 * sin(point_xz.x * .34 + sin(point_xz.y * .49)) * sin(point_xz.y * .31);
	ALBEDO = mix(vec3(.10, .30, .247), vec3(.23, .45, .31), fringe) + .035 * dapple;
	ALPHA = .065 + .11 * fresnel + .13 * fringe;
	ROUGHNESS = .23;
}
