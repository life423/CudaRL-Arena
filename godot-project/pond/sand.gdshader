shader_type spatial;
render_mode cull_disabled;
varying vec3 point;
void vertex() { point = VERTEX; }
void fragment() {
    float grain = sin(point.x * 57.0 + sin(point.z * 43.0)) * sin(point.z * 61.0);
    float wet = 1.0 - smoothstep(0.36, 0.66, point.y);
    vec3 color = mix(vec3(0.45, 0.37, 0.24), vec3(0.10, 0.22, 0.18), wet);
    float wave = abs(sin(point.x * 3.7 + sin(point.z * 4.2 + TIME * 0.23)) * sin(point.z * 3.9 + sin(point.x * 3.2 - TIME * 0.19)));
    float caustic = 1.0 - smoothstep(0.025, 0.10, wave);
    ALBEDO = color + grain * 0.012 + caustic * wet * vec3(0.035, 0.055, 0.04);
    ROUGHNESS = 0.88;
}
