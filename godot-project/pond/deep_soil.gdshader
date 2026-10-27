shader_type spatial;
render_mode cull_disabled;
varying vec3 point;
void vertex() { point = VERTEX; }
void fragment() {
    if (point.z > 0.4501) { discard; }
    float depth = clamp(-point.y / 4.25, 0.0, 1.0);
    float grain = sin(point.x * 61.0 + point.y * 37.0) * sin(point.y * 49.0 + point.z * 33.0);
    float layers = sin(point.y * 9.0 + sin(point.x * 0.8) * 0.2);
    ALBEDO = mix(vec3(0.20, 0.19, 0.12), vec3(0.07, 0.095, 0.075), depth) + grain * 0.008 + layers * 0.006;
    ROUGHNESS = 0.95;
}
