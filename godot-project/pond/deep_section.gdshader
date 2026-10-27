shader_type spatial;
render_mode cull_disabled, depth_draw_never, shadows_disabled, specular_disabled;
varying vec3 point;
void vertex() { point = VERTEX; }
void fragment() {
    if (point.z <= 0.45 || point.y <= -3.35 || point.y > 0.005) { discard; }
    float depth = clamp(-point.y / 4.0, 0.0, 1.0);
    float fresnel = pow(1.0 - abs(dot(normalize(NORMAL), VIEW)), 4.0);
    float edge = 1.0 - smoothstep(0.012, 0.05, abs(point.z - 0.45));
    float bottom_edge = 1.0 - smoothstep(0.012, 0.04, abs(point.y + 3.35));
    ALBEDO = mix(vec3(0.12, 0.48, 0.43), vec3(0.035, 0.24, 0.29), depth);
    EMISSION = vec3(0.06, 0.16, 0.15) * max(edge, bottom_edge);
    ROUGHNESS = 0.2;
    SPECULAR = 0.5;
    ALPHA = 0.12 + depth * 0.04 + fresnel * 0.1 + max(edge, bottom_edge) * 0.12;
}
