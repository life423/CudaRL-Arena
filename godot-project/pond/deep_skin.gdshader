shader_type spatial;
render_mode cull_disabled, diffuse_burley;
uniform bool tail_surface = false;
varying vec3 point;
void vertex() { point = VERTEX; }
void fragment() {
    // The tail mesh contains world-space points, so use its radial UV for the
    // dorsal colour instead of mistaking its swimming depth for an underside.
    float top = tail_surface ? smoothstep(-0.1, 0.3, sin(UV.y * TAU)) : smoothstep(-0.12, 0.11, point.y);
    vec3 color = mix(vec3(0.68, 0.66, 0.45), vec3(0.035, 0.34, 0.32), top);
    float pattern = sin(UV.x * 69.0 + sin(UV.y * 27.0)) * sin(UV.y * 71.0 + sin(UV.x * 31.0));
    color *= 1.0 - smoothstep(0.78, 0.96, pattern) * top * 0.12;
    float edge = pow(1.0 - clamp(dot(normalize(NORMAL), VIEW), 0.0, 1.0), 3.0);
    ALBEDO = color;
    ROUGHNESS = 0.38;
    SPECULAR = 0.3;
    RIM = 0.18;
    RIM_TINT = 0.6;
    EMISSION = vec3(0.13, 0.42, 0.40) * edge * 0.055;
    SSS_STRENGTH = 0.08;
}
