shader_type spatial;
render_mode cull_disabled, diffuse_burley;
varying vec3 point;
void vertex() { point = VERTEX; }
void fragment() {
    float top = smoothstep(-0.12, 0.11, point.y);
    vec3 color = mix(vec3(0.50, 0.76, 0.59), vec3(0.09, 0.48, 0.43), top);
    float pattern = sin(UV.x * 69.0 + sin(UV.y * 27.0)) * sin(UV.y * 71.0 + sin(UV.x * 31.0));
    color *= 1.0 - smoothstep(0.78, 0.96, pattern) * top * 0.18;
    ALBEDO = color;
    ROUGHNESS = 0.30;
    SPECULAR = 0.45;
    SSS_STRENGTH = 0.12;
}
