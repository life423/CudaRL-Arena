shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_disabled;
void vertex() {
    VERTEX.y += sin(VERTEX.x * 2.1 + TIME * 0.45) * 0.003 + cos(VERTEX.z * 2.7 - TIME * 0.34) * 0.003;
    NORMAL = normalize(vec3(-cos(VERTEX.x * 2.1 + TIME * 0.45) * 0.0063, 1.0, sin(VERTEX.z * 2.7 - TIME * 0.34) * 0.0081));
}
void fragment() {
    float fresnel = pow(1.0 - max(dot(normalize(NORMAL), normalize(VIEW)), 0.0), 3.0);
    ALBEDO = vec3(0.13, 0.43, 0.40);
    ROUGHNESS = 0.23;
    SPECULAR = 0.55;
    ALPHA = 0.065 + fresnel * 0.16;
}
