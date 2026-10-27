shader_type spatial;
render_mode cull_disabled, depth_draw_never, specular_disabled;
uniform sampler2D depth_texture : hint_depth_texture, repeat_disable, filter_nearest;
varying vec3 point;
void vertex() { point = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz; }
void fragment() {
    float phase_a = point.x * 3.8 + point.z * 1.9 + TIME * 0.55;
    float phase_b = point.z * 5.1 - point.x * 1.4 - TIME * 0.38;
    vec3 world_normal = normalize(vec3(0.022 * cos(phase_a), 1.0, 0.018 * cos(phase_b)));
    NORMAL = normalize((VIEW_MATRIX * vec4(world_normal, 0.0)).xyz);
    float fresnel = pow(1.0 - abs(dot(NORMAL, VIEW)), 4.0);
    float depth = textureLod(depth_texture, SCREEN_UV, 0.0).r;
    vec4 scene = INV_PROJECTION_MATRIX * vec4(SCREEN_UV * 2.0 - 1.0, depth, 1.0);
    scene.xyz /= scene.w;
    float thickness = clamp(length(scene.xyz - VERTEX), 0.0, 5.0);
    float absorption = 1.0 - exp(-thickness * 0.28);
    vec3 tint = mix(vec3(0.16, 0.49, 0.43), vec3(0.035, 0.25, 0.29), absorption);
    ALBEDO = mix(tint, vec3(0.49, 0.68, 0.65), fresnel * 0.4);
    ROUGHNESS = 0.12;
    SPECULAR = 0.35;
    // Keep the animal readable through the water; no large refraction offsets.
    ALPHA = 0.12 + absorption * 0.08 + fresnel * 0.16;
}
