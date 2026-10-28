shader_type spatial;
render_mode cull_disabled, depth_draw_never, specular_disabled;
uniform sampler2D depth_texture : hint_depth_texture, repeat_disable, filter_nearest;
uniform vec2 flow_offset = vec2(0.0);
uniform vec2 flow_direction = vec2(1.0, 0.0);
uniform float flow_strength = 0.0;
varying vec3 point;
float flow_hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float flow_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(flow_hash(i), flow_hash(i + vec2(1.0, 0.0)), u.x), mix(flow_hash(i + vec2(0.0, 1.0)), flow_hash(i + vec2(1.0)), u.x), u.y);
}
void vertex() { point = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz; }
void fragment() {
    float phase_a = point.x * 3.8 + point.z * 1.9 + TIME * 0.55;
    float phase_b = point.z * 5.1 - point.x * 1.4 - TIME * 0.38;
    vec2 advected = point.xz - flow_offset;
    vec2 along = vec2(dot(advected, flow_direction), dot(advected, vec2(-flow_direction.y, flow_direction.x)));
    float coarse = flow_noise(along * vec2(1.2, 2.6));
    float fine = flow_noise(along * vec2(3.3, 8.0) + vec2(7.0, 9.0));
    float strands = pow(0.5 + 0.5 * sin(along.y * 18.0 + coarse * 8.0 + fine * 3.0), 18.0);
    float broken = smoothstep(0.32, 0.78, flow_noise(along * vec2(1.0, 2.0) + vec2(2.0, 9.0)));
    float drift = flow_strength * strands * broken;
    vec3 world_normal = normalize(vec3(0.022 * cos(phase_a), 1.0, 0.018 * cos(phase_b)));
    NORMAL = normalize((VIEW_MATRIX * vec4(world_normal, 0.0)).xyz);
    float fresnel = pow(1.0 - abs(dot(NORMAL, VIEW)), 4.0);
    float depth = textureLod(depth_texture, SCREEN_UV, 0.0).r;
#if CURRENT_RENDERER == RENDERER_COMPATIBILITY
    vec3 ndc = vec3(SCREEN_UV, depth) * 2.0 - 1.0;
#else
    vec3 ndc = vec3(SCREEN_UV * 2.0 - 1.0, depth);
#endif
    vec4 scene = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
    scene.xyz /= scene.w;
    float thickness = clamp(length(scene.xyz - VERTEX), 0.0, 5.0);
    float absorption = 1.0 - exp(-thickness * 0.28);
    vec3 tint = mix(vec3(0.16, 0.49, 0.43), vec3(0.035, 0.25, 0.29), absorption);
    ALBEDO = mix(tint, vec3(0.49, 0.68, 0.65), fresnel * 0.4) + drift * vec3(0.10, 0.16, 0.13);
    ROUGHNESS = 0.12;
    SPECULAR = 0.35;
    // Keep the animal readable through the water; no large refraction offsets.
    ALPHA = 0.12 + absorption * 0.08 + fresnel * 0.16 + drift * 0.08;
}
