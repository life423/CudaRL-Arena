uniform vec2 focus_uv = vec2(0.5);
uniform float focus_depth = -10.0;
uniform float focus_radius = 0.08;
uniform bool visibility_fade = true;
bool pond_occlusion_cut(vec2 screen_uv, float view_z, vec2 pixel, vec2 viewport_size) {
    float aspect = viewport_size.x / viewport_size.y;
    float d = length((screen_uv - focus_uv) * vec2(aspect, 1.0));
    float mask = 1.0 - smoothstep(focus_radius * 0.65, focus_radius, d);
    float occludes = step(focus_depth + 0.35, view_z);
    float opacity = mix(1.0, 0.05, mask * occludes);
    float grain = fract(sin(dot(floor(pixel), vec2(127.1, 311.7))) * 43758.5453);
    return visibility_fade && grain > opacity;
}
