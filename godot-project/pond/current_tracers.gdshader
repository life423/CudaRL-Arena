shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_never, world_vertex_coords;
uniform vec3 flow_offset = vec3(0.0);
uniform float flow_strength = 0.0;
varying vec3 point;
void vertex() {
    VERTEX.x = mod(VERTEX.x + flow_offset.x + 4.6, 9.2) - 4.6;
    VERTEX.z = mod(VERTEX.z + flow_offset.z + 3.5, 7.0) - 3.5;
    point = VERTEX;
}
void fragment() {
    float radius = length(point.xz / vec2(4.6, 3.5));
    float floor_y = -4.0 + 4.5 * pow(min(radius, 1.0), 6.0);
    float wet = (1.0 - smoothstep(0.78, 0.96, radius)) * step(floor_y + 0.10, point.y) * step(point.y, -0.05);
    ALBEDO = vec3(0.66, 0.83, 0.74);
    ALPHA = 0.42 * flow_strength * wet;
}
