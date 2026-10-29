shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_never, shadows_disabled;

uniform vec2 predator_center = vec2(0.0);
uniform vec2 predator_direction = vec2(0.0, -1.0);
varying vec3 world_point;

float cross2(vec2 a, vec2 b) { return a.x * b.y - a.y * b.x; }

float triangle_mask(vec2 p, vec2 a, vec2 b, vec2 c) {
    float orientation = sign(cross2(b - a, c - a));
    float distance_a = orientation * cross2(b - a, p - a) / length(b - a);
    float distance_b = orientation * cross2(c - b, p - b) / length(c - b);
    float distance_c = orientation * cross2(a - c, p - c) / length(a - c);
    return smoothstep(-0.035, 0.035, min(distance_a, min(distance_b, distance_c)));
}

void vertex() {
    world_point = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
    vec2 forward = normalize(predator_direction);
    vec2 right = vec2(-forward.y, forward.x);
    vec2 offset = world_point.xz - predator_center;
    vec2 p = vec2(dot(offset, right), -dot(offset, forward));
    float body = 1.0 - smoothstep(0.85, 1.08, length((p - vec2(0.0, 0.05)) / vec2(0.19, 0.62)));
    float head = 1.0 - smoothstep(0.10, 0.15, length(p - vec2(0.0, -0.55)));
    float wing = 0.0;
    for (int side = 0; side < 2; side++) {
        float sign_x = side == 0 ? 1.0 : -1.0;
        vec2 mirrored = vec2(p.x * sign_x, p.y);
        wing = max(wing, triangle_mask(mirrored, vec2(0.12, -0.25), vec2(1.5, 0.4), vec2(0.6, 0.25)));
        wing = max(wing, triangle_mask(mirrored, vec2(0.6, 0.25), vec2(1.5, 0.4), vec2(1.65, 0.8)));
    }
    float tail = triangle_mask(p, vec2(0.0, 0.35), vec2(-0.3, 0.9), vec2(0.3, 0.9));
    float silhouette = max(max(body, head), max(wing, tail));
    ALBEDO = vec3(0.025, 0.055, 0.055);
    ALPHA = silhouette * 0.30;
}
