shader_type spatial;
render_mode cull_disabled;
#include "res://pond/pond_visibility.gdshaderinc"
uniform bool bed_detail = true;
uniform bool sunlight_patterns = true;
varying vec3 point;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0)), u.x), u.y);
}
vec3 pond_bed(vec2 p, float depth) {
    float patches = noise(p * 0.72 + vec2(8.3, 1.7)) * 0.7 + noise(p * 1.8) * 0.3;
    float sand_mask = smoothstep(0.35, 0.65, patches);
    vec3 sand = mix(vec3(0.19, 0.14, 0.085), vec3(0.29, 0.23, 0.15), noise(p * 3.2));
    vec3 silt = mix(vec3(0.045, 0.035, 0.025), vec3(0.085, 0.067, 0.046), noise(p * 2.6));
    float ripples = sin(p.x * 18.0 + p.y * 4.0 + noise(p * 1.3) * 4.5);
    vec3 bed = mix(silt, sand + ripples * 0.009, sand_mask);
    // Flat embedded gravel is a material pattern, not an unmodelled obstacle.
    vec2 grid = p * 6.0;
    vec2 cell = floor(grid);
    vec2 local = fract(grid);
    float stone_id = hash(cell + vec2(32.7, 9.1));
    vec2 jitter = vec2(hash(cell), hash(cell + vec2(7.1, 4.8)));
    float radius = mix(0.13, 0.25, stone_id);
    vec2 stone_offset = (local - vec2(0.5) - (jitter - vec2(0.5)) * 0.35) / radius;
    float nearest = length(stone_offset * vec2(1.0, mix(0.85, 1.25, jitter.y)));
    // Derivative-free filtering stays stable inside the wet/dry material branch.
    float stone = (1.0 - smoothstep(0.65, 1.0, nearest)) * step(0.46, stone_id);
    stone *= smoothstep(0.25, 0.52, noise(p * 1.7 + vec2(3.1)));
    vec3 gravel = mix(vec3(0.09, 0.075, 0.06), vec3(0.23, 0.21, 0.17), stone_id);
    gravel *= 0.9 + dot(stone_offset, vec2(-0.12, -0.15));
    bed = mix(bed, gravel, stone * 0.55);
    return bed * mix(1.0, 0.84, depth);
}
void vertex() { point = VERTEX; }
void fragment() {
    float broad = noise(point.xz * 1.35) * 0.65 + noise(point.xz * 3.6) * 0.35;
    float flecks = noise(point.xz * 28.0);
    float depth = clamp(-point.y / 4.0, 0.0, 1.0);
    vec3 sediment = mix(vec3(0.18, 0.24, 0.15), vec3(0.045, 0.14, 0.12), depth) * (0.85 + broad * 0.30);
    if (bed_detail && point.y < 0.15) { sediment = pond_bed(point.xz, depth); }
    vec3 mud = mix(vec3(0.12, 0.095, 0.050), vec3(0.22, 0.18, 0.095), broad);
    vec3 grass = mix(vec3(0.075, 0.13, 0.035), vec3(0.19, 0.26, 0.075), broad);
    float wet_edge = smoothstep(-0.04, 0.28, point.y + (broad - 0.5) * 0.22);
    float grass_edge = smoothstep(0.19, 0.48, point.y + (broad - 0.5) * 0.35);
    vec3 land = mix(mud, grass, grass_edge);
    vec2 light_uv = point.xz + point.y * vec2(0.10, 0.28);
    float wave = abs(sin(light_uv.x * 3.7 + sin(light_uv.y * 4.2 + TIME * 0.16)) * sin(light_uv.y * 3.9 + sin(light_uv.x * 3.2 - TIME * 0.13)));
    float caustic = 1.0 - smoothstep(0.035, 0.17, wave);
    float pool = smoothstep(0.18, 0.72, noise(light_uv * 0.33 + vec2(5.4, 1.2)));
    vec3 world_normal = normalize((INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
    caustic *= smoothstep(0.2, 0.8, abs(world_normal.y)) * (1.0 - wet_edge);
    if (sunlight_patterns) {
        sediment *= 0.84 + pool * 0.28;
        sediment += caustic * (0.35 + pool * 0.65) * vec3(0.025, 0.029, 0.018);
    }
    ALBEDO = mix(sediment, land, wet_edge) + (flecks - 0.5) * 0.025;
    ROUGHNESS = mix(0.68, 0.95, wet_edge);
    // Fade only terrain between the camera and the creature, never its floor.
    // Opaque terrain with a dithered occlusion window avoids sorting an entire
    // transparent landscape against the water surface.
    if (pond_occlusion_cut(SCREEN_UV, VERTEX.z, FRAGCOORD.xy, VIEWPORT_SIZE)) { discard; }
}
