shader_type spatial;
render_mode cull_disabled;
#include "res://pond/pond_visibility.gdshaderinc"
uniform vec4 base_color : source_color = vec4(1.0);
void fragment() {
    if (pond_occlusion_cut(SCREEN_UV, VERTEX.z, FRAGCOORD.xy, VIEWPORT_SIZE)) { discard; }
    ALBEDO = base_color.rgb * COLOR.rgb;
    ROUGHNESS = 0.94;
}
