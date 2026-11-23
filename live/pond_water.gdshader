shader_type spatial;
render_mode cull_disabled,depth_draw_never,specular_disabled;
uniform sampler2D depth_texture : hint_depth_texture,repeat_disable,filter_nearest;
varying vec3 point;
void vertex() { point=(MODEL_MATRIX*vec4(VERTEX,1.0)).xyz; }
void fragment() {
 vec3 normal=normalize(vec3(.014*cos(point.x*1.4+point.z*.5),1.0,.012*cos(point.z*1.7-point.x*.3)));
 NORMAL=normalize((VIEW_MATRIX*vec4(normal,0.0)).xyz);
 float fresnel=pow(1.0-abs(dot(NORMAL,VIEW)),4.0);
 float depth=textureLod(depth_texture,SCREEN_UV,0.0).r;
#if CURRENT_RENDERER == RENDERER_COMPATIBILITY
 vec3 ndc=vec3(SCREEN_UV,depth)*2.0-1.0;
#else
 vec3 ndc=vec3(SCREEN_UV*2.0-1.0,depth);
#endif
 vec4 scene=INV_PROJECTION_MATRIX*vec4(ndc,1.0);
 scene.xyz/=scene.w;
 float thickness=clamp(length(scene.xyz-VERTEX),0.0,8.0);
 float absorption=1.0-exp(-thickness*.2);
 ALBEDO=mix(vec3(.21,.42,.35),vec3(.12,.35,.32),absorption)+fresnel*.05;
 ROUGHNESS=.2;
 ALPHA=.08+absorption*.05+fresnel*.06;
}
