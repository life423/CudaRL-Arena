shader_type spatial;
render_mode cull_disabled;
varying vec3 point;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p) {
 vec2 i=floor(p),f=fract(p),u=f*f*(3.0-2.0*f);
 return mix(mix(hash(i),hash(i+vec2(1.0,0.0)),u.x),mix(hash(i+vec2(0.0,1.0)),hash(i+vec2(1.0)),u.x),u.y);
}
void vertex() { point=VERTEX; }
void fragment() {
 float patches=noise(point.xz*.8)*.7+noise(point.xz*2.1)*.3;
 float flecks=noise(point.xz*23.0);
 vec3 sand=mix(vec3(.265,.275,.23),vec3(.29,.30,.255),noise(point.xz*2.7));
 vec3 silt=mix(vec3(.23,.26,.23),vec3(.25,.28,.245),noise(point.xz*1.3));
 vec3 bed=mix(silt,sand,smoothstep(.30,.70,patches));
 vec2 cell=floor(point.xz*5.0),local=fract(point.xz*5.0)-.5;
 float stone=1.0-smoothstep(.12,.25,length(local+(vec2(hash(cell),hash(cell+7.0))-.5)*.3));
 bed=mix(bed,vec3(.24,.26,.21),stone*step(.72,hash(cell+3.0))*.22);
 float caustic=1.0-smoothstep(.025,.16,abs(sin(point.x*1.45+sin(point.z*1.7))*sin(point.z*1.4+sin(point.x*1.1))));
 bed+=caustic*vec3(.006,.009,.007);
 vec3 mud=mix(vec3(.105,.125,.07),vec3(.17,.18,.10),patches);
 float grass_detail=noise(point.xz*7.5)*.75+noise(point.xz*.35)*.25;
 vec3 grass=mix(vec3(.23,.28,.14),vec3(.29,.34,.19),grass_detail);
 float path_z=13.8+6.0*sin(point.x*.052)+2.0*sin(point.x*.11);
 float path=1.0-smoothstep(.32,.65,abs(point.z-path_z));
 path*=smoothstep(.3,1.1,point.y);
 grass=mix(grass,vec3(.245,.245,.17),path*.65);
 float wet=smoothstep(-.12,.09,point.y);
 float green=smoothstep(.04,.30,point.y+(patches-.5)*.12);
 ALBEDO=mix(bed,mix(mud,grass,green),wet)+(flecks-.5)*.012;
 ROUGHNESS=.94;
}
