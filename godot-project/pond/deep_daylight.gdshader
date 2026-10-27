shader_type sky;
void sky() {
    // A restrained reflection environment without a large bright sun disc.
    float height = smoothstep(-0.15, 0.85, EYEDIR.y);
    vec3 horizon = vec3(0.24, 0.36, 0.32);
    vec3 zenith = vec3(0.10, 0.22, 0.29);
    float clouds = smoothstep(0.4, 0.9, sin(EYEDIR.x * 5.0 + EYEDIR.z * 2.0) * sin(EYEDIR.z * 4.0 - EYEDIR.x));
    COLOR = mix(horizon, zenith, height) + clouds * 0.025 * height;
}
