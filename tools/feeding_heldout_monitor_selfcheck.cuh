#ifndef CUDARL_FEEDING_HELDOUT_MONITOR_SELFCHECK_CUH
#define CUDARL_FEEDING_HELDOUT_MONITOR_SELFCHECK_CUH

// Synthetic observations calibrate the passive monitor independently of either
// actor. No physics, network, optimizer or saved learner is advanced or changed.
__global__ void monitor_test_update_states(fd::State* states,int tick) {
    const int world=blockIdx.x*blockDim.x+threadIdx.x;
    if(world>=6) return;
    fd::State state;
    state.environment=static_cast<uint32_t>(world);
    state.steps=tick;state.since_food=tick;
    state.body.position={0,-1,0};
    auto& food=state.pellets[0];
    food.active=world!=1;food.position={0,-1,1};food.id=1;
    state.target=food.active?0:-1;state.next_id=food.active?2u:1u;
    if(world==2) state.body.position.x=.01f*static_cast<float>(tick);
    if(world==3) {
        state.eaten=tick/25;state.since_food=tick%25;
        food.id=1u+static_cast<uint32_t>(state.eaten);
        state.next_id=food.id+1u;
    }
    if(world==4) food.position.y=.5f;
    if(world==5) state.body.velocity[3]=__int_as_float(0x7fc00000);
    states[world]=state;
}

void passive_monitor_self_check() {
    constexpr int count=6,ticks=120;
    device_ptr<fd::State> states(count);device_ptr<Metrics> device_metrics(count);
    monitor_test_update_states<<<1,32>>>(states.get(),0);CUDA_CHECK_THROW(cudaGetLastError());
    begin_observation<<<1,32>>>(states.get(),device_metrics.get(),count);CUDA_CHECK_THROW(cudaGetLastError());
    std::array<Metrics,count> before_threshold{},at_threshold{},final{};
    for(int tick=1;tick<=ticks;++tick) {
        monitor_test_update_states<<<1,32>>>(states.get(),tick);CUDA_CHECK_THROW(cudaGetLastError());
        observe<<<1,32>>>(states.get(),device_metrics.get(),count,tick);CUDA_CHECK_THROW(cudaGetLastError());
        if(tick==window-1) device_metrics.copy_to_host(before_threshold.data(),count);
        if(tick==window) device_metrics.copy_to_host(at_threshold.data(),count);
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());device_metrics.copy_to_host(final.data(),count);
    int checks=0;
    auto require=[&](bool ok,const char* message) {
        ++checks;
        if(!ok) throw std::runtime_error(std::string("Passive monitor self-check failed: ")+message);
    };
    require(window==100 && std::fabs(static_cast<double>(d::dt)*window-5.)<1e-6,"five-second sample window");
    require(before_threshold[0].immobility_events==0,"stationary case confirmed before five seconds");
    require(at_threshold[0].immobility_events==1 && at_threshold[0].max_immobile_ticks==100,"stationary case not confirmed at five seconds");
    require(final[0].immobility_events==1 && final[0].immobile_active==1,"stationary food-present case event count");
    require(final[0].max_immobile_ticks==120,"stationary case six-second duration");
    require(final[0].max_available_gap==120,"stationary food-present wait duration");
    require(final[1].immobility_events==0 && final[1].max_immobile_ticks==0,"empty-pond rest counted as a stall");
    require(final[1].empty_ticks==120 && final[1].max_available_gap==0,"empty-pond accounting");
    require(final[2].immobility_events==0 && final[2].max_immobile_ticks==0,"moving head counted as immobile");
    require(final[3].immobility_events==0,"regular collection counted as a stall");
    require(final[3].foods==4 && final[3].first_food_tick==25,"periodic collection accounting");
    require(final[3].max_gap==25,"completed intermeal gap lost its final tick");
    require(final[3].max_available_gap==25,"completed food-available wait lost its final tick");
    require(final[4].invalid_food_ticks==120 && final[4].invalid_body_ticks==0,"above-surface food not detected independently");
    require(final[5].invalid_body_ticks==120 && final[5].invalid_food_ticks==0,"nonfinite body velocity not detected independently");
    for(int world=0;world<count;++world) {
        if(world!=4) require(final[world].invalid_food_ticks==0,"valid food falsely rejected");
        if(world!=5) require(final[world].invalid_body_ticks==0,"finite body falsely rejected");
    }
    std::printf("PASSIVE_MONITOR_SELF_CHECK worlds=%d ticks=%d checks=%d PASS\n",count,ticks,checks);
    std::fflush(stdout);
}

#endif
