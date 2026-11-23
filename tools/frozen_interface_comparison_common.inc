std::vector<float> load(const fs::path& root){
    require(cp::checksum(cp::read((root/"parent-checkpoint.bin").string()))==2039539270511780368ull&&
        cp::checksum(cp::read((root/"parent-protocol.bin").string()))==14874979637633386686ull&&
        cp::checksum(cp::read((root/"holding-parent.bin").string()))==16983070322273901339ull,"Exact frozen parent checksum differs");
    const auto state=cudarl::rl::read_ppo_training_state(cp::read((root/"parent-checkpoint.bin").string()),cp::checksum(cp::read((root/"parent-protocol.bin").string())));
    require(state.actor.update==336&&state.actor.observation_schema==2&&state.actor.reward_version==5,"Wrong frozen parent");
    const auto weights=cudarl::rl::append_zero_input_column(state.actor.parameters);
    const auto bytes=cp::read((root/"holding-parent.bin").string());cp::Reader h{bytes};
    require(h.value<uint32_t>()==0x314d4148&&h.value<uint32_t>()==1&&h.value<uint64_t>()!=0,"Wrong holding parent");
    require(h.value<uint64_t>()==cp::checksum(cp::read((root/"parent-checkpoint.bin").string())),"Holding parent binding differs");
    for(uint32_t v:{6u,6u,3u,4u,41u,4u,336u})require(h.value<uint32_t>()==v,"Holding contract differs");
    require(h.value<int>()==7&&h.value<int>()==112&&h.array<float>(2532)==weights,"Holding backbone/history differs");
    const auto head=h.array<float>(2);const float gain=h.value<float>(),sigma=h.value<float>();(void)h.array<uint8_t>();h.finish();
    require(head.size()==2&&std::isfinite(gain)&&gain>0&&gain<1&&std::isfinite(sigma)&&sigma>0,"Holding metadata invalid");
    return weights;
}
uint64_t protocol(const fs::path& root){return cp::checksum(cp::read((root/"protocol.bin").string()));}
uint64_t projected_run_bytes(){return uint64_t(maximum_substeps)*700+uint64_t(maximum_controls+100)*2200+
    uint64_t(maximum_controls+100)*(sizeof(joint::World)+sizeof(old::ForwardTask)+240)+1024*1024;}
cp::Writer binding(const fs::path& root,const fs::path& exe){cp::Writer w;w.value(uint32_t{0x31504946});w.value(uint32_t{1});w.value(protocol(root));
    for(const char* n:{"approved-protocol.md","parent-checkpoint.bin","parent-protocol.bin","holding-parent.bin"})w.value(grant::whole_file(root/n));
    w.value(grant::whole_file(exe));return w;}
void body(std::ostream& out,const p::BodyState& b){
    out<<std::setprecision(9);for(float v:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w})out<<','<<v;
    for(float v:b.angle)out<<','<<v;for(float v:b.velocity)out<<','<<v;out<<','<<b.work<<','<<b.travel;
}
void body_header(std::ostream& o){for(const char* n:{"x","y","z","qx","qy","qz","qw","j0","j1","j2","j3","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","body_work","body_travel"})o<<','<<n;}
void checked_line(std::ofstream& stream,const std::string& text,Budget& budget){budget.charge(text.size());stream<<text;require(bool(stream),"Probe evidence write failed");}
void capture(std::ofstream& control,std::ofstream& substeps,cp::Writer& raw,Budget& budget,int group,int id,int backend,int tick,const Frame& f){
    std::ostringstream o;o<<std::setprecision(17)<<group<<','<<id<<','<<backend<<','<<tick<<','<<f.world.physical.time_seconds<<','<<f.world.physical.control_steps<<','<<f.world.attempted_substeps<<','<<f.world.physical.substeps<<','<<f.attempted<<','<<f.accepted<<','<<f.reason<<','<<f.failure<<','<<int(f.task.outcome)<<','<<f.task.controls;
    body(o,f.world.physical.body);for(float v:{f.task.start_com_model.x,f.task.start_com_model.y,f.task.start_com_model.z,f.task.request.position.x,f.task.request.position.y,f.task.request.position.z,f.task.request.heading.x,f.task.request.heading.y,f.task.request.heading.z,f.task.request.terminal_velocity.x,f.task.request.terminal_velocity.y,f.task.request.terminal_velocity.z})o<<','<<v;
    for(float v:f.world.physical.held_targets)o<<','<<v;for(float v:f.world.reference_rate)o<<','<<v;for(float v:f.world.physical.last_actual_torques)o<<','<<v;
    for(float v:f.observation)o<<','<<v;for(float v:f.base)o<<','<<v;for(float v:f.command)o<<','<<v;
    double e[6];energy(f.world,e);o<<std::setprecision(17);for(double v:e)o<<','<<v;o<<'\n';checked_line(control,o.str(),budget);
    if(tick)for(int k=0;k<f.accepted;++k){const auto& z=f.substep[k];std::ostringstream row;row<<group<<','<<id<<','<<backend<<','<<tick<<','<<k+1<<','<<z.attempted<<','<<z.accepted;body(row,z.body);row<<std::setprecision(17);for(double v:z.energy)row<<','<<v;row<<'\n';checked_line(substeps,row.str(),budget);}
    const auto before=raw.data.size();raw.value(group);raw.value(id);raw.value(backend);raw.value(tick);raw.value(f.world);raw.value(f.task);
    for(float v:f.observation)raw.value(v);for(float v:f.base)raw.value(v);for(float v:f.command)raw.value(v);raw.value(f.attempted);raw.value(f.accepted);raw.value(f.reason);raw.value(f.failure);
    budget.charge(raw.data.size()-before);
}
void agreement(const Frame& a,const Frame& b,double& max_position){
    const auto& x=a.world.physical.body;const auto& y=b.world.physical.body;
    const float* left=reinterpret_cast<const float*>(&x);const float* right=reinterpret_cast<const float*>(&y);
    for(int j=0;j<21;++j){const double d=std::abs(double(left[j])-right[j]);require(std::isfinite(d),"Nonfinite backend difference");
        const double tolerance=j<3?2e-5:j<11?2e-5:j<14?2e-7:2e-5;
        if(j<3)max_position=std::max(max_position,d);require(d<=tolerance,"Matched-command physics disagreement");}
    for(int j=0;j<4;++j)for(const auto pair:{std::pair<float,float>{a.world.physical.held_targets[j],b.world.physical.held_targets[j]},
        {a.world.reference_rate[j],b.world.reference_rate[j]},{a.world.physical.last_actual_torques[j],b.world.physical.last_actual_torques[j]}})
        require(std::isfinite(pair.first)&&std::isfinite(pair.second)&&std::abs(double(pair.first)-pair.second)<=2e-5,"Matched-command motor difference");
    for(int j=0;j<40;++j)require(std::isfinite(a.world.physical.actual_torques[j])&&std::isfinite(b.world.physical.actual_torques[j])&&
        std::abs(double(a.world.physical.actual_torques[j])-b.world.physical.actual_torques[j])<=2e-5,"Matched-command substep torque difference");
    double ae[6],be[6];energy(a.world,ae);energy(b.world,be);
    for(int j=0;j<6;++j)require(std::isfinite(ae[j])&&std::isfinite(be[j])&&std::abs(ae[j]-be[j])<=1e-12+1e-4*std::max(std::abs(ae[j]),std::abs(be[j])),"Matched-command energy difference");
}
void infer(nn::GpuMlp& network,device_ptr<Frame>& d_state,device_ptr<float>& d_input,int n){observations<<<1,128>>>(d_state.get(),n);gather<<<1,128>>>(d_state.get(),d_input.get(),n);network.forward(d_input.get(),n);decode<<<1,128>>>(d_state.get(),network.output(),n);sync();}
