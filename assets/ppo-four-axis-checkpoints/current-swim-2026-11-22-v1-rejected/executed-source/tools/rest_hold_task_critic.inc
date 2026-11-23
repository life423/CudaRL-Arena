/* Task-normalized two-readout value calibration for the reset-20 study. */
struct TaskCriticScale { float rms[2]{}, mean[2]{}; int fit_n[2]{}; };
struct TaskCriticFit {
    TaskCriticScale scale;
    std::vector<float> initial,final,parameters;
    std::vector<uint8_t> optimizer;
    int steps=0;
    double raw_swim_initial_error=0;
};

TaskCriticScale fit_task_scale(const std::vector<float>& returns) {
    require(returns.size()==10240,"Task-critic scale row count changed");
    TaskCriticScale scale;
    double sum[2]{},sum2[2]{};
    for(int row=0;row<int(returns.size());++row){
        const int world=row%32,mode=(world/2)%2;
        if(world%8>=4)continue;
        const double x=returns[row];
        require(std::isfinite(x),"Task-critic fit return nonfinite");
        ++scale.fit_n[mode];sum[mode]+=x;sum2[mode]+=x*x;
    }
    for(int mode=0;mode<2;++mode){
        require(scale.fit_n[mode]==2560,"Task-critic scale fit split wrong");
        scale.mean[mode]=float(sum[mode]/scale.fit_n[mode]);
        scale.rms[mode]=float(sqrt(sum2[mode]/scale.fit_n[mode]));
        require(std::isfinite(scale.rms[mode])&&scale.rms[mode]>0,"Task-critic RMS not positive");
    }
    return scale;
}

__global__ void task_value_targets(const float* x,const float* raw_return,const int* index,float* xb,
                                   float* target,int* task,TaskCriticScale scale) {
    const int i=threadIdx.x,row=index[i],mode=(row%32/2)%2;
    for(int j=0;j<41;++j)xb[i*41+j]=x[row*41+j];
    target[i]=raw_return[row]/scale.rms[mode];
    task[i]=mode;
}

__global__ void task_value_gradient(const float* prediction,const float* target,const int* task,float* gradient) {
    const int i=threadIdx.x,mode=task[i];
    gradient[2*i]=0;
    gradient[2*i+1]=0;
    gradient[2*i+mode]=.5f*(prediction[2*i+mode]-target[i])/256.f;
}

__global__ void task_value_raw(const float* prediction,float* out,int count,TaskCriticScale scale) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<count){
        const int mode=(row%32/2)%2;
        out[row]=prediction[row*2+mode]*scale.rms[mode];
    }
}

std::vector<float> read_raw_task_value(nn::GpuMlp& net,const float* x,cudaStream_t stream,TaskCriticScale scale) {
    device_ptr<float> raw(rows);
    net.forward(x,rows,stream);
    task_value_raw<<<(rows+255)/256,256,0,stream>>>(net.output(),raw.get(),rows,scale);
    CUDA_CHECK_THROW(cudaGetLastError());
    return download(raw.get(),rows,stream);
}

void task_critic_zero_fixture(const fs::path& root) {
    std::vector<float> returns(10240);
    for(int row=0;row<10240;++row){
        const int world=row%32,mode=(world/2)%2;
        returns[row]=world%8>=4?-999.f:(mode?-.0625f:-.5f);
    }
    const auto s=fit_task_scale(returns);
    require(s.mean[0]==-.5f&&s.rms[0]==.5f&&s.mean[1]==-.0625f&&s.rms[1]==.0625f,
            "Held-out leakage or scale arithmetic changed");
    Stream stream;
    const auto d=stream.value;
    device_ptr<float> prediction(64),out(32);
    const std::vector<float> vals(64,-1.f);
    prediction.copy_from_host(vals.data(),vals.size());
    task_value_raw<<<1,32,0,d>>>(prediction.get(),out.get(),32,s);
    const auto actual=download(out.get(),32,d);
    for(int row=0;row<32;++row)
        require(actual[row]==(((row/2)%2)?-.0625f:-.5f),"Raw value routing failed");
    std::ofstream f(root/"zero-step-task-critic.json");
    f<<"{\"passed\":true,\"held_out_excluded\":true,\"raw_value_task_routing\":true,\"optimizer_steps\":0,\"physical_controls\":0}\n";
}

void fit_task_critic(Arm& a,const std::vector<float>& returns,const std::vector<float>& inherited,TaskCriticFit& f) {
    auto& d=*a.data;
    const auto s=a.stream->value;
    const auto actor=a.actor->get_params();
    cp::Writer actor_adam;
    a.actor_opt->save(actor_adam);
    f.scale=fit_task_scale(returns);
    const auto before=a.critic->get_params();
    const auto single=spec(1),dual=spec(2);
    require(single.weight_offset(2)==dual.weight_offset(2)&&before.size()==size_t(single.num_params())&&
            dual.num_params()==2466,"Task critic shape mismatch");
    std::vector<float> initial(static_cast<size_t>(dual.num_params()),0.f);
    std::copy_n(before.begin(),single.weight_offset(2),initial.begin());
    for(int j=0;j<32;++j)
        initial[dual.weight_offset(2)+j]=before[single.weight_offset(2)+j]/f.scale.rms[0];
    initial[dual.bias_offset(2)]=before[single.bias_offset(2)]/f.scale.rms[0];
    initial[dual.bias_offset(2)+1]=f.scale.mean[1]/f.scale.rms[1];
    nn::GpuMlp net(dual,rows,seed+1);
    net.set_params(initial);
    nn::GpuAdam optimizer(dual.num_params(),initial_optimizer_config(false));
    f.initial=read_raw_task_value(net,d.obs.get(),s,f.scale);
    for(int row=0;row<rows;++row){
        if((row%32/2)%2)
            require(fabsf(f.initial[row]-f.scale.mean[1])<=1e-7f,"Hold readout initial fit mean parity failed");
        else
            f.raw_swim_initial_error=std::max(f.raw_swim_initial_error,fabs(double(f.initial[row])-inherited[row]));
    }
    require(f.raw_swim_initial_error<=1e-6,"Swim readout initial raw parity failed");
    std::vector<int> fitting;
    for(int row=0;row<rows;++row)if(row%32%8<4)fitting.push_back(row);
    require(fitting.size()==5120,"Task critic fit split changed");
    std::ofstream split(a.root/"calibration-world-split.csv");
    split<<"world,pose,hold,fit\n";
    for(int i=0;i<32;++i)split<<i<<','<<(i%2?3:0)<<','<<(i/2)%2<<','<<(i%8<4)<<'\n';
    split.close();
    device_ptr<int> indices(256),task(256);
    device_ptr<float> target(256),gradient(512);
    std::ofstream log(a.root/"critic-calibration.csv");
    log<<"epoch,minibatch,critic_steps\n";
    std::cout<<"TASK_CRITIC_CALIBRATION_STARTED unix="<<std::setprecision(17)<<unix_now()
             <<" exactly80critic_steps; actor_frozen"<<std::endl;
    for(int epoch=0;epoch<4;++epoch){
        const auto permutation=rl::make_permutation(5120,3141592653u+uint32_t(epoch));
        std::vector<bool> seen(5120,false);
        for(int first=0;first<5120;first+=256){
            wall_guard();
            std::vector<int> idx(256);
            for(int i=0;i<256;++i){
                const int fit=int(rl::permute(permutation,first+i));
                require(!seen[fit],"Duplicate normalized fit row");
                seen[fit]=true;idx[i]=fitting[fit];
                require(idx[i]%32%8<4,"Held-out row leaked into normalized fit");
            }
            indices.copy_from_host(idx.data(),256);
            task_value_targets<<<1,256,0,s>>>(d.obs.get(),d.returns.get(),indices.get(),a.batch->obs.get(),target.get(),task.get(),f.scale);
            net.forward(a.batch->obs.get(),256,s);
            task_value_gradient<<<1,256,0,s>>>(net.output(),target.get(),task.get(),gradient.get());
            CUDA_CHECK_THROW(cudaGetLastError());
            if(epoch==0&&first==0){
                const auto predicted=download(net.output(),512,s),y=download(target.get(),256,s),g=download(gradient.get(),512,s);
                for(int i=0;i<256;++i){
                    const int mode=(idx[i]%32/2)%2;
                    require(fabsf(y[i]-returns[idx[i]]/f.scale.rms[mode])<1e-6f&&g[2*i+1-mode]==0&&
                            fabsf(g[2*i+mode]-.5f*(predicted[2*i+mode]-y[i])/256.f)<1e-7f,
                            "Normalized value head/gradient invariant failed");
                }
            }
            net.zero_grad(s);net.backward(gradient.get(),256,nullptr,s);
            optimizer.step(net.params(),net.grads(),s);f.steps=optimizer.steps();
            gpu::finite_parameters<<<(2466+255)/256,256,0,s>>>(net.params(),2466,d.fault.get());
            require(download(d.fault.get(),1,s)[0]==0,"Nonfinite task critic");
            log<<epoch<<','<<first/256<<','<<f.steps<<'\n';log.flush();
        }
        for(bool used:seen)require(used,"Normalized fit missed row");
    }
    f.final=read_raw_task_value(net,d.obs.get(),s,f.scale);
    f.parameters=net.get_params();
    cp::Writer adam,actor_after;
    optimizer.save(adam);a.actor_opt->save(actor_after);f.optimizer=std::move(adam.data);
    require(f.steps==80&&actor_after.data==actor_adam.data&&a.actor->get_params()==actor&&a.actor_opt->steps()==21700&&
            a.actor_opt->fresh_steps()==0,"Task critic calibration modified actor or clocks");

}

void save_task_critic(const fs::path& root,const TaskCriticFit& f) {
    cp::Writer out;
    out.value(uint32_t{0x32435452});out.value(uint32_t{1});
    out.value(cp::checksum(rest_protocol()));out.value(uint32_t{2466});
    for(int mode=0;mode<2;++mode){
        out.value(f.scale.rms[mode]);out.value(f.scale.mean[mode]);out.value(f.scale.fit_n[mode]);
    }
    out.array(f.parameters);out.array(f.optimizer);
    const auto path=root/"qualified-task-critic-v1.bin";
    require(!fs::exists(path),"Task critic output exists");
    cp::write_atomic(path.string(),out.data);
    const auto payload=cp::read(path.string());
    cp::Reader verify{payload};
    require(verify.value<uint32_t>()==0x32435452&&verify.value<uint32_t>()==1&&
            verify.value<uint64_t>()==cp::checksum(rest_protocol())&&verify.value<uint32_t>()==2466,
            "Task critic schema verification failed");
    for(int mode=0;mode<2;++mode)
        require(verify.value<float>()==f.scale.rms[mode]&&verify.value<float>()==f.scale.mean[mode]&&
                verify.value<int>()==f.scale.fit_n[mode],"Task critic scale verification failed");
    require(verify.array<float>(2466)==f.parameters&&verify.array<uint8_t>()==f.optimizer,"Task critic byte roundtrip failed");
    verify.finish();
    cp::Reader read_optimizer{f.optimizer};
    const auto state=rl::read_append_adam(read_optimizer,2466);
    read_optimizer.finish();
    require(state.steps==80,"Qualified task critic optimizer clock wrong");
}
