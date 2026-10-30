#include "rl/shelter_motor_run.h"
#include "cuda/device.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

// One immutable real-actor numerical experiment. The double reference below
// deliberately does not use CpuMlp, activate(), shelter action helpers, the
// Gaussian loss, GPU backward, or Adam's production update implementation.
namespace {
using namespace cudarl;
namespace fs=std::filesystem;
constexpr int inputs=59,outputs=4,rows=16;
constexpr uint64_t expected_source=9724089455708033410ull;
constexpr uint64_t expected_dataset=17387901361934989816ull;
void require(bool condition,const std::string& text) {if(!condition) throw std::runtime_error(text);}
uint64_t fingerprint(const std::vector<uint8_t>& bytes) {
    uint64_t value=14695981039346656037ull;
    for(uint8_t byte:bytes) {value^=byte;value*=1099511628211ull;}return value;
}
std::string quoted(const std::string& value) {std::ostringstream out;out<<std::quoted(value);return out.str();}
struct ProtectedInput {fs::path path;std::vector<uint8_t> payload;};
bool preservation(const fs::path& output,const std::vector<ProtectedInput>& protected_inputs,const char* name) {
    std::ofstream file(output/name);file<<"{\"inputs\":[";bool all=true;
    for(size_t i=0;i<protected_inputs.size();++i) {
        const auto& item=protected_inputs[i];bool same=false;
        try {same=checkpoint::read(item.path.u8string())==item.payload;} catch(...) {same=false;}
        all=all&&same;if(i) file<<',';
        file<<"{\"path\":"<<quoted(item.path.generic_u8string())<<",\"checksummed_payload_bytes\":"<<item.payload.size()
            <<",\"payload_fnv1a64\":"<<fingerprint(item.payload)<<",\"unchanged\":"<<(same?"true":"false")<<'}';
    }
    file<<"],\"all_unchanged\":"<<(all?"true":"false")<<",\"fingerprint_kind\":\"FNV1a64 provenance; source framing also validated by checkpoint checksum\"}\n";
    file.flush();require(bool(file),"Could not write input-preservation manifest");return all;
}
struct Metadata {uint32_t seed=0;int trajectory=0,tick=0,goal=-1;bool training=false,cover=false;float current=0;uint64_t condition=0;};
struct Dataset {
    std::vector<float> observation,target;std::vector<Metadata> metadata;
    std::vector<shelter_motor::State> starts;
};
Dataset read_dataset(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};require(in.value<uint32_t>()==0x48464431u,"Unsupported completed supervised dataset");Dataset out;
    out.observation=in.array<float>(3872u*inputs);out.target=in.array<float>(3872u*outputs);
    const auto count=in.value<uint64_t>();require(count==3872,"Expected the completed 3840-food/32-cover TRAIN dataset");
    out.metadata.resize(static_cast<size_t>(count));int food=0,cover=0;
    for(size_t row=0;row<count;++row) {
        auto& item=out.metadata[row];item.seed=in.value<uint32_t>();item.trajectory=in.value<int>();item.tick=in.value<int>();item.goal=in.value<int>();
        item.training=in.value<bool>();item.cover=in.value<bool>();item.current=in.value<float>();item.condition=in.value<uint64_t>();
        require(item.training,"Held-out data is forbidden in the numerical preflight");
        if(item.cover) {++cover;require(item.seed==0xC07A2026u,"Unexpected TRAIN cover provenance");}
        else {++food;require(item.seed==0xF17A2026u,"Unexpected TRAIN food provenance");}
    }
    out.starts=in.array<shelter_motor::State>(96);in.finish();
    require(out.starts.size()==96 && food==3840 && cover==32 && out.observation.size()==count*inputs && out.target.size()==count*outputs,"TRAIN dataset dimensions differ");
    for(float value:out.observation) require(std::isfinite(value),"Non-finite saved TRAIN observation");
    for(float value:out.target) require(std::isfinite(value),"Non-finite saved TRAIN target");
    for(size_t row=0;row<count;++row) {
        require((out.observation[row*inputs+55]>=.5f)==out.metadata[row].cover,"Saved row task metadata disagrees with actual network input");
        if(out.metadata[row].cover) for(int j=0;j<outputs;++j) require(out.target[row*outputs+j]==0,"Cover row is not a zero-command target");
    }
    return out;
}
struct Units {float food[outputs]{1,1,1,1};};
Units training_units(const Dataset& dataset) {
    Units out;for(int joint=0;joint<outputs;++joint) {
        double sum=0;size_t count=0;
        for(size_t row=0;row<dataset.metadata.size();++row) if(!dataset.metadata[row].cover) {
            const double value=dataset.target[row*outputs+joint];sum+=value*value;++count;
        }
        out.food[joint]=static_cast<float>(std::max(1.,std::sqrt(sum/count)));
    }
    const float expected[outputs]={2.8552513122558594f,1.2456375360488892f,702.636474609375f,1.f};
    for(int j=0;j<outputs;++j) require(out.food[j]==expected[j],"TRAIN-only target RMS differs from the completed fit's frozen scale");return out;
}
struct FrozenRows {std::vector<float> observation,target;std::vector<size_t> indices;};
FrozenRows first_rows(const Dataset& dataset) {
    FrozenRows out;for(bool cover:{false,true}) {
        int count=0;for(size_t row=0;row<dataset.metadata.size() && count<8;++row) if(dataset.metadata[row].cover==cover) {out.indices.push_back(row);++count;}
        require(count==8,"Could not select exactly the first eight rows of each mode");
    }
    for(size_t index:out.indices) {
        out.observation.insert(out.observation.end(),dataset.observation.begin()+index*inputs,dataset.observation.begin()+(index+1)*inputs);
        out.target.insert(out.target.end(),dataset.target.begin()+index*outputs,dataset.target.begin()+(index+1)*outputs);
    }return out;
}
// Explicit layout independently reconstructed from 59 -> 32 -> 32 -> 4.
struct Layout {
    std::array<int,4> width{59,32,32,4};std::array<size_t,3> weight{},bias{};size_t parameters=0;
    Layout() {for(int layer=0;layer<3;++layer) {weight[layer]=parameters;bias[layer]=parameters+width[layer]*width[layer+1];parameters=bias[layer]+width[layer+1];}}
};
struct Reference {double loss=0;std::vector<double> raw,physical,gradient;};
Reference cpu_reference(const Layout& layout,const std::vector<double>& parameter,const FrozenRows& batch,
                        const shelter_motor::ActuatorScale& gains,const Units& units,bool backward=true) {
    require(parameter.size()==layout.parameters,"Independent reference parameter dimensions differ");
    std::array<std::vector<double>,4> a;a[0].assign(batch.observation.begin(),batch.observation.end());
    for(int layer=0;layer<3;++layer) {
        const int before=layout.width[layer],after=layout.width[layer+1];a[layer+1].resize(rows*after);
        for(int row=0;row<rows;++row) for(int output=0;output<after;++output) {
            double value=parameter[layout.bias[layer]+output];
            for(int input=0;input<before;++input) value+=parameter[layout.weight[layer]+output*before+input]*a[layer][row*before+input];
            a[layer+1][row*after+output]=layer<2?std::tanh(value):value;
        }
    }
    Reference out;out.raw=a[3];out.physical.resize(rows*outputs);out.gradient.assign(layout.parameters,0);
    std::vector<double> delta(rows*outputs);
    for(int row=0;row<rows;++row) for(int joint=0;joint<outputs;++joint) {
        const size_t index=row*outputs+joint;const bool cover=batch.observation[row*inputs+55]>=.5f;
        const double gain=cover?1.:gains.value[joint],unit=cover?1.:units.food[joint];
        out.physical[index]=out.raw[index]*gain;const double error=out.physical[index]-batch.target[index];
        out.loss+=(error/unit)*(error/unit)/static_cast<double>(rows*outputs);
        delta[index]=2.*error*gain/(unit*unit*static_cast<double>(rows*outputs));
    }
    if(backward) for(int layer=2;layer>=0;--layer) {
        const int before=layout.width[layer],after=layout.width[layer+1];std::vector<double> previous(rows*before,0);
        for(int row=0;row<rows;++row) for(int output=0;output<after;++output) {
            const double activation=a[layer+1][row*after+output];
            const double dz=delta[row*after+output]*(layer<2?1.-activation*activation:1.);
            out.gradient[layout.bias[layer]+output]+=dz;
            for(int input=0;input<before;++input) {
                const size_t weight=layout.weight[layer]+output*before+input;
                out.gradient[weight]+=dz*a[layer][row*before+input];previous[row*before+input]+=parameter[weight]*dz;
            }
        }
        delta=std::move(previous);
    }
    require(std::isfinite(out.loss),"Independent double loss is non-finite");
    for(const auto* values:{&out.raw,&out.physical,&out.gradient}) for(double value:*values) require(std::isfinite(value),"Independent double reference is non-finite");return out;
}
// GPU loss is separate from the CPU formula and uses the production physical
// means. Production policy_mean_grad subsequently supplies the required gain.
__global__ void physical_mse(const float* means,const float* target,const float* observation,Units units,float* loss,float* gradient) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=rows*outputs) return;
    const bool cover=observation[(index/outputs)*inputs+55]>=.5f;const float unit=cover?1.f:units.food[index%outputs];
    const float error=means[index]-target[index];loss[index]=(error/unit)*(error/unit)/static_cast<float>(rows*outputs);
    gradient[index]=2.f*error/(unit*unit*static_cast<float>(rows*outputs));
}
template<class T> void upload(device_ptr<T>& buffer,const std::vector<T>& values,cudaStream_t stream) {
    require(buffer.size()==values.size(),"GPU upload dimensions differ");
    CUDA_CHECK_THROW(cudaMemcpyAsync(buffer.get(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
template<class T> std::vector<T> download(const T* data,size_t count,cudaStream_t stream) {
    std::vector<T> out(count);CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return out;
}
struct Metric {size_t count=0,failed=0;double maximum_absolute=0,maximum_tolerance_ratio=0;};
struct Checks {
    std::ofstream file;std::array<Metric,6> metrics{};
    const std::array<const char*,6> names{"physical_forward","standardized_loss","full_gradient","selected_cpu_finite_difference","selected_gpu_finite_difference","fresh_adam_change"};
    explicit Checks(const fs::path& path):file(path) {file<<std::setprecision(17)<<"check,index,reference,actual,absolute_error,tolerance,passed\n";}
    void compare(int category,size_t index,double expected,double actual,double absolute,double relative) {
        const double tolerance=absolute+relative*std::abs(expected),error=std::abs(actual-expected);
        const bool passed=std::isfinite(actual)&&std::isfinite(expected)&&error<=tolerance;auto& m=metrics[category];++m.count;m.failed+=!passed;
        m.maximum_absolute=std::max(m.maximum_absolute,std::isfinite(error)?error:std::numeric_limits<double>::infinity());m.maximum_tolerance_ratio=std::max(m.maximum_tolerance_ratio,error/tolerance);
        file<<names[category]<<','<<index<<','<<expected<<','<<actual<<','<<error<<','<<tolerance<<','<<passed<<'\n';
    }
    bool passed() const {for(const auto& m:metrics) if(m.failed) return false;return true;}
};
struct AdamReference {std::vector<double> change,moment1,moment2;double norm=0,clip=1;};
AdamReference cpu_adam(const std::vector<double>& gradient) {
    AdamReference out;for(double value:gradient) out.norm+=value*value;out.norm=std::sqrt(out.norm);
    out.clip=std::min(1.,.5/(out.norm+1e-6));out.change.resize(gradient.size());out.moment1.resize(gradient.size());out.moment2.resize(gradient.size());
    // Fresh moments, t=1. Keep the bias-correction algebra explicit, rather
    // than importing AdamConfig or the production update kernel as reference.
    for(size_t i=0;i<gradient.size();++i) {
        const double g=gradient[i]*out.clip;out.moment1[i]=(.1)*g;out.moment2[i]=(.001)*g*g;
        const double corrected_m=out.moment1[i]/(1.-std::pow(.9,1)),corrected_v=out.moment2[i]/(1.-std::pow(.999,1));
        out.change[i]=-.0003*corrected_m/(std::sqrt(corrected_v)+.00001);
    }return out;
}
void verify_fresh_adam(const nn::GpuAdam& optimizer,size_t count) {
    checkpoint::Writer writer;optimizer.save(writer);checkpoint::Reader reader{writer.data};const auto config=reader.value<nn::AdamConfig>();const int step=reader.value<int>();
    const auto first=reader.array<float>(count),second=reader.array<float>(count);reader.finish();
    require(step==0 && config.lr==.0003f && config.beta1==.9f && config.beta2==.999f && config.eps==.00001f && config.max_grad_norm==.5f,"GPU Adam does not have the preregistered fresh configuration");
    require(first.size()==count && second.size()==count,"Fresh Adam moment dimensions differ");
    for(size_t i=0;i<count;++i) require(first[i]==0 && second[i]==0,"Numerical preflight must not inherit PPO moments");
}
}

int main(int argc,char** argv) {
    using namespace cudarl;
    fs::path output;std::vector<ProtectedInput> protected_inputs;bool owned_output=false;
    try {
        require(argc==4,"Usage: motor_numerical_check <v5-update300-full-run> <completed-train-dataset.bin> <NEW-output-under-build>");
        output=fs::weakly_canonical(fs::absolute(argv[3]));const auto build=fs::weakly_canonical(fs::current_path()/"build");const auto relative=output.lexically_relative(build);
        require(!relative.empty() && relative!=fs::path(".") && *relative.begin()!=fs::path("..") && !fs::exists(output),"Output must be a NEW directory under repository build/");
        for(int argument:{1,2}) {const auto path=fs::weakly_canonical(fs::absolute(argv[argument]));protected_inputs.push_back({path,checkpoint::read(path.u8string())});}
        require(fingerprint(protected_inputs[0].payload)==expected_source,"Input is not the protected completed v5/update300 learner");
        require(fingerprint(protected_inputs[1].payload)==expected_dataset,"Input is not the completed run-v1 TRAIN dataset");
        const auto source=rl::read_shelter_motor_run(protected_inputs[0].payload);const auto dataset=read_dataset(protected_inputs[1].payload);
        require(rl::shelter_motor_protocol==5 && source.update==300 && source.worlds==1024 && source.seed==73,"Saved actor protocol/identity differs from the registered check");
        const float expected_gains[outputs]={8,4,1024,2};for(int j=0;j<outputs;++j) require(source.scales.value[j]==expected_gains[j],"Saved physical gain differs");
        const Layout layout;const auto specification=rl::swimmer_specification(4,59);
        require(specification.sizes==std::vector<int>({59,32,32,4}) && specification.activations==std::vector<nn::Activation>({nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}) && source.actor.size()==layout.parameters,"Real actor is not the registered three-layer MLP");
        const auto batch=first_rows(dataset);const auto units=training_units(dataset);const std::vector<double> parameters(source.actor.begin(),source.actor.end());
        const auto reference=cpu_reference(layout,parameters,batch,source.scales,units);const auto adam_reference=cpu_adam(reference.gradient);
        require(fs::create_directories(output),"Could not exclusively create the new output directory");owned_output=true;
        require(preservation(output,protected_inputs,"preservation-before.json"),"Input changed before numerical comparison");
        std::ofstream row_file(output/"frozen-rows.csv");row_file<<"row,saved_dataset_row,cover,seed,trajectory,tick,goal,condition\n";
        for(int row=0;row<rows;++row) {const size_t original=batch.indices[row];const auto& info=dataset.metadata[original];row_file<<row<<','<<original<<','<<info.cover<<','<<info.seed<<','<<info.trajectory<<','<<info.tick<<','<<info.goal<<','<<info.condition<<'\n';}
        checkpoint::Writer frozen;frozen.value(uint32_t{0x4D4E5231u});frozen.value(source.scales);frozen.value(units);frozen.array(batch.observation);frozen.array(batch.target);checkpoint::write_atomic((output/"frozen16.bin").u8string(),frozen.data);
        Checks checks(output/"component-checks.csv");std::ofstream fd(output/"selected-finite-differences.csv");fd<<std::setprecision(17)<<"layer,kind,index,parameter,h,cpu_analytic,cpu_central_difference,gpu_gradient\n";
        const auto device=probe_device();require(device.available,"Real CUDA device/kernel probe failed: "+device.error);
        rl::SwimmerStream stream;nn::GpuMlp actor(specification,rows,7);actor.set_params(source.actor);
        nn::AdamConfig config;config.lr=.0003f;config.beta1=.9f;config.beta2=.999f;config.eps=.00001f;config.max_grad_norm=.5f;nn::GpuAdam optimizer(actor.num_params(),config);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());require(actor.get_params()==source.actor,"Private GPU actor did not restore exact source parameters");verify_fresh_adam(optimizer,layout.parameters);
        device_ptr<float> observations(batch.observation.size()),targets(batch.target.size()),means(rows*outputs),loss(rows*outputs),gradient(rows*outputs);
        upload(observations,batch.observation,stream.value);upload(targets,batch.target,stream.value);
        actor.forward(observations.get(),rows,stream.value);shelter_motor::policy_means(actor.output(),observations.get(),means.get(),rows,source.scales,stream.value);
        physical_mse<<<1,128,0,stream.value>>>(means.get(),targets.get(),observations.get(),units,loss.get(),gradient.get());CUDA_CHECK_THROW(cudaGetLastError());
        shelter_motor::policy_mean_grad(gradient.get(),observations.get(),rows,source.scales,stream.value);
        actor.zero_grad(stream.value);actor.backward(gradient.get(),rows,nullptr,stream.value);CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
        const auto gpu_means=download(means.get(),rows*outputs,stream.value),gpu_losses=download(loss.get(),rows*outputs,stream.value);const auto gpu_gradient=actor.get_grads();
        for(float value:gpu_means) require(std::isfinite(value),"GPU physical mean is non-finite");
        for(float value:gpu_gradient) require(std::isfinite(value),"GPU actor gradient is non-finite");
        double gpu_loss=0;for(float value:gpu_losses) {require(std::isfinite(value),"GPU loss contribution is non-finite");gpu_loss+=value;}
        for(size_t i=0;i<gpu_means.size();++i) checks.compare(0,i,reference.physical[i],gpu_means[i],2e-5,2e-5);
        checks.compare(1,0,reference.loss,gpu_loss,1e-7,2e-5);
        require(gpu_gradient.size()==reference.gradient.size(),"Full GPU gradient dimensions differ");
        for(size_t i=0;i<gpu_gradient.size();++i) checks.compare(2,i,reference.gradient[i],gpu_gradient[i],2e-6,.002);
        for(int layer=0;layer<3;++layer) for(int kind=0;kind<2;++kind) {
            const size_t index=kind?layout.bias[layer]:layout.weight[layer];const double center=parameters[index],h=1e-5*std::max(1.,std::abs(center));
            auto plus=parameters,minus=parameters;plus[index]+=h;minus[index]-=h;
            const double numerical=(cpu_reference(layout,plus,batch,source.scales,units,false).loss-cpu_reference(layout,minus,batch,source.scales,units,false).loss)/(2*h);
            checks.compare(3,index,numerical,reference.gradient[index],2e-6,.002);checks.compare(4,index,numerical,gpu_gradient[index],2e-6,.002);
            fd<<layer<<','<<(kind?"first_bias":"first_weight")<<','<<index<<','<<center<<','<<h<<','<<reference.gradient[index]<<','<<numerical<<','<<gpu_gradient[index]<<'\n';
        }
        optimizer.step(actor.params(),actor.grads(),stream.value);CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));require(optimizer.steps()==1,"Numerical preflight must execute exactly one fresh Adam step");
        const double gpu_norm=optimizer.last_grad_norm(),gpu_clip=std::min(1.,.5/(gpu_norm+1e-6));const auto updated=actor.get_params();
        require(std::isfinite(gpu_norm),"GPU global gradient norm is non-finite");
        for(float value:updated) require(std::isfinite(value),"GPU fresh Adam parameter is non-finite");
        const std::vector<double> gpu_gradient_double(gpu_gradient.begin(),gpu_gradient.end());const auto adam_from_gpu_gradient=cpu_adam(gpu_gradient_double);
        std::ofstream adam_file(output/"fresh-adam.csv");adam_file<<std::setprecision(17)<<"index,old_parameter,cpu_reference_change,gpu_parameter_change,cpu_change_from_gpu_gradient,gpu_updated_parameter\n";
        for(size_t i=0;i<updated.size();++i) {
            const double change=static_cast<double>(updated[i])-source.actor[i];checks.compare(5,i,adam_reference.change[i],change,2e-7,.002);
            adam_file<<i<<','<<source.actor[i]<<','<<adam_reference.change[i]<<','<<change<<','<<adam_from_gpu_gradient.change[i]<<','<<updated[i]<<'\n';
        }
        checkpoint::Writer one_step;one_step.value(uint32_t{0x4D4E4331u});one_step.array(updated);optimizer.save(one_step);checkpoint::write_atomic((output/"private-one-step-evidence.bin").u8string(),one_step.data);
        const bool unchanged=preservation(output,protected_inputs,"preservation-after.json");require(unchanged,"A protected source input changed during the numerical check");
        std::ofstream summary(output/"summary.json");summary<<std::setprecision(17)<<"{\"passed\":"<<(checks.passed()?"true":"false")<<",\"scope\":\"one real saved actor, sixteen frozen TRAIN rows, one private fresh Adam step; no PPO or physical rollout\",\"protocol\":5,\"source_update\":300,\"food_rows\":8,\"cover_rows\":8,\"parameters\":"<<layout.parameters
            <<",\"independent_CPU_double_forward_backprop_transform_MSE_Adam\":true,\"independent_CPU_selected_central_differences\":6,\"source_inputs_unchanged\":true,\"source_payload_fnv1a64\":"<<expected_source<<",\"train_dataset_payload_fnv1a64\":"<<expected_dataset
            <<",\"CPU_standardized_loss\":"<<reference.loss<<",\"GPU_standardized_loss\":"<<gpu_loss<<",\"CPU_gradient_norm\":"<<adam_reference.norm<<",\"GPU_gradient_norm\":"<<gpu_norm<<",\"CPU_clip_scale\":"<<adam_reference.clip<<",\"GPU_clip_scale\":"<<gpu_clip
            <<",\"CPU_clipping_active\":"<<(adam_reference.clip<1?"true":"false")<<",\"GPU_clipping_active\":"<<(gpu_clip<1?"true":"false")<<",\"fresh_Adam_steps\":1,\"learning_rate\":0.0003,\"beta1\":0.9,\"beta2\":0.999,\"epsilon\":0.00001,\"gradient_clip_norm\":0.5,\"food_gain\":[8,4,1024,2],\"TRAIN_food_units\":["<<units.food[0]<<','<<units.food[1]<<','<<units.food[2]<<','<<units.food[3]<<"],\"cover_units\":[1,1,1,1],\"comparisons\":[";
        for(size_t i=0;i<checks.metrics.size();++i) {const auto& item=checks.metrics[i];if(i) summary<<',';summary<<"{\"kind\":"<<quoted(checks.names[i])<<",\"count\":"<<item.count<<",\"failed\":"<<item.failed<<",\"maximum_absolute_error\":"<<item.maximum_absolute<<",\"maximum_error_over_tolerance\":"<<item.maximum_tolerance_ratio<<'}';}
        summary<<"],\"tolerances_were_not_changed\":true,\"actor_adopted\":false,\"production_files_modified\":false,\"physical_rollouts\":0,\"PPO_updates\":0}\n";
        for(auto* file:{&row_file,&fd,&adam_file,&summary,&checks.file}) {file->flush();require(bool(*file),"Could not finish numerical evidence files");}
        std::cout<<"MOTOR_NUMERICAL_CHECK_DONE passed="<<checks.passed()<<" compared_rows=16 selected_FD_parameters=6 private_Adam_steps=1 source_unchanged=1 output="<<output.u8string()<<'\n';
        if(!checks.passed()) std::cerr<<"MOTOR_NUMERICAL_CHECK_DISAGREEMENT Local numerical comparison failed its registered tolerance. Investigate the identified path; this does not justify a whole-core rewrite.\n";
        return checks.passed()?0:1;
    } catch(const std::exception& error) {
        std::cerr<<"MOTOR_NUMERICAL_CHECK_ERROR "<<error.what()<<'\n';
        if(owned_output) {
            try {preservation(output,protected_inputs,"preservation-error.json");std::ofstream file(output/"error.json");file<<"{\"passed\":false,\"error\":"<<quoted(error.what())<<",\"actor_adopted\":false,\"production_files_modified\":false}\n";} catch(...) {}
        }
        return 1;
    }
}
