// One bounded signed-cadence learning entry; no UI or background sessions.
#include "motor/frozen_controller.h"
#include "motor/transition_examples.h"
#include "nn/mlp_gpu.h"
#include "nn/adam.h"
#include "rl/gaussian_kernels.h"
#include "nn/loss.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <functional>
namespace {
namespace fs=std::filesystem;namespace p=cudarl::physics;namespace m=cudarl::straight_motor;
namespace nn=cudarl::nn;namespace rl=cudarl::rl;namespace cp=cudarl::checkpoint;namespace demo=cudarl::demonstration;
constexpr int minibatch=256;int fit_steps=600;float wall_limit=20;uint32_t seed=0x53574D32u;
auto wall_start=std::chrono::steady_clock::now();
struct Stream {cudaStream_t value=nullptr;Stream(){CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void finite(const std::vector<float>& values){for(float x:values)require(std::isfinite(x),"Nonfinite numerical buffer; stop before further work");}
std::vector<float> download(const float* pointer,size_t count,cudaStream_t stream){std::vector<float> out(count);CUDA_CHECK_THROW(cudaMemcpyAsync(out.data(),pointer,count*sizeof(float),cudaMemcpyDeviceToHost,stream));CUDA_CHECK_THROW(cudaStreamSynchronize(stream));finite(out);return out;}
void upload(float* pointer,const std::vector<float>& values,cudaStream_t stream){finite(values);CUDA_CHECK_THROW(cudaMemcpyAsync(pointer,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,stream));}
nn::MlpSpec specification(int outputs){return {{42,32,32,outputs},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
void wall_guard(){if(std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()>=wall_limit)throw std::runtime_error("Native20s wall cap reached");}
std::vector<float> fit_samples(const demo::Dataset& data,const std::vector<float>& initial,Stream& stream,const fs::path& root,int physical_controls,int& completed,const std::function<void(int,nn::GpuMlp&,nn::GpuAdam&)>& save){
        nn::GpuMlp actor(specification(2),minibatch,seed);actor.set_params(initial);nn::AdamConfig config;config.lr=.001f;config.eps=.00001f;config.max_grad_norm=.5f;nn::GpuAdam optimizer(actor.num_params(),config);
        device_ptr<float> observations(data.observation.size()),targets(data.target.size()),batch_observation(minibatch*42),batch_target(minibatch*2),gradient(minibatch*2);upload(observations.get(),data.observation,stream.value);upload(targets.get(),data.target,stream.value);
        std::ofstream progress(root/"training.csv");progress<<"step,latent_MSE,Adam_steps,physical_controls\n";
        for(int step=0;step<fit_steps;++step){wall_guard();const auto permutation=rl::make_permutation(static_cast<uint32_t>(data.count()),seed+static_cast<uint32_t>(step));
            rl::gather_rows(permutation,0,minibatch,42,observations.get(),batch_observation.get(),stream.value);rl::gather_rows(permutation,0,minibatch,2,targets.get(),batch_target.get(),stream.value);actor.forward(batch_observation.get(),minibatch,stream.value);
            const auto predicted=download(actor.output(),minibatch*2,stream.value),expected=download(batch_target.get(),minibatch*2,stream.value);double loss=0;for(size_t j=0;j<predicted.size();++j)loss+=(predicted[j]-expected[j])*(predicted[j]-expected[j])/predicted.size();require(std::isfinite(loss),"Nonfinite supervised loss");
            nn::mse_loss_grad(actor.output(),batch_target.get(),gradient.get(),minibatch*2,stream.value);actor.zero_grad(stream.value);actor.backward(gradient.get(),minibatch,nullptr,stream.value);finite(download(actor.grads(),actor.num_params(),stream.value));optimizer.step(actor.params(),actor.grads(),stream.value);++completed;
            require(optimizer.steps()==completed,"Supervised optimizer count mismatch");progress<<completed<<','<<loss<<','<<optimizer.steps()<<','<<physical_controls<<'\n';if(completed%100==0||completed==fit_steps){progress.flush();save(completed,actor,optimizer);}
        }
    finite(actor.get_params());return actor.get_params();
}
#include "transition_motor.h"
}
int main(int argc,char** argv){try{
    require(argc==7&&std::string(argv[3])=="--actor"&&std::string(argv[5])=="--examples","Use --prepare-transitions/--run-transitions DIR --actor FILE --examples FILE");
    const auto root=fs::absolute(argv[2]),actor=fs::absolute(argv[4]),examples=fs::absolute(argv[6]);
    if(std::string(argv[1])=="--prepare-transitions")return transition_experiment::prepare(root,actor,examples);
    require(std::string(argv[1])=="--run-transitions","Unknown bounded command");return transition_experiment::run(root,actor,examples);
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
