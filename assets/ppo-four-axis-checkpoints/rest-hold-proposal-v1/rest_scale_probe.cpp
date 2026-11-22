// Static state/reward/network arithmetic only. Never integrates physics or constructs an optimizer.
#include "nn/motor/straight_task.h"
#include "nn/mlp_cpu.h"
#include "rl/append_hold_checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace fs=std::filesystem;namespace m=cudarl::straight_motor;namespace p=cudarl::physics;
namespace nn=cudarl::nn;namespace rl=cudarl::rl;namespace cp=cudarl::checkpoint;
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
nn::MlpSpec spec(int n,int out){return {{n,32,32,out},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}};}
int main(int argc,char** argv){try{
 require(argc==3,"Use REPO OUTPUT");const auto repo=fs::absolute(argv[1]),out=fs::absolute(argv[2]);require(!fs::exists(out),"Static output exists");fs::create_directories(out);
 const auto base=repo/"assets/ppo-four-axis-checkpoints/update-336-episode64";
 const auto old=rl::read_ppo_training_state(cp::read((base/"ppo-update-336.bin").string()),cp::checksum(cp::read((base/"protocol.bin").string())));
 const auto migrated=rl::migrate_appended_hold(old);
 const auto B=repo/"assets/ppo-four-axis-checkpoints/balanced-hold-scale-336to344/B";
 const auto stateB=rl::read_appended_hold(cp::read((B/"ppo-append-update-344.bin").string()),cp::checksum(cp::read((B/"protocol.bin").string())));
 nn::CpuMlp<float> olda(spec(40,4)),oldc(spec(40,1)),newa(spec(41,4)),newc(spec(41,1)),ba(spec(41,4)),bc(spec(41,1));
 olda.params()=old.actor.parameters;oldc.params()=old.critic;newa.params()=migrated.ppo.actor.parameters;newc.params()=migrated.ppo.critic;
 ba.params()=stateB.ppo.actor.parameters;bc.params()=stateB.ppo.critic;
 std::ofstream starts(out/"rest-starts.csv"),grid(out/"instantaneous-reward-grid.csv");starts<<std::setprecision(12);grid<<std::setprecision(12);
 starts<<"pose,offset_mm,target_model_y,actual_signed_depth_mm,hold_error_mm,input34,input40,reward_unscaled,reward_div1024,critic336,criticB344,critic336_with_time_feature0,criticB344_with_time_feature0,original40_observation_error,migration_actor_error,migration_critic_error";
 for(int j=0;j<4;++j)starts<<",mean336_"<<j<<",command336_"<<j<<",meanB344_"<<j<<",commandB344_"<<j;starts<<'\n';
 float obs_error=0,actor_error=0,critic_error=0;
 for(int pose:{0,3})for(float offset:{0.f,-.0005f,.0005f}){
  auto w=m::initial_world(pose);w.arena=m::long_swim_arena();auto c=m::config_for_speed(.008f);c.maximum_controls=320;c.observation_schema=4;c.target_speed_m_s=0;c.stationary_request=true;c.position_hold=true;
  m::ForwardTask t;require(m::begin_forward(w,c,t)&&m::offset_episode_start(w,t,c,offset),"Static rest initialization failed");
  const auto position=w.physical.body.position;require(m::update_velocity_request(w,c,t),"Rest anchor setup failed");
  require(p::norm(position-w.physical.body.position)==0&&t.controls==0&&w.physical.control_steps==0&&w.attempted_substeps==0&&w.physical.substeps==0,"Static preparation changed physical counters/state");
  require(t.hold_anchor_set&&t.hold_anchor_model.y==t.start_com_model.y,"Hold anchor changed target depth");
  for(float v:w.physical.body.velocity)require(v==0,"Rest velocity nonzero");for(float v:w.reference_rate)require(v==0,"Initial filter rate nonzero");
  for(int j=0;j<4;++j)require(w.physical.held_targets[j]==w.physical.body.angle[j],"Initial targets do not match original pose");
  float x[41]{},legacy[40]{};require(m::observe_appended(t,w,c,x),"Rest observation failed");auto oc=c;oc.position_hold=false;oc.observation_schema=2;
  require(m::observe(t,w,oc,legacy),"Legacy40 rest observation failed");require(x[34]==1&&x[40]==0,"Rest clock or signed forward anchor feature wrong");
  for(int j=0;j<40;++j)obs_error=std::max(obs_error,fabsf(x[j]-legacy[j]));for(int j=3;j<6;++j)require(x[j]==0,"Zero goal absent");
  olda.forward(legacy,1);oldc.forward(legacy,1);newa.forward(x,1);newc.forward(x,1);ba.forward(x,1);bc.forward(x,1);
  for(int j=0;j<4;++j)actor_error=std::max(actor_error,fabsf(olda.output()[j]-newa.output()[j]));critic_error=std::max(critic_error,fabsf(oldc.output()[0]-newc.output()[0]));
  const auto metrics=m::measure(t,w);const auto costs=m::reward_costs(metrics,w,c,false);const float reward=-p::dt*costs.total();
  float time0[41];std::copy_n(x,41,time0);time0[34]=0;oldc.forward(time0,1);bc.forward(time0,1);const float v0=oldc.output()[0],b0=bc.output()[0];bc.forward(x,1);
  starts<<pose<<','<<offset*1000<<','<<t.start_com_model.y<<','<<metrics.signed_depth_m*1000<<','<<p::norm(metrics.hold_error_m)*1000<<','<<x[34]<<','<<x[40]<<','<<reward<<','<<reward/1024<<','<<newc.output()[0]<<','<<bc.output()[0]<<','<<v0<<','<<b0<<','<<obs_error<<','<<actor_error<<','<<critic_error;
  for(int j=0;j<4;++j)starts<<','<<newa.output()[j]<<','<<tanhf(newa.output()[j])<<','<<ba.output()[j]<<','<<tanhf(ba.output()[j]);starts<<'\n';
 }
 grid<<"scenario,total_cost,instant_reward_gain1,instant_reward_gain1over1024,constant16s_return_gain1,constant16s_return_gain1over1024,speed,lane,depth,posture,effort\n";
 auto c=m::StopSequence{}.config(640,4);c.position_hold=true;auto w=m::initial_world(0);
 auto record=[&](const std::string& name,m::Metrics metrics){const auto costs=m::reward_costs(metrics,w,c,false);const double reward=-double(p::dt)*costs.total();
  const double factor=(1-std::pow(double(.99f),320))/(1-double(.99f));grid<<name<<','<<costs.total()<<','<<reward<<','<<reward/1024<<','<<reward*factor<<','<<reward*factor/1024<<','<<costs.speed<<','<<costs.lane<<','<<costs.depth<<','<<costs.posture<<','<<costs.effort<<'\n';};
 record("exact_rest",{});
 for(float speed:{.0005f,.001f,.002f,.004f,.008f}){m::Metrics metrics;metrics.velocity_m_s={speed,0,0};record("forward_speed_"+std::to_string(speed*1000)+"mm_s",metrics);}
 for(float error:{.002f,.004f,.020f,.050f,.200f}){m::Metrics metrics;metrics.hold_error_m={error,0,0};record("anchor_error_"+std::to_string(error*1000)+"mm",metrics);}
 for(float error:{.0005f,.001f}){m::Metrics metrics;metrics.depth_m=error;metrics.signed_depth_m=error;record("depth_error_"+std::to_string(error*1000)+"mm",metrics);}
 {m::Metrics metrics;metrics.roll_deg=10;record("roll10deg",metrics);metrics={};metrics.pitch_deg=10;record("pitch10deg",metrics);}
 require(obs_error==0&&actor_error<=1e-6f&&critic_error<=1e-6f,"Static original40/migration comparison failed");
 std::ofstream result(out/"result.json");result<<std::setprecision(12)<<"{\"passed\":true,\"physical_controls\":0,\"physics_substeps\":0,\"optimizer_steps\":0,\"rest_states\":6,\"original40_error\":"<<obs_error<<",\"actor_migration_error\":"<<actor_error<<",\"critic_migration_error\":"<<critic_error<<",\"constant_cost_returns_are_counterfactual_not_rollouts\":true,\"time_feature0_is_a_neural_counterfactual_only\":true}\n";
 std::cout<<"REST_STATIC_SCALE_CHECK_PASSED zero physics/optimizer steps"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
