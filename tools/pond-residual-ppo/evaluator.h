// Compiled through the common assessment translation unit. The physical fixtures,
// scorer, output writer and retention receipt are shared with other studies.
namespace residual = residual_contract;

void residual_initial_equivalence(const std::vector<float>& base,const std::vector<float>& weights) {
    require(residual::output_layer_zero(weights),"Initial residual output layer is not zero");
    nn::CpuMlp<float> motor({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});
    nn::CpuMlp<float> steering({{42,16,int(residual::outputs)},{nn::Activation::Tanh,nn::Activation::Identity}});
    motor.params()=base;steering.params()=weights;
    for(int sample=0;sample<64;++sample){
        float input[42];for(int j=0;j<42;++j)input[j]=float(int((sample*83+j*13)%251)-125)/64.f;
        motor.forward(input,1);steering.forward(input,1);
        for(uint32_t j=0;j<residual::outputs;++j)require(steering.output()[j]==0,"Zero residual failed nonzero-input probe");
        float command[4];motor_command(motor.output(),steering.output(),residual::outputs,true,1.f,command);
        for(int j=0;j<4;++j)require(command[j]==tanhf(motor.output()[j]),"Initial residual changes inherited commands");
    }
}
void residual_inactive_equivalence(const std::vector<float>& base,const std::vector<float>& weights) {
    nn::CpuMlp<float> motor({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});
    nn::CpuMlp<float> steering({{42,16,int(residual::outputs)},{nn::Activation::Tanh,nn::Activation::Identity}});
    motor.params()=base;steering.params()=weights;
    for(int sample=0;sample<64;++sample){
        float input[42];for(int j=0;j<42;++j)input[j]=float(int((sample*83+j*13)%251)-125)/64.f;
        motor.forward(input,1);steering.forward(input,1);
        for(float gain:{1.f,pp::frozen_holding_gain}){
            float command[4];motor_command(motor.output(),steering.output(),residual::outputs,false,gain,command);
            for(int j=0;j<4;++j)require(command[j]==tanhf(gain*motor.output()[j]),"Inactive residual changes original swimming or resting");
        }
    }
}
Summary archived_residual_retention(const std::string& text,const Options& options,const pp::CandidateState& candidate,uint64_t parent) {
    Options baseline=options;baseline.baseline=true;baseline.reuse_baseline=true;
    auto original=candidate;original.counters={};
    Summary archived;verified_prior_baseline(text,baseline,original,parent,archived);return archived;
}

int main(int argc,char** argv) {
    Options o;pp::CandidateState candidate;Budget budget;Summary summary;Ledger ledger;
    std::unique_ptr<cp::Lease> lease;bool owns_output=false,ledger_started=false;
    uint64_t checkpoint=0,parent_checksum=0,protocol_checksum=0;
    try{
        o=options(argc,argv);o.residual_evaluation=true;
        if(o.self_test){
            pure_tests();
            auto weights=nn::init_params({{42,16,int(residual::outputs)},{nn::Activation::Tanh,nn::Activation::Identity}},97);
            std::fill(weights.begin()+residual::output_offset,weights.end(),0.f);
            residual_initial_equivalence(nn::init_params({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}},93),weights);
            auto nonzero=weights;std::fill(nonzero.begin()+residual::output_offset,nonzero.end(),.25f);
            residual_inactive_equivalence(nn::init_params({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}},93),nonzero);
            residual::Actor example;example.parent=17;example.protocol=19;example.weights=weights;
            require(residual::read_actor(residual::write_actor(example),17,19).update==0,"Initial residual contract roundtrip failed");
            std::cout<<"{\"residual_checks_passed\":true,\"deterministic_input_probes\":64,\"physical_controls\":0,\"optimizer_steps\":0}\n";return 0;
        }
        if(o.describe){
            if(!o.protocol.empty()){const auto bytes=file(o.protocol);require(validated_study_id({bytes.begin(),bytes.end()})==residual::protocol_id,"Wrong residual evaluation protocol");}
            std::cout<<"{\"protocol_id\":\""<<residual::protocol_id<<"\",\"eval_source_identity_fnv1a64\":\""<<eval_source()
                <<"\",\"eval_source_sha256\":\""<<pond_candidate_source_sha256<<"\",\"runtime_version\":4,\"query_arithmetic_version\":2"
                <<",\"task_stride\":"<<sizeof(ps::SignedCourseTask)<<",\"actor_inputs\":42,\"actor_parameters\":"<<residual::parameters<<",\"base_actor_inputs\":41"
                <<",\"retention_cases\":20,\"turn_cases\":16,\"max_controls_per_checkpoint\":36800,\"max_assessments\":5,\"optimizer_steps\":0}\n";return 0;
        }
        require(!o.final_partial&&o.distilled_actor.empty(),"Residual assessment has its own checkpoint schedule");
        const auto parent=load_parent(o.parent);parent_checksum=parent.checksum;
        const auto plan=file(o.protocol);protocol_checksum=cp::checksum(plan);const std::string text(plan.begin(),plan.end());
        o.study_id=validated_study_id(text);require(o.study_id==residual::protocol_id,"Residual protocol required");
        require(std::stoull(scalar(text,"retention_evaluator_source_fnv1a64"))==eval_source(),"Residual evaluator source differs from sealed protocol");
        require(sha256(file(o.reference))==scalar(text,"long_reference_sha256"),"Long-swim reference differs");
        const auto payload=cp::read(o.candidate.string());checkpoint=cp::checksum(payload);
        const auto learned=residual::read_actor(payload,parent.checksum,protocol_checksum);
        candidate.actor_inputs=41;candidate.actor=parent.actor;candidate.frozen_gain=parent.gain;
        candidate.counters.completed_updates=learned.update;candidate.counters.attempted_controls=learned.training_controls;
        candidate.counters.attempted_substeps=learned.training_substeps;candidate.counters.actor_adam_calls=learned.actor_adam_calls;
        candidate.counters.critic_adam_calls=learned.critic_adam_calls;
        const int expected_slot=static_cast<int>(learned.update/8);
        assessment_slot(learned.update,o.baseline,false,(uint64_t{1}<<expected_slot)-1);
        if(learned.update==0)residual_initial_equivalence(parent.actor,learned.weights);
        nn::CpuMlp<float> motor({{41,32,32,4},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}});
        nn::CpuMlp<float> steering({{42,16,int(residual::outputs)},{nn::Activation::Tanh,nn::Activation::Identity}});
        motor.params()=parent.actor;steering.params()=learned.weights;
        const auto expected_keys=long_reference(o.reference);
        require(!fs::exists(o.out)&&fs::create_directories(o.out),"Evaluation output already exists");owns_output=true;
        if(o.preflight){summary.reason="residual_parsed_without_physics";write_result(o,candidate,budget,summary,true,checkpoint);return 0;}
        lease=std::make_unique<cp::Lease>(o.ledger.string()+".assessment");ledger=load_ledger(o.ledger,parent.checksum,protocol_checksum);budget.prior=ledger;
        const int slot=assessment_slot(learned.update,o.baseline,false,ledger.mask);
        require(ledger.controls+evaluation_controls<=aggregate_controls&&ledger.substeps+evaluation_substeps<=aggregate_substeps,"Aggregate evaluation reservation exceeded");
        ledger.mask|=uint64_t{1}<<slot;ledger.active=true;ledger.closed=learned.update==32;save_ledger(o.ledger,ledger);ledger_started=true;
        if(o.reuse_baseline){
            auto receipt=verified_prior_baseline(text,o,candidate,parent.checksum,summary);
            receipt.protocol_checksum=protocol_checksum;receipt.checkpoint_checksum=checkpoint;receipt.eval_source_identity=eval_source();
            cp::write_atomic((o.out/"baseline-receipt.bin").string(),pp::write_retention_receipt(receipt));
        }else{
            Keys candidate_keys;uint32_t left_passed=0,right_passed=0;
            {
                Output output(o.out);
                for(const auto& spec:specs())if(spec.turn){
                    const auto outcome=evaluate(spec,candidate,motor,budget,output,expected_keys,&steering);
                    ++summary.turn_attempted;summary.turn_passed+=outcome.passed?1u:0u;summary.turn_completed+=outcome.completed?1u:0u;
                    if(outcome.passed){if(spec.turn_degrees>0)++left_passed;else ++right_passed;}
                    if(outcome.hard){summary.hard=true;summary.reason=outcome.reason;break;}
                }
                const bool meaningful=summary.turn_passed>=4&&left_passed>=2&&right_passed>=2;
                if(!summary.hard&&(!o.turn_first||meaningful)){
                    o.full_retention_executed=true;
                    for(const auto& spec:specs())if(!spec.turn){
                        const auto outcome=evaluate(spec,candidate,motor,budget,output,expected_keys,&steering);
                        ++summary.retention_attempted;summary.retention_passed+=outcome.passed?1u:0u;summary.retention_completed+=outcome.completed?1u:0u;
                        candidate_keys.insert(outcome.long_keys.begin(),outcome.long_keys.end());
                        if(spec.holding){++summary.hold_attempted;summary.hold_passed+=outcome.passed?1u:0u;}
                        else if(spec.long_swim){++summary.long_attempted;summary.long_passed+=outcome.passed?1u:0u;}
                        else{++summary.short_attempted;summary.short_passed+=outcome.passed?1u:0u;}
                        if(!outcome.passed&&summary.first_retention_failure.empty())summary.first_retention_failure=std::string(spec.suite)+"_"+std::to_string(spec.id);
                        if(outcome.hard){summary.hard=true;summary.reason=outcome.reason;break;}
                    }
                    summary.actual_long_passes=static_cast<uint32_t>(candidate_keys.size());
                    for(const auto& key:expected_keys)summary.retained_long_passes+=candidate_keys.count(key)?1u:0u;
                    summary.retention=retention_qualified(summary,budget);
                }else if(!summary.hard){
                    residual_inactive_equivalence(parent.actor,learned.weights);
                    const auto turns=summary;summary=archived_residual_retention(text,o,candidate,parent.checksum);
                    summary.turn_attempted=turns.turn_attempted;summary.turn_passed=turns.turn_passed;summary.turn_completed=turns.turn_completed;
                    o.retention_reused_after_exact_inactive_proof=true;output.retention_reused=true;
                }
            }
            if(!summary.hard&&!summary.retention)summary.reason="required_retention_failed_"+summary.first_retention_failure;
            if(summary.retention&&assessment_completed(summary)){
                pp::RetentionReceipt receipt;receipt.checkpoint_checksum=checkpoint;receipt.parent_package_checksum=parent.checksum;receipt.protocol_checksum=protocol_checksum;receipt.eval_source_identity=eval_source();
                receipt.cases_passed=20;receipt.total=20;receipt.completed_cases=20;
                receipt.controls=o.retention_reused_after_exact_inactive_proof?30400:budget.retention_controls;
                receipt.substeps=o.retention_reused_after_exact_inactive_proof?304000:budget.retention_substeps;receipt.passed=true;receipt.baseline_only=o.baseline;
                cp::write_atomic((o.out/(o.baseline?"baseline-receipt.bin":"retention-receipt.bin")).string(),pp::write_retention_receipt(receipt));
                summary.reason=o.retention_reused_after_exact_inactive_proof?"turning_still_learning_retention_reused_exact_inactive":summary.turn_passed==16?"retention_and_turning_qualified":"retention_passed_turning_still_learning";
            }
        }
        require(cp::checksum(cp::read(o.candidate.string()))==checkpoint&&cp::checksum(cp::read(o.parent.string()))==parent_checksum&&cp::checksum(file(o.protocol))==protocol_checksum,"Residual assessment inputs changed");
        require(budget.prior.bytes+directory_bytes(o.out)<=evidence_cap,"Evaluation evidence cap exceeded");
        write_result(o,candidate,budget,summary,false,checkpoint);
        ledger.controls+=budget.controls;ledger.substeps+=budget.substeps;ledger.seconds+=budget.seconds();ledger.bytes+=directory_bytes(o.out);ledger.active=false;save_ledger(o.ledger,ledger);
        return summary.retention&&assessment_completed(summary)?0:2;
    }catch(const std::exception& error){
        summary.hard=true;summary.reason=error.what();if(owns_output)write_result(o,candidate,budget,summary,o.preflight,checkpoint);
        if(ledger_started){ledger.controls+=budget.controls;ledger.substeps+=budget.substeps;ledger.seconds+=budget.seconds();ledger.bytes+=directory_bytes(o.out);ledger.active=false;save_ledger(o.ledger,ledger);}
        std::cerr<<error.what()<<'\n';return 2;
    }
}
