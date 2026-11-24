// Use the exact qualification's immutable input parser, interleaved per-substep
// physical tolerances, SHA checks, CUDA timing and watchdog in both comparisons.
#define main original_exact_driver
#include "../pond-clearance-dynamics/dynamics.cpp"
#undef main
#include "compare.h"
struct CertificateTotals{
    uint64_t free_steps=0,exact_steps=0,exact_queries=0,free_trials=0,precise_controls=0;
    int violations=0;double maximum_bound_excess_m=0,maximum_exact_min_error_m=0;
    std::string first;
    void check(bool ok,const char* cause){if(!ok){++violations;if(first.empty())first=cause;}}
    void verify(const Output& exact,const Output& fast,const swept::Certificate& c,Differences& physics){
        Output comparable=fast;
        double exact_low=c.start_lower,exact_high=c.start_upper,prior_high=c.start_upper;int free_count=0;
        check(std::isfinite(exact_low)&&std::isfinite(exact_high)&&exact_low<=exact_high&&exact_high-exact_low<=1e-5,"invalid_initial_reference");
        for(int k=0;k<ticks;++k){
            const auto& oracle=exact.steps[k];exact_low=std::min(exact_low,oracle.low);exact_high=std::min(exact_high,oracle.high);
            check(std::isfinite(oracle.low)&&std::isfinite(oracle.high)&&oracle.low<=oracle.high&&oracle.high-oracle.low<=1e-5,"invalid_exact_reference");
            if(c.measured[k]){
                const auto& s=fast.steps[k];check(s.low<=s.high&&s.high-s.low<=1e-5&&std::abs(s.low-oracle.low)<=1e-5&&std::abs(s.high-oracle.high)<=1e-5,"fallback_gap");
            }else{
                ++free_count;
                const double excess=c.bounds[k]-std::min(prior_high,oracle.high);
                maximum_bound_excess_m=std::max(maximum_bound_excess_m,excess);
                check(std::isfinite(c.bounds[k])&&c.bounds[k]>.01001&&excess<=1e-12,"unproven_swept_bound");
                check(oracle.low>.010&&std::all_of(oracle.mode,oracle.mode+3,[](int mode){return mode==0;}),"false_free_step");
                check(fast.steps[k].low==1e30&&fast.steps[k].high==1e30,"unmeasured_step_format");
                // Actual values are deliberately unavailable; omit only these two
                // fields from the exact-number comparator after checking their flags,
                // mathematical inequality and threshold classification independently.
                comparable.steps[k].low=oracle.low;comparable.steps[k].high=oracle.high;
            }
            prior_high=oracle.high;
        }
        check(free_count==c.free_steps&&c.free_steps+c.exact_steps==10&&c.free_steps<=c.free_trials&&c.free_trials<=10,"query_accounting");
        check(c.exact_queries>=c.exact_steps&&c.exact_queries<=2*c.exact_steps,"fallback_query_accounting");
        check(std::isfinite(c.bound)&&c.bound<=exact_high+1e-12&&((c.bound<.010)==(exact_low<.010)),"wrong_control_class");
        if(c.precise){
            const double diff=std::abs(c.minimum-exact_low);maximum_exact_min_error_m=std::max(maximum_exact_min_error_m,diff);
            check(std::isfinite(c.minimum)&&diff<=1e-5,"wrong_precise_minimum");
        }else check(c.free_steps>0&&std::isnan(c.minimum)&&c.bound>.010&&exact_low>.010,"unmeasured_control_format");
        free_steps+=c.free_steps;exact_steps+=c.exact_steps;exact_queries+=c.exact_queries;free_trials+=c.free_trials;precise_controls+=c.precise;
        physics.assess(exact,comparable);
    }
    void report(std::ostream& o){
        o<<"{\"free_substeps\":"<<free_steps<<",\"exact_substeps\":"<<exact_steps<<",\"exact_queries\":"<<exact_queries<<",\"free_trial_solves\":"<<free_trials
         <<",\"controls_with_precise_minimum\":"<<precise_controls<<",\"violations\":"<<violations<<",\"first_violation\":\""<<first
         <<"\",\"max_bound_excess_m\":"<<maximum_bound_excess_m<<",\"max_precise_minimum_error_m\":"<<maximum_exact_min_error_m<<'}';
    }
};
int main(int argc,char** argv){
    const auto started=Clock::now();fs::path output;bool owns=false;
    try{
        const std::string binding=std::string("{\"protocol\":\"pond-broadphase-dynamics-v1\",\"reference_library_sha256\":\"")+old_sha+"\",\"runtime_sha256\":\""+new_sha
          +"\",\"exact_geometry_sha256\":\""+common_sha+"\",\"sweep_sha256\":\""+swept_sha+"\",\"maximum_gpu_controls\":2056,\"maximum_cpu_controls\":8,\"maximum_physical_substeps\":20640,\"wall_cap_seconds\":30,\"optimizer_calls\":0}";
        if(argc==2&&std::string(argv[1])=="--describe"){std::cout<<binding<<'\n';return 0;}
        const bool preflight=argc==3&&std::string(argv[1])=="--preflight";
        require(preflight||(argc==5&&std::string(argv[1])=="--run"),"Use --describe, --preflight <root>, or --run <root> <sealed-file> <seal SHA256>");
        const auto root=fs::canonical(argv[2]);output=root/"assets/research-evidence/pond-broadphase-dynamics-v1";
        std::unique_ptr<Deadline> deadline;
        if(!preflight){
            require(digest(argv[3])==argv[4]&&fs::file_size(argv[3])>80,"Invalid physics seal");
            require(!fs::exists(output)&&fs::create_directories(output),"This finite replay has already been attempted");owns=true;
            deadline=std::make_unique<Deadline>(output,started);
        }
        std::vector<Group> groups;groups.push_back(rollout(root/"build/turn38-residual/updates-1-8/batch-8.bin",early_hash,8,{0,1}));
        groups.push_back(rollout(root/"build/turn38-residual/updates-9-16/batch-12.bin",late_hash,12,{55,315}));groups.push_back(contacts(root));
        if(preflight){std::cout<<"{\"verified_fixture_and_command_sets\":1028,\"physical_controls\":0}\n";return 0;}
        const double inputs_ms=ms(started);const auto setup=Clock::now();Device d;swept::load();cuda(cudaFree(d.scratch));d.scratch=nullptr;
        cuda(cudaMalloc(&d.scratch,worlds*std::max(candidate::scratch(),swept::scratch())));
        swept::Certificate* diagnostics=nullptr;cuda(cudaMalloc(&diagnostics,worlds*sizeof(swept::Certificate)));swept::attach(diagnostics);
        const double setup_ms=ms(setup);
        std::ofstream csv(output/"worlds.csv"),log(output/"ledger.csv"),bank(output/"bank-forces.csv");
        csv<<"stage,world,backend,failed_physical_components,failed_certificate_checks,first_reason,free_substeps,exact_queries,precise,lower_bound_m,numeric_minimum_m,force_substeps,max_force_delta_N\n";
        bank<<"bank_id,backend,substep,link,mode,x_N,y_N,z_N\n";log<<"stage,arm,controls,substeps,status\n";csv<<std::setprecision(17);bank<<std::setprecision(17);
        uint64_t gpu_count=0,cpu_count=0,gpu_steps=0,cpu_steps=0;bool passed=true;int index=0;std::ostringstream stages;stages<<std::setprecision(17);
        for(const auto& g:groups){
            const int n=int(g.requests.size());const bool is_bank=n==4;
            std::vector<Output> reference(n),fast(n);std::vector<swept::Certificate> certificates(n);
            std::vector<Output>* sets[2]={&reference,&fast};const Issue arms[2]={candidate::launch,swept::launch};double times[2]{},transfers=0,cpu_ms=0;
            for(int arm=0;arm<2;++arm){
                const char* label=arm?"swept_cuda":"exact_cuda";count(log,g,label,n,0,"issued");times[arm]=d.run(arms[arm],g,*sets[arm],transfers);
                uint64_t c=0,s=0;for(const auto& row:*sets[arm]){c+=row.controls;s+=row.substeps;}gpu_count+=c;gpu_steps+=s;count(log,g,label,c,s,"actual");
            }
            const auto copy=Clock::now();cuda(cudaMemcpy(certificates.data(),diagnostics,n*sizeof(swept::Certificate),cudaMemcpyDeviceToHost));transfers+=ms(copy);
            Differences gpu,cpu,direct;CertificateTotals gpu_cert,cpu_cert;
            auto check_row=[&](int id,const Output& actual,const swept::Certificate& c,const char* name,CertificateTotals& summary,Differences& physics){
                CertificateTotals row;Differences delta;row.verify(reference[id],actual,c,delta);summary.verify(reference[id],actual,c,physics);
                if(is_bank){row.check(c.exact_steps==10&&delta.contact_steps>0,"bank_coverage");}
                csv<<g.name<<','<<(is_bank?id+4:id)<<','<<name<<','<<delta.failures<<','<<row.violations<<','<<(row.first.empty()?delta.first:row.first)<<','<<c.free_steps<<','<<c.exact_queries<<','<<c.precise<<','<<c.bound<<',';
                if(c.precise)csv<<c.minimum;csv<<','<<delta.contact_steps<<','<<delta.force<<'\n';passed&=delta.failures==0&&row.violations==0;
            };
            auto save_force=[&](int id,const char* name,const Output& row){for(int k=0;k<ticks;++k)for(int b=0;b<3;++b){const auto& s=row.steps[k];
                bank<<id+4<<','<<name<<','<<k+1<<','<<b<<','<<s.mode[b]<<','<<s.force[b][0]<<','<<s.force[b][1]<<','<<s.force[b][2]<<'\n';}};
            for(int i=0;i<n;++i){check_row(i,fast[i],certificates[i],"swept_cuda",gpu_cert,gpu);
                if(is_bank){save_force(i,"exact_cuda",reference[i]);save_force(i,"swept_cuda",fast[i]);}}
            for(int id:g.cpu_ids){
                const auto before=Clock::now();swept::Certificate c;const auto actual=swept::cpu(g.states.data()+id*world_bytes,fast[id].command,c);cpu_ms+=ms(before);
                cpu_count+=actual.controls;cpu_steps+=actual.substeps;count(log,g,"swept_cpu",actual.controls,actual.substeps,"actual");
                check_row(id,actual,c,"swept_cpu",cpu_cert,cpu);
                // Both fast outputs use the same explicit sentinel for unmeasured
                // samples. Also bind their validity tags so a sentinel cannot pass as a measurement.
                direct.assess(actual,fast[id]);
                direct.test(c.precise==certificates[id].precise&&c.free_steps==certificates[id].free_steps&&
                    c.exact_queries==certificates[id].exact_queries,"cpu_cuda_classification");
                for(int k=0;k<ticks;++k)direct.test(c.measured[k]==certificates[id].measured[k],"cpu_cuda_step_class");
                passed&=direct.failures==0;if(is_bank)save_force(id,"swept_cpu",actual);
            }
            if(index++)stages<<',';stages<<"{\"stage\":\""<<g.name<<"\",\"worlds\":"<<n<<",\"exact_gpu_ms\":"<<times[0]<<",\"swept_gpu_ms\":"<<times[1]
              <<",\"exact_arm_full_queries\":"<<11*n<<",\"transfer_wall_ms\":"<<transfers<<",\"cpu_reference_ms\":"<<cpu_ms<<",\"cuda_certificates\":";
            gpu_cert.report(stages);stages<<",\"cpu_certificates\":";cpu_cert.report(stages);
            stages<<",\"exact_gpu_vs_swept_gpu\":";gpu.print(stages);stages<<",\"exact_gpu_vs_swept_cpu\":";cpu.print(stages);
            stages<<",\"swept_cpu_vs_swept_gpu\":";direct.print(stages);stages<<'}';
            csv.flush();bank.flush();if(!passed)break;
        }
        cuda(cudaFree(diagnostics));passed&=gpu_count==2056&&cpu_count==8&&gpu_steps==20560&&cpu_steps==80;
        std::ofstream o(output/"result.json");o<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")<<",\"seal_sha256\":\""<<argv[4]
          <<"\",\"bindings\":"<<binding<<",\"gpu_controls\":"<<gpu_count<<",\"gpu_substeps\":"<<gpu_steps<<",\"cpu_controls\":"<<cpu_count<<",\"cpu_substeps\":"<<cpu_steps
          <<",\"read_ms\":"<<inputs_ms<<",\"setup_ms\":"<<setup_ms<<",\"stages\":["<<stages.str()<<"],\"wall_ms\":"<<ms(started)<<"}\n";
        o.close();csv.close();log.close();bank.close();require(bool(o)&&bool(csv)&&bool(log)&&bool(bank),"Evidence finalization");return passed?0:2;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(owns)std::ofstream(output/"error.txt")<<e.what()<<'\n';return 2;}
}
