// Pure constitutive/velocity algebra: no world, body integration, geometry,
// network, physical substep or optimizer invocation.
#include "simulation/pond_runtime.h"
#include <cstdio>
#include <cmath>
namespace pr=cudarl::pond_runtime;namespace p=cudarl::physics;namespace m=cudarl::movement;
int assertions=0,cases=0,system_evaluations=0,residual_evaluations=0,backtracks=0;bool passed=true;
double max_velocity_difference=0,max_passivity_excess=0,max_work_identity_error=0,max_true_residual=0;
void check(bool ok,const char* text){++assertions;if(!ok){passed=false;std::fprintf(stderr,"FAIL:%s\n",text);}}
double energy(double gap){return gap<0?.5*pr::stiffness_N_m*gap*gap:0;}
// Independent integral of the original physical spring/dashpot over the
// compressed part of a linear path, using clipped time endpoints/trapezoids.
double integrated_force(double g0,double h,double velocity,double damping) {
    const double g1=g0+h*velocity;if(g0>=0&&g1>=0)return 0;
    double begin=0,end=1;
    if(g0>=0)begin=g0/(g0-g1);else if(g1>=0)end=-g0/(g1-g0);
    const double ga=g0+(g1-g0)*begin,gb=g0+(g1-g0)*end,fraction=end-begin;
    return -.5*pr::stiffness_N_m*(ga+gb)*fraction+(velocity<0?-damping*velocity*fraction:0);
}
int main(){
    // Fixed before preparation:1e-8m/s scalar error,1e-12J passive allowance,
    //1e-18J work identity and unchanged1e-8true momentum residual. These do
    // not change physical qualification criteria.
    constexpr double velocity_limit=1e-8,passivity_limit=1e-12,identity_limit=1e-18;
    check(pr::version==3,"runtime version explicit");
    for(float mass:{.0002f,.002f,.02f})for(double gap:{-.0004,-.0001,0.,.0001,.0004})
    for(float old_velocity:{-.01f,-.001f,0.f,.001f,.01f})for(float h:{.0025f,.005f})for(float external:{-.001f,0.f,.001f}){
        ++cases;p::BodyState body;body.velocity[0]=old_velocity;p::System base;
        for(int i=0;i<10;++i)base.M[i][i]=i==0?mass:1.f;
        base.R[0][0]=.001f;base.force[0]=external;
        pr::ContactPoint points[3];points[0].gap_m=gap;points[0].gradient={1,0,0};points[0].jacobian[0]={1,0,0};
        points[0].damping_N_s_m=2*sqrtf(pr::stiffness_N_m*mass);points[1].gap_m=points[2].gap_m=1;
        pr::ContactSolution solution;m::Telemetry telemetry;m::Reference reference;int failure=0;
        const bool solved=pr::solve_contacts(body,base,reference,h,points,solution,telemetry,failure);
        system_evaluations+=int(solution.iterations);residual_evaluations+=int(solution.residual_evaluations);backtracks+=int(solution.backtracks);
        check(solved&&failure==0&&solution.iterations<=32,"finite Newton solves scalar fixture");
        if(!solved){std::printf("unsolved m%.9g g%.9g v%.9g h%.9g F%.9g reason%d\n",double(mass),gap,double(old_velocity),double(h),double(external),failure);continue;}
        double A[10][10]{},rhs[10]{};m::discrete_system(base,body,h,A,rhs);
        double low=-1,high=1;
        for(int iteration=0;iteration<80;++iteration){const double middle=(low+high)*.5;
            const double residual=A[0][0]*middle-rhs[0]-double(h)*integrated_force(gap,h,middle,points[0].damping_N_s_m);
            if(residual>0)high=middle;else low=middle;}
        const double expected=(low+high)*.5,velocity=solution.velocity[0],difference=fabs(velocity-expected);
        if(difference>max_velocity_difference)max_velocity_difference=difference;
        check(difference<=velocity_limit,"Newton matches independent time-integral reference");
        check(telemetry.solve_residual<=1e-8,"true nonlinear momentum residual passes unchanged limit");
        const double true_rhs=rhs[0]+double(h)*integrated_force(gap,h,velocity,points[0].damping_N_s_m);
        const double independently_balanced=fabs(A[0][0]*velocity-true_rhs)/(fabs(true_rhs)+fabs(A[0][0]*velocity)+1e-15);
        check(independently_balanced<=1e-8,"independent nonlinear force balance meets unchanged residual gate");
        if(telemetry.solve_residual>max_true_residual)max_true_residual=telemetry.solve_residual;
        const auto& law=solution.law[0];
        check(law.elastic_multiplier_N>=0&&law.damping_multiplier_N>=0&&law.resistance_N_s_m>=0,"no tensile force or negative resistance");
        check(law.contact_fraction>=0&&law.contact_fraction<=1,"compressed time fraction is valid");
        check((law.damping_multiplier_N>0)==(velocity<0&&law.contact_fraction>0),"dashpot uses approaching compressed duration");
        const double damping_loss=-double(h)*law.damping_multiplier_N*velocity;
        const double identity=double(h)*law.elastic_multiplier_N*velocity+energy(law.predicted_gap_m)-energy(gap);
        if(fabs(identity)>max_work_identity_error)max_work_identity_error=fabs(identity);
        check(fabs(identity)<=identity_limit,"spring work equals negative potential change");
        const double excess=.5*mass*velocity*velocity+energy(law.predicted_gap_m)-.5*mass*old_velocity*old_velocity-energy(gap)-double(h)*external*velocity+damping_loss+double(h)*.001f*velocity*velocity;
        if(excess>max_passivity_excess)max_passivity_excess=excess;
        check(excess<=passivity_limit,"time-average contact law is passive");
        if(gap>=0&&law.predicted_gap_m>=0){double old[10]{};const bool baseline=m::solve(A,rhs,old,10);
            check(baseline&&solution.velocity[0]==old[0],"separated path preserves native velocity exactly");}
    }
    const auto entering=pr::contact_law(.0001,.005,-.03,.08);
    check(entering.mode==2&&entering.contact_fraction>0&&entering.contact_fraction<1,"entry includes partial-duration damping immediately");
    const auto exiting=pr::contact_law(-.0001,.005,.03,.08);
    check(exiting.mode==1&&exiting.elastic_multiplier_N>0&&exiting.damping_multiplier_N==0,"release averages spring impulse without attractive damping");
    const auto still=pr::contact_law(-.0001,.005,0,.08);
    check(fabs(still.elastic_multiplier_N-pr::stiffness_N_m*.0001)<1e-18,"zero-rate spring limit unchanged");
    const auto before=pr::contact_law(.0001,.005,-.02+1e-8,.08),after=pr::contact_law(.0001,.005,-.02-1e-8,.08);
    check(fabs((after.elastic_multiplier_N+after.damping_multiplier_N)-(before.elastic_multiplier_N+before.damping_multiplier_N))<1e-8,"first-touch force continuous");
    std::printf("{\"passed\":%s,\"runtime_version\":3,\"assertions\":%d,\"velocity_algebra_cases\":%d,\"contact_system_evaluations\":%d,\"true_residual_evaluations\":%d,\"backtracks\":%d,\"maximum_velocity_difference_m_s\":%.17g,\"maximum_passivity_excess_J\":%.17g,\"maximum_spring_work_identity_error_J\":%.17g,\"maximum_true_momentum_residual\":%.17g,\"physical_controls\":0,\"physical_substeps\":0,\"geometry_queries\":0,\"CUDA_kernels\":0,\"optimizer_steps\":0}\n",passed?"true":"false",assertions,cases,system_evaluations,residual_evaluations,backtracks,max_velocity_difference,max_passivity_excess,max_work_identity_error,max_true_residual);
    return passed?0:1;
}
