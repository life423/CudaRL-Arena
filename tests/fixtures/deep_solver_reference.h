#ifndef CUDARL_DEEP_SOLVER_REFERENCE_H
#define CUDARL_DEEP_SOLVER_REFERENCE_H
#include "envs/swimmer/deep_physics.h"
// Frozen pre-optimization dense Cholesky workspace, October 27, 2026.
// Test/experiment reference only. Assembly and integration equations are shared.
namespace cudarl::deep::reference {
CUDARL_HD inline void substep(BodyState& s,const float* action,float h,Physics physics={}) {
    auto a=assemble(s,physics);
    for(int j=0;j<4;++j) {
        const float displacement=s.angle[j]-clamp(s.angle[j],-limit,limit);
        a.force[6+j]+=torque*clamp(action[j],-1,1)-1e-4f*displacement;
        a.R[6+j][6+j]+=torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    float L[dofs][dofs]{},rhs[dofs]{},v[dofs]{};
    for(int i=0;i<dofs;++i) {
        rhs[i]=h*(a.force[i]-a.C[i]);for(int j=0;j<dofs;++j) rhs[i]+=a.M[i][j]*s.velocity[j];
        for(int j=0;j<=i;++j) {
            float x=a.M[i][j]+h*a.R[i][j];for(int k=0;k<j;++k) x-=L[i][k]*L[j][k];
            L[i][j]=i==j?sqrtf(fmaxf(x,1e-18f)):x/L[j][j];
        }
    }
    for(int i=0;i<dofs;++i) {float x=rhs[i];for(int j=0;j<i;++j) x-=L[i][j]*v[j];v[i]=x/L[i][i];}
    for(int i=dofs-1;i>=0;--i) {float x=v[i];for(int j=i+1;j<dofs;++j) x-=L[j][i]*v[j];v[i]=x/L[i][i];}
    s.position+=V{v[0],v[1],v[2]}*(h/scale);
    const V omega{v[3],v[4],v[5]};const float speed=norm(omega);
    if(speed>1e-10f) s.orientation=axis_rotation(omega/speed,h*speed)*s.orientation;
    s.orientation.normalize();
    for(int j=0;j<4;++j) {s.angle[j]+=h*v[6+j];s.work+=h*fmaxf(0,torque*clamp(action[j],-1,1)*v[6+j]);}
    for(int i=0;i<dofs;++i) s.velocity[i]=v[i];
    s.travel+=h*norm({v[0],v[1],v[2]})/scale;
}
}
#endif
