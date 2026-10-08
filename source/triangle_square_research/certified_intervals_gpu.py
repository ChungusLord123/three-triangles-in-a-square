"""Exact fixed-point interval arithmetic on Metal, with an MPS cross-check.

Endpoints are dyadic rationals (Q=2^24). Intermediate products and signed
division use GPU int64, with explicit floor/ceiling. No floating point is used
for certified arithmetic. Overflow or division uncertainty disables pruning.
"""
import os,json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
import mlx.core as mx
import torch
from n3_branches_gpu import ROOT

mx.set_default_device(mx.gpu)
Q=16777216
HEADER=r'''
constant long CQ=16777216L;
constant long CWIDE=1099511627776L;
struct CI {long lo;long hi;};
struct CP {CI x;CI y;};
inline CI point(long x){return CI{x,x};}
inline long fdiv(long n,long d){long q=n/d,r=n%d;return q-((r!=0)&&((r<0)!=(d<0)));}
inline long cdiv(long n,long d){long q=n/d,r=n%d;return q+((r!=0)&&((r<0)==(d<0)));}
inline CI ci_add(CI a,CI b){return CI{a.lo+b.lo,a.hi+b.hi};}
inline CI ci_sub(CI a,CI b){return CI{a.lo-b.hi,a.hi-b.lo};}
inline CI negative(CI a){return CI{-a.hi,-a.lo};}
inline CI times_int(CI a,long k){return k>=0?CI{a.lo*k,a.hi*k}:CI{a.hi*k,a.lo*k};}
inline bool has_zero(CI a){return a.lo<=0&&a.hi>=0;}
inline bool empty(CI a){return a.lo>a.hi;}
inline bool intersect(thread CI& a,CI b){a.lo=max(a.lo,b.lo);a.hi=min(a.hi,b.hi);return !empty(a);}
inline CI times(CI a,CI b,thread bool& unsafe){
 if(max(abs(a.lo),abs(a.hi))>1073741824L||max(abs(b.lo),abs(b.hi))>1073741824L){unsafe=true;return CI{-CWIDE,CWIDE};}
 long z0=a.lo*b.lo,z1=a.lo*b.hi,z2=a.hi*b.lo,z3=a.hi*b.hi;
 return CI{fdiv(min(min(z0,z1),min(z2,z3)),CQ),cdiv(max(max(z0,z1),max(z2,z3)),CQ)};
}
inline CI quotient(CI a,CI b,thread bool& unsafe){
 if(has_zero(b)||max(abs(a.lo),abs(a.hi))>68719476736L){unsafe=true;return CI{-CWIDE,CWIDE};}
 long x0=a.lo*CQ,x1=a.hi*CQ;
 long l0=fdiv(x0,b.lo),l1=fdiv(x0,b.hi),l2=fdiv(x1,b.lo),l3=fdiv(x1,b.hi);
 long h0=cdiv(x0,b.lo),h1=cdiv(x0,b.hi),h2=cdiv(x1,b.lo),h3=cdiv(x1,b.hi);
 return CI{min(min(l0,l1),min(l2,l3)),max(max(h0,h1),max(h2,h3))};
}
inline CI square(CI a,thread bool& unsafe){
 if(max(abs(a.lo),abs(a.hi))>1073741824L){unsafe=true;return CI{0,CWIDE};}
 long x=a.lo*a.lo,y=a.hi*a.hi;
 return CI{has_zero(a)?0:fdiv(min(x,y),CQ),cdiv(max(x,y),CQ)};
}
inline CI dot(CP a,CP b,thread bool& unsafe){return ci_add(times(a.x,b.x,unsafe),times(a.y,b.y,unsafe));}
inline CP addp(CP a,CP b){return CP{ci_add(a.x,b.x),ci_add(a.y,b.y)};}
inline CP subp(CP a,CP b){return CP{ci_sub(a.x,b.x),ci_sub(a.y,b.y)};}
inline CP rotate(CI c,CI s,CP a,thread bool& unsafe){return CP{ci_sub(times(c,a.x,unsafe),times(s,a.y,unsafe)),ci_add(times(s,a.x,unsafe),times(c,a.y,unsafe))};}
inline CI sine_endpoint(long t,thread bool& unsafe){return quotient(point(2*t),ci_add(point(CQ),square(point(t),unsafe)),unsafe);}
inline void rational_rotation(CI t,thread CI& c,thread CI& s,thread bool& unsafe){
 CI tt=square(t,unsafe);
 c=CI{quotient(point(CQ-tt.hi),point(CQ+tt.hi),unsafe).lo,
       quotient(point(CQ-tt.lo),point(CQ+tt.lo),unsafe).hi};
 s=CI{sine_endpoint(t.lo,unsafe).lo,sine_endpoint(t.hi,unsafe).hi};
 intersect(c,CI{0,CQ});intersect(s,CI{-CQ,CQ});
}
inline long isqrt_exact(long n){
 long lo=0,hi=3*CQ+1;
 while(hi-lo>1){long m=(lo+hi)/2;if(m*m<=n)lo=m;else hi=m;}
 return lo;
}
'''

CONSTANT_SOURCE=r'''
bool unsafe=false;
long r3=isqrt_exact(3*CQ*CQ),r6=isqrt_exact(6*CQ*CQ);
CI k3=CI{r3,r3+1},k6=CI{r6,r6+1};
CI h=quotient(k3,point(2*CQ),unsafe),q=quotient(k6,point(4*CQ),unsafe);
long lower=CQ,upper=2*CQ;
while(upper-lower>1){
 long m=(lower+upper)/2;CI p=times_int(square(square(point(m),unsafe),unsafe),16);
 if(p.hi<27*CQ)lower=m;else if(p.lo>27*CQ)upper=m;else break;
}
long targetlo=fdiv((long)fraction[0]*CQ,(long)fraction[1]);
long targethi=cdiv((long)fraction[0]*CQ,(long)fraction[1]);
out[0]=(int)r3;out[1]=(int)(r3+1);out[2]=(int)r6;out[3]=(int)(r6+1);
out[4]=(int)lower;out[5]=(int)targetlo;out[6]=(int)targethi;
out[7]=(int)(h.lo+q.lo);out[8]=(int)(h.hi+q.hi);out[9]=(int)unsafe;
'''

def constants(numerator=14783,denominator=10000):
 k=mx.fast.metal_kernel(name='certified_constants',input_names=['fraction'],output_names=['out'],source=CONSTANT_SOURCE,header=HEADER)
 out=k(inputs=[mx.array([numerator,denominator],dtype=mx.int32)],grid=(1,1,1),threadgroup=(1,1,1),
       output_shapes=[(10,)],output_dtypes=[mx.int32])[0]
 mx.eval(out)
 assert out[9].item()==0
 return out

TEST_SOURCE=r'''
uint i=thread_position_in_grid.x;if(i>=amount[0])return;
CI a=CI{inputs[4*i],inputs[4*i+1]},b=CI{inputs[4*i+2],inputs[4*i+3]};
bool unsafe=false;
CI c=ci_add(a,b),d=ci_sub(a,b),e=times(a,b,unsafe),f=quotient(a,b,unsafe),g=square(a,unsafe);
CI t=CI{angles[2*i],angles[2*i+1]},co,si;rational_rotation(t,co,si,unsafe);
CI z[7]={c,d,e,f,g,co,si};
for(uint j=0;j<7;j++){out[14*i+2*j]=z[j].lo;out[14*i+2*j+1]=z[j].hi;}
flags[i]=(uint)unsafe;
'''

def arithmetic_tests(count=12000):
 mx.random.seed(9929)
 raw=mx.random.randint(-2*Q,2*Q,shape=(count,2),dtype=mx.int32)
 a=mx.sort(raw,axis=1)
 den=mx.sort(mx.random.randint(Q//32,2*Q,shape=(count,2),dtype=mx.int32),axis=1)
 b=mx.where((mx.arange(count)%2)[:,None]==0,den,-den[:,::-1])
 theta=mx.sort(mx.random.randint(-Q,Q+1,shape=(count,2),dtype=mx.int32),axis=1)
 inputs=mx.concatenate([a,b],axis=1)
 kernel=mx.fast.metal_kernel(name='certified_arithmetic_tests',input_names=['inputs','angles','amount'],
      output_names=['out','flags'],source=TEST_SOURCE,header=HEADER)
 values,flags=kernel(inputs=[inputs,theta,mx.array([count],dtype=mx.uint32)],grid=(count,1,1),threadgroup=(256,1,1),
          output_shapes=[(count,14),(count,)],output_dtypes=[mx.int64,mx.uint32])
 mx.eval(inputs,theta,values,flags)
 assert not mx.any(flags).item()
 # Data transfer only; independent int64 operations are on MPS.
 d=torch.device('mps')
 inp=torch.tensor(inputs.tolist(),device=d,dtype=torch.int64)
 t=torch.tensor(theta.tolist(),device=d,dtype=torch.int64)
 lo=inp[:,0];hi=inp[:,1];bl=inp[:,2];bh=inp[:,3]
 def fd(n,den):return torch.div(n,den,rounding_mode='floor')
 def cd(n,den):return -fd(-n,den)
 def quot(al,ah,dl,dh):
  n=torch.stack([al,al,ah,ah],dim=1)*Q;dv=torch.stack([dl,dh,dl,dh],dim=1)
  return fd(n,dv).amin(1),cd(n,dv).amax(1)
 prod=torch.stack([lo*bl,lo*bh,hi*bl,hi*bh],dim=1)
 eql=torch.stack([lo.square(),hi.square()],dim=1)
 sl=torch.where((lo<=0)&(hi>=0),0,fd(eql.amin(1),Q));sh=cd(eql.amax(1),Q)
 dl,dh=quot(lo,hi,bl,bh)
 ts=torch.stack([t[:,0].square(),t[:,1].square()],dim=1)
 tl=torch.where((t[:,0]<=0)&(t[:,1]>=0),0,fd(ts.amin(1),Q));th=cd(ts.amax(1),Q)
 cl,_=quot(Q-th,Q-th,Q+th,Q+th)
 _,ch=quot(Q-tl,Q-tl,Q+tl,Q+tl)
 def sine_endpoint(x):
  p=x.square();pl=fd(p,Q);ph=cd(p,Q)
  return quot(2*x,2*x,Q+pl,Q+ph)
 sil,_=sine_endpoint(t[:,0]);_,sih=sine_endpoint(t[:,1])
 expected=torch.stack([lo+bl,hi+bh,lo-bh,hi-bl,fd(prod.amin(1),Q),cd(prod.amax(1),Q),
             dl,dh,sl,sh,cl.clamp(0,Q),ch.clamp(0,Q),sil.clamp(-Q,Q),sih.clamp(-Q,Q)],dim=1)
 match=torch.equal(expected,torch.tensor(values.tolist(),device=d,dtype=torch.int64))
 assert match
 co=constants();ct=torch.tensor(co.tolist(),device=d,dtype=torch.int64)
 radical3=(ct[0].square()<=3*Q*Q)&(ct[1].square()>3*Q*Q)
 radical6=(ct[2].square()<=6*Q*Q)&(ct[3].square()>6*Q*Q)
 assert radical3.item() and radical6.item()
 result={'passed':True,'interval_cases_checked':count,'operations':['addition','subtraction','multiplication','division','squaring','rational rotation enclosures'],
  'arithmetic':'signed GPU int64; dyadic endpoints Q=2^24; explicit outward floor/ceiling',
  'independent_engine':'PyTorch MPS int64','radical_enclosures_checked':True,
  'cpu_numerical_fallback':False,'constants':co.tolist()}
 dest=ROOT/'certified_spatial';dest.mkdir(exist_ok=True)
 (dest/'arithmetic_checks.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 return result

if __name__=='__main__':arithmetic_tests()
