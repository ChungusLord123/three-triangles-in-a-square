"""Independent stronger interval replay: exact rational rotation endpoints
computed with GPU 128-bit product comparisons, rather than rounded squares.
"""
import mlx.core as mx
from certified_intervals_gpu import HEADER,Q
from certified_model_gpu import build_kernel

REFERENCE_ROTATION=r'''
struct CU128 {ulong high;ulong low;};
inline CU128 cref_product(ulong a,ulong b){
 ulong p0=(a&0xffffffffUL)*b,p1=(a>>32)*b;
 ulong shifted=p1<<32,low=p0+shifted;
 return CU128{(p1>>32)+(low<p0?1UL:0UL),low};
}
inline bool cref_le(CU128 a,CU128 b){return a.high<b.high||(a.high==b.high&&a.low<=b.low);}
inline bool cref_eq(CU128 a,CU128 b){return a.high==b.high&&a.low==b.low;}
inline CI cref_ratio(CU128 numerator,ulong denominator){
 ulong lo=0,hi=(ulong)CQ+1;
 while(hi-lo>1){ulong m=(lo+hi)/2;if(cref_le(cref_product(denominator,m),numerator))lo=m;else hi=m;}
 bool exact=cref_eq(cref_product(denominator,lo),numerator);
 return CI{(long)lo,(long)(lo+(exact?0UL:1UL))};
}
inline CI cref_sine(long t){
 ulong a=(ulong)abs(t),qq=(ulong)CQ*(ulong)CQ;
 CI answer=cref_ratio(cref_product(qq,2*a),qq+a*a);
 return t<0?negative(answer):answer;
}
inline CI cref_cosine(long t){
 ulong a=(ulong)abs(t),qq=(ulong)CQ*(ulong)CQ;
 return cref_ratio(cref_product(qq-a*a,(ulong)CQ),qq+a*a);
}
inline void rational_rotation(CI t,thread CI& c,thread CI& s,thread bool& unsafe){
 if(t.lo<-CQ||t.hi>CQ){unsafe=true;c=CI{-CWIDE,CWIDE};s=c;return;}
 long largest=max(abs(t.lo),abs(t.hi));long smallest=has_zero(t)?0:min(abs(t.lo),abs(t.hi));
 c=CI{cref_cosine(largest).lo,cref_cosine(smallest).hi};
 s=CI{cref_sine(t.lo).lo,cref_sine(t.hi).hi};
}
'''

def reference_header():
 h=HEADER.replace('inline long fdiv(long n,long d){long q=n/d,r=n%d;return q-((r!=0)&&((r<0)!=(d<0)));}',
 '''inline long fdiv(long n,long d){ulong a=(ulong)abs(n),b=(ulong)abs(d),q=a/b,r=a%b;bool neg=(n<0)!=(d<0);return neg?-(long)q-(r!=0):(long)q;}''')
 h=h.replace('inline long cdiv(long n,long d){long q=n/d,r=n%d;return q+((r!=0)&&((r<0)==(d<0)));}',
 '''inline long cdiv(long n,long d){ulong a=(ulong)abs(n),b=(ulong)abs(d),q=a/b,r=a%b;bool neg=(n<0)!=(d<0);return neg?-(long)q:(long)q+(r!=0);}''')
 h=h.replace('inline void rational_rotation(', 'inline void primary_rotation_unused(')
 return h+REFERENCE_ROTATION

def reference_kernel():return build_kernel(header=reference_header(),name='independent_certified_spatial_reference')

TEST_SOURCE=r'''
uint i=thread_position_in_grid.x;if(i>=amount[0])return;
CI t=CI{input[2*i],input[2*i+1]},c,s;bool unsafe=false;
rational_rotation(t,c,s,unsafe);
out[4*i]=c.lo;out[4*i+1]=c.hi;out[4*i+2]=s.lo;out[4*i+3]=s.hi;
'''

def tests(count=8000):
 import os,json,torch
 os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
 from n3_branches_gpu import ROOT
 from certified_intervals_gpu import TEST_SOURCE as PRIMARY_SOURCE
 mx.random.seed(9941)
 t=mx.sort(mx.random.randint(-Q,Q+1,shape=(count,2),dtype=mx.int32),axis=1)
 k=mx.fast.metal_kernel(name='exact_rotation_reference_test',input_names=['input','amount'],output_names=['out'],source=TEST_SOURCE,header=reference_header())
 result=k(inputs=[t,mx.array([count],dtype=mx.uint32)],grid=(count,1,1),threadgroup=(256,1,1),
          output_shapes=[(count,4)],output_dtypes=[mx.int64])[0]
 # Primary enclosure must contain the independently computed reference bounds.
 primary_kernel=mx.fast.metal_kernel(name='primary_rotation_compare',input_names=['input','amount'],output_names=['out'],source=TEST_SOURCE,header=HEADER)
 p=primary_kernel(inputs=[t,mx.array([count],dtype=mx.uint32)],grid=(count,1,1),threadgroup=(256,1,1),output_shapes=[(count,4)],output_dtypes=[mx.int64])[0]
 contains=mx.all((p[:,0]<=result[:,0])&(p[:,1]>=result[:,1])&(p[:,2]<=result[:,2])&(p[:,3]>=result[:,3]))
 mx.eval(t,result,contains);assert contains.item()
 # Independently verify each scalar endpoint's floor/ceiling inequalities on
 # MPS using base-2^24 limbs. Every intermediate remains below 2^50.
 d=torch.device('mps');ti=torch.tensor(t.tolist(),device=d,dtype=torch.int64)
 ri=torch.tensor(result.tolist(),device=d,dtype=torch.int64)
 base=Q;mask=Q-1
 def product_limbs(den,q):
  d0=den&mask;d1=(den>>24)&mask;d2=den>>48
  a=d0*q;w0=a&mask;a=d1*q+(a>>24);w1=a&mask;a=d2*q+(a>>24)
  return torch.stack([a>>24,a&mask,w1,w0],dim=-1)
 def le(a,b):
  equal=torch.ones(a.shape[0],device=d,dtype=torch.bool);less=torch.zeros_like(equal)
  for j in range(4):less|=equal&(a[:,j]<b[:,j]);equal&=a[:,j]==b[:,j]
  return less|equal
 def check(numer,den,lo,hi):
  a=product_limbs(den,lo);b=product_limbs(den,lo+1)
  c=product_limbs(den,hi)
  assert le(a,numer).all().item() and (~le(b,numer)).all().item()
  assert le(numer,c).all().item() and ((hi==lo)|(hi==lo+1)).all().item()
  exact=(a==numer).all(-1)
  assert torch.equal(hi,lo+(~exact).to(torch.int64))
 amin=torch.where((ti[:,0]<=0)&(ti[:,1]>=0),0,torch.minimum(ti[:,0].abs(),ti[:,1].abs()))
 amax=torch.maximum(ti[:,0].abs(),ti[:,1].abs())
 for a,column in [(amax,0),(amin,1)]:
  den=Q*Q+a*a;num=Q*Q-a*a
  limbs=torch.stack([num>>48,(num>>24)&mask,num&mask,torch.zeros_like(num)],dim=-1)
  endpoint=ri[:,column]
  if column==0:lo=endpoint;hi=endpoint+((product_limbs(den,endpoint)!=limbs).any(-1)).to(torch.int64)
  else:
   hi=endpoint;previous=product_limbs(den,endpoint)
   lo=endpoint-((previous!=limbs).any(-1)).to(torch.int64)
  check(limbs,den,lo,hi)
 for a,column in [(ti[:,0],2),(ti[:,1],3)]:
  den=Q*Q+a*a;twice=2*a.abs()
  limbs=torch.stack([twice>>24,twice&mask,torch.zeros_like(a),torch.zeros_like(a)],dim=-1)
  endpoint=ri[:,column];is_lower=(a>=0) if column==2 else (a<0)
  magnitude=endpoint.abs()
  prod=product_limbs(den,magnitude);notexact=(prod!=limbs).any(-1).to(torch.int64)
  lo=torch.where(is_lower,magnitude,magnitude-notexact)
  hi=torch.where(is_lower,magnitude+notexact,magnitude)
  check(limbs,den,lo,hi)
 summary={'passed':True,'rotation_interval_cases':count,'primary_enclosures_contain_reference':True,
   'reference_method':'Exact rational endpoint division via 128-bit product comparisons on Metal',
   'independent_MPS_base24_limb_checks':True,'cpu_numerical_fallback':False}
 (ROOT/'certified_spatial'/'reference_arithmetic_checks.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':tests()
