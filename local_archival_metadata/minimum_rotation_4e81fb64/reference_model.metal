
constant long CQ=16777216L;
constant long CWIDE=1099511627776L;
struct CI {long lo;long hi;};
struct CP {CI x;CI y;};
inline CI point(long x){return CI{x,x};}
inline long fdiv(long n,long d){ulong a=(ulong)abs(n),b=(ulong)abs(d),q=a/b,r=a%b;bool neg=(n<0)!=(d<0);return neg?-(long)q-(r!=0):(long)q;}
inline long cdiv(long n,long d){ulong a=(ulong)abs(n),b=(ulong)abs(d),q=a/b,r=a%b;bool neg=(n<0)!=(d<0);return neg?-(long)q:(long)q+(r!=0);}
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
inline void primary_rotation_unused(CI t,thread CI& c,thread CI& s,thread bool& unsafe){
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

inline CP bounded_rotate(CI c,CI s,CI cl,CI sl,CI ch,CI sh,CP a,long radius,thread bool& unsafe){
 CP all=rotate(c,s,a,unsafe),low=rotate(cl,sl,a,unsafe),high=rotate(ch,sh,a,unsafe);
 CI old_x=all.x,old_y=all.y;
 if(old_y.lo>=0)intersect(all.x,CI{high.x.lo,low.x.hi});
 else if(old_y.hi<=0)intersect(all.x,CI{low.x.lo,high.x.hi});
 if(old_x.lo>=0)intersect(all.y,CI{low.y.lo,high.y.hi});
 else if(old_x.hi<=0)intersect(all.y,CI{high.y.lo,low.y.hi});
 intersect(all.x,CI{-radius,radius});intersect(all.y,CI{-radius,radius});
 return all;
}

uint row=thread_position_in_grid.x;if(row>=amount[0])return;
bool unsafe=false;CI v[10];for(uint j=0;j<10;j++)v[j]=CI{states[23*row+3+2*j],states[23*row+4+2*j]};
uint id=(uint)states[23*row],chart=(uint)states[23*row+1],wi=id/50653u,m=id%50653u;
uint code[3]={m/1369u,(m/37u)%37u,m%37u},left[3]={0,0,1},right[3]={1,2,2};
uint wall[3][3],connection[3]={1u,2u,4u};
for(uint p=0;p<3;p++)if(code[p]){connection[left[p]]|=1u<<right[p];connection[right[p]]|=1u<<left[p];}
for(uint k=0;k<3;k++)for(uint i=0;i<3;i++)if(connection[i]&(1u<<k))connection[i]|=connection[k];
CI radical={constants_data[0],constants_data[1]},r=quotient(radical,point(6*CQ),unsafe);
CP base[3]={CP{point(CQ/2),negative(r)},CP{point(0),times_int(r,2)},CP{point(-CQ/2),negative(r)}};
CP pt[3][3];CI sidehalf=quotient(v[9],point(2*CQ),unsafe);
for(uint i=0;i<3;i++){
 uint mask=wmasks[triples[3*wi+i]];
 CI c,s,cl,sl,ch,sh;rational_rotation(v[6+i],c,s,unsafe);
 rational_rotation(point(v[6+i].lo),cl,sl,unsafe);rational_rotation(point(v[6+i].hi),ch,sh,unsafe);
 if((chart>>i)&1u){c=negative(c);s=negative(s);cl=negative(cl);sl=negative(sl);ch=negative(ch);sh=negative(sh);}
 for(uint a=0;a<3;a++){
  wall[i][a]=(mask>>(4*a))&15u;CP b=bounded_rotate(c,s,cl,sl,ch,sh,base[a],2*r.hi,unsafe);
  pt[i][a]=CP{ci_add(v[2*i],b.x),ci_add(v[2*i+1],b.y)};
  for(uint w=0;w<4;w++)if(wall[i][a]&(1u<<w)){
   CI target=times_int(sidehalf,(w==0||w==2)?-1:1);
   if(w<2){if(!intersect(pt[i][a].x,target))unsafe=true;}
   else if(!intersect(pt[i][a].y,target))unsafe=true;
  }
 }
}
bool candidate=false;
for(uint component=0;component<3;component++){
 long low[4]={CWIDE,CWIDE,CWIDE,CWIDE},high[4]={-CWIDE,-CWIDE,-CWIDE,-CWIDE};uint flags=0;
 for(uint i=0;i<3;i++)if(connection[component]&(1u<<i))for(uint a=0;a<3;a++)for(uint w=0;w<4;w++)if(wall[i][a]&(1u<<w)){
  CI z=w<2?pt[i][a].y:pt[i][a].x;low[w]=min(low[w],z.lo);high[w]=max(high[w],z.hi);flags|=1u<<w;
 }
 bool horizontal=(flags&3u)==3u,vertical=(flags&12u)==12u;
 if(horizontal&&vertical){
  bool positive_descent=high[0]<=low[1]&&high[3]<=low[2];
  bool negative_descent=low[0]>=high[1]&&low[3]>=high[2];
  if(!positive_descent&&!negative_descent)candidate=true;
 }else if(horizontal){if(high[0]>low[1]&&high[1]>low[0])candidate=true;}
 else if(vertical){if(high[2]>low[3]&&high[3]>low[2])candidate=true;}
}
keep[row]=(candidate||unsafe)?1u:0u;
