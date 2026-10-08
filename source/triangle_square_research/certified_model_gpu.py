"""Certified spatial contact model with exact dyadic interval contractors.
No Jacobian-rank or numerical-fitting rejection is used. Unknown/unsafe boxes
are retained. Rotation charts cover the whole circle without trig evaluation.
"""
import mlx.core as mx
from certified_intervals_gpu import HEADER,constants,Q
from contact_catalog_gpu import OUT

MODEL_HEADER=r'''
inline CP world_point(thread CI* v,uint triangle,CP local){return CP{ci_add(v[2*triangle],local.x),ci_add(v[2*triangle+1],local.y)};}
inline CI tangent_grid(int p,CI radical,thread bool& unsafe){
 int a=abs(p);CI z=a==0?point(0):a==1?ci_sub(point(2*CQ),radical):a==2?quotient(point(CQ),radical,unsafe):point(CQ);
 return p<0?negative(z):z;
}
inline bool equal_point_plane(thread CP& a,thread CP& b,CP n,CI height,thread bool& unsafe){
 if(!has_zero(n.x)&&min(abs(n.x.lo),abs(n.x.hi))>=CQ/32){
  CI d=quotient(ci_sub(height,times(n.y,ci_sub(b.y,a.y),unsafe)),n.x,unsafe);
  if(!intersect(b.x,ci_add(a.x,d))||!intersect(a.x,ci_sub(b.x,d)))return false;
 }
 if(!has_zero(n.y)&&min(abs(n.y.lo),abs(n.y.hi))>=CQ/32){
  CI d=quotient(ci_sub(height,times(n.x,ci_sub(b.x,a.x),unsafe)),n.y,unsafe);
  if(!intersect(b.y,ci_add(a.y,d))||!intersect(a.y,ci_sub(b.y,d)))return false;
 }
 return true;
}
inline bool outside_point_plane(CP a,thread CP& b,CP n,CI height,thread bool& unsafe){
 if(dot(n,subp(b,a),unsafe).hi<height.lo)return false;
 if(!has_zero(n.x)&&min(abs(n.x.lo),abs(n.x.hi))>=CQ/32){
  CI goal=ci_add(a.x,quotient(ci_sub(height,times(n.y,ci_sub(b.y,a.y),unsafe)),n.x,unsafe));
  if(n.x.lo>0)b.x.lo=max(b.x.lo,goal.lo);else b.x.hi=min(b.x.hi,goal.hi);
  if(empty(b.x))return false;
 }
 if(!has_zero(n.y)&&min(abs(n.y.lo),abs(n.y.hi))>=CQ/32){
  CI goal=ci_add(a.y,quotient(ci_sub(height,times(n.x,ci_sub(b.x,a.x),unsafe)),n.y,unsafe));
  if(n.y.lo>0)b.y.lo=max(b.y.lo,goal.lo);else b.y.hi=min(b.y.hi,goal.hi);
  if(empty(b.y))return false;
 }
 return true;
}
inline CI squared_distance(CP a,CP b,thread bool& unsafe){CP d=subp(a,b);return ci_add(square(d.x,unsafe),square(d.y,unsafe));}

inline int certified_evaluate(thread CI* v,uint id,uint chart,
  device const uint* wmasks,device const uint* triples,device const uint* angle_masks,
  device const int* constants_data,uint rounds,bool allow_sos,
  thread bool& unsafe,thread bool* fixed){
 uint windex=id/50653u,m=id%50653u;
 uint code[3]={m/1369u,(m/37u)%37u,m%37u};
 uint left[3]={0,0,1},right[3]={1,2,2};
 uint wid[3]={triples[3*windex],triples[3*windex+1],triples[3*windex+2]};
 uint wall[3][3];
 for(uint i=0;i<3;i++)for(uint a=0;a<3;a++)wall[i][a]=(wmasks[wid[i]]>>(4*a))&15u;
 if(allow_sos&&wmasks[wid[0]]==5&&wmasks[wid[1]]==34&&wmasks[wid[2]]==136&&code[0]==15&&code[1]==15&&code[2]==9&&constants_data[6]<constants_data[7])return 6;
 CI radical=CI{constants_data[0],constants_data[1]};
 CI h=quotient(radical,point(2*CQ),unsafe),r=quotient(radical,point(6*CQ),unsafe);
 CP base[3]={CP{point(CQ/2),negative(r)},CP{point(0),times_int(r,2)},CP{point(-CQ/2),negative(r)}};
 CP normal[3]={CP{h,point(CQ/2)},CP{negative(h),point(CQ/2)},CP{point(0),point(-CQ)}};
 CP radial[3]={CP{h,point(-CQ/2)},CP{point(0),point(CQ)},CP{negative(h),point(-CQ/2)}};
 CP walln[4]={CP{point(-CQ),point(0)},CP{point(CQ),point(0)},CP{point(0),point(-CQ)},CP{point(0),point(CQ)}};
 CP local[3][3],norm[3][3],rad[3][3];
 int full_edge[3]={-1,-1,-1},full_wall[3]={-1,-1,-1};
 for(uint i=0;i<3;i++){
  uint phase=(chart>>i)&1u;
  int pmin=4,pmax=-4;
  for(int p=-3;p<=3;p++){
   uint angle=uint((p+6*int(phase)+12)%12);
   if(angle_masks[wid[i]]&(1u<<angle)){pmin=min(pmin,p);pmax=max(pmax,p);}
  }
  if(pmax<pmin)return 7;
  CI lo=tangent_grid(pmin,radical,unsafe),hi=tangent_grid(pmax,radical,unsafe);
  if(!intersect(v[6+i],CI{lo.lo,hi.hi}))return 1;
  for(uint e=0;e<3;e++)for(uint w=0;w<4;w++)if((wall[i][e]&(1u<<w))&&(wall[i][(e+1)%3]&(1u<<w))){full_edge[i]=int(e);full_wall[i]=int(w);}
  CI c,s;fixed[i]=full_edge[i]>=0;
  if(fixed[i]){
   CP n=normal[full_edge[i]],nw=walln[full_wall[i]];
   c=dot(n,nw,unsafe);s=ci_sub(times(n.x,nw.y,unsafe),times(n.y,nw.x,unsafe));
   if((phase==0&&c.hi<0)||(phase==1&&c.lo>0))return 7;
   CI cc=phase?negative(c):c,ss=phase?negative(s):s;
   CI goal=quotient(ss,ci_add(point(CQ),cc),unsafe);
   if(!intersect(v[6+i],goal))return 1;
  }else{
   rational_rotation(v[6+i],c,s,unsafe);if(phase){c=negative(c);s=negative(s);}
  }
  for(uint a=0;a<3;a++){local[i][a]=rotate(c,s,base[a],unsafe);norm[i][a]=rotate(c,s,normal[a],unsafe);rad[i][a]=rotate(c,s,radial[a],unsafe);}
  if(fixed[i]){
   uint e=uint(full_edge[i]),next=(e+1)%3,opposite=(e+2)%3;CP nw=walln[full_wall[i]],tangent=CP{negative(nw.y),nw.x};
   CP rn=CP{times(r,nw.x,unsafe),times(r,nw.y,unsafe)};
   CP ht=CP{times(point(CQ/2),tangent.x,unsafe),times(point(CQ/2),tangent.y,unsafe)};
   local[i][e]=subp(rn,ht);local[i][next]=addp(rn,ht);
   local[i][opposite]=CP{times_int(rn.x,-2),times_int(rn.y,-2)};
   CP hn=CP{times(point(CQ/2),nw.x,unsafe),times(point(CQ/2),nw.y,unsafe)};
   CP altitude_t=CP{times(h,tangent.x,unsafe),times(h,tangent.y,unsafe)};
   rad[i][e]=subp(hn,altitude_t);rad[i][next]=addp(hn,altitude_t);rad[i][opposite]=CP{negative(nw.x),negative(nw.y)};
   norm[i][e]=nw;
   norm[i][next]=addp(CP{negative(hn.x),negative(hn.y)},altitude_t);
   norm[i][opposite]=subp(CP{negative(hn.x),negative(hn.y)},altitude_t);
  }
  for(uint a=0;a<3;a++)for(uint w=0;w<4;w++)if(wall[i][a]&(1u<<w))if(dot(rad[i][a],walln[w],unsafe).hi<CQ/2)return 4;
 }
 for(uint p=0;p<3;p++){
  uint i=left[p],j=right[p],c=code[p];
  if(c>0&&c<10){uint z=c-1;if(dot(rad[i][z/3],rad[j][z%3],unsafe).lo>CQ/2)return 4;}
  else if(c>=10&&c<28){uint z=c-10,e=(z%9)/3,a=z%3;uint owner=z/9==0?i:j,other=z/9==0?j:i;CP n=norm[owner][e];n.x=negative(n.x);n.y=negative(n.y);if(dot(n,rad[other][a],unsafe).hi<CQ/2)return 4;}
  else if(c>=28){uint z=c-28;CP n=addp(norm[i][z/3],norm[j][z%3]);if(!has_zero(n.x)||!has_zero(n.y))return 3;}
 }
 for(uint sweep=0;sweep<rounds;sweep++){
  if(times_int(square(square(v[9],unsafe),unsafe),16).hi<27*CQ)return 8;
  CI side_half=quotient(v[9],point(2*CQ),unsafe);
  CP pt[3][3];
  for(uint i=0;i<3;i++)for(uint a=0;a<3;a++){
   pt[i][a]=world_point(v,i,local[i][a]);
   if(!intersect(pt[i][a].x,CI{-side_half.hi,side_half.hi})||!intersect(pt[i][a].y,CI{-side_half.hi,side_half.hi}))return 2;
   for(uint w=0;w<4;w++)if(wall[i][a]&(1u<<w)){
    int sign=(w==0||w==2)?-1:1;CI target=times_int(side_half,sign);
    if(w<2){if(!intersect(pt[i][a].x,target))return 1;}
    else {if(!intersect(pt[i][a].y,target))return 1;}
   }
   if(full_edge[i]>=0){
    int w=full_wall[i],opposite=(full_edge[i]+2)%3,sign=(w==0||w==2)?-1:1;
    CI coordinate=times_int(a==uint(opposite)?ci_sub(side_half,h):side_half,sign);
    if(w<2){if(!intersect(pt[i][a].x,coordinate))return 1;}
    else{if(!intersect(pt[i][a].y,coordinate))return 1;}
   }
  }
  for(uint p=0;p<3;p++){
   uint i=left[p],j=right[p],c=code[p];
   if(c>0&&c<10){
    uint z=c-1,a=z/3,b=z%3;
    CI x=pt[i][a].x,y=pt[i][a].y;
    if(!intersect(x,pt[j][b].x)||!intersect(y,pt[j][b].y))return 1;
    pt[i][a].x=x;pt[j][b].x=x;pt[i][a].y=y;pt[j][b].y=y;
   }else if(c>=10&&c<28){
    uint z=c-10,e=(z%9)/3,a=z%3;uint owner=z/9==0?i:j,other=z/9==0?j:i;
    uint opposite=(e+2)%3;CP n=norm[owner][e];
    if(!equal_point_plane(pt[owner][opposite],pt[other][a],n,h,unsafe))return 1;
    if(!has_zero(ci_sub(dot(n,subp(pt[other][a],pt[owner][opposite]),unsafe),h)))return 3;
    CP tangent=CP{negative(n.y),n.x};CI along=dot(tangent,subp(pt[other][a],pt[owner][e]),unsafe);
    if(along.hi<0||along.lo>CQ)return 5;
    if(squared_distance(pt[other][a],pt[owner][opposite],unsafe).hi<3*CQ/4)return 4;
    for(uint b=0;b<3;b++)if(!outside_point_plane(pt[owner][opposite],pt[other][b],n,h,unsafe))return 4;
   }else if(c>=28){
    uint z=c-28,e=z/3,f=z%3,oa=(e+2)%3,ob=(f+2)%3;CP n=norm[i][e];
    if(!equal_point_plane(pt[i][oa],pt[j][ob],n,times_int(h,2),unsafe))return 1;
    if(!has_zero(ci_sub(dot(n,subp(pt[j][ob],pt[i][oa]),unsafe),times_int(h,2))))return 3;
    CP tangent=CP{negative(n.y),n.x};CI along=dot(tangent,subp(pt[j][f],pt[i][e]),unsafe);
    if(along.hi<0||along.lo>2*CQ)return 5;
    if(squared_distance(pt[j][ob],pt[i][oa],unsafe).hi<3*CQ)return 4;
    for(uint b=0;b<3;b++)if(!outside_point_plane(pt[i][oa],pt[j][b],n,h,unsafe))return 4;
    CP nb=norm[j][f];for(uint a=0;a<3;a++)if(!outside_point_plane(pt[j][ob],pt[i][a],nb,h,unsafe))return 4;
   }
  }
  // Unit triangle edge lengths and minimum projection width are redundant
  // exact geometric constraints, applied to tightened vertex enclosures.
  for(uint i=0;i<3;i++){
   long xl=min(pt[i][0].x.lo,min(pt[i][1].x.lo,pt[i][2].x.lo));
   long xh=max(pt[i][0].x.hi,max(pt[i][1].x.hi,pt[i][2].x.hi));
   long yl=min(pt[i][0].y.lo,min(pt[i][1].y.lo,pt[i][2].y.lo));
   long yh=max(pt[i][0].y.hi,max(pt[i][1].y.hi,pt[i][2].y.hi));
   if(xh-xl<h.lo||yh-yl<h.lo)return 4;
   for(uint a=0;a<3;a++){
    CI distance=squared_distance(pt[i][a],pt[i][(a+1)%3],unsafe);
    if(distance.lo>CQ||distance.hi<CQ)return 3;
    if(!intersect(v[2*i],ci_sub(pt[i][a].x,local[i][a].x))||!intersect(v[2*i+1],ci_sub(pt[i][a].y,local[i][a].y)))return 1;
    for(uint w=0;w<4;w++)if(wall[i][a]&(1u<<w)){
     int sign=(w==0||w==2)?-1:1;
     CI coordinate=w<2?pt[i][a].x:pt[i][a].y;
     if(!intersect(v[9],times_int(coordinate,2*sign)))return 1;
    }
   }
  }
  // General non-overlap, including no-contact and vertex-vertex pairs.
  for(uint p=0;p<3;p++){
   uint i=left[p],j=right[p];long possible=-CWIDE;
   for(uint side=0;side<2;side++){
    uint owner=side?j:i,other=side?i:j;
    for(uint e=0;e<3;e++){
     long upper=CWIDE;CP n=norm[owner][e];
     for(uint a=0;a<3;a++)upper=min(upper,ci_sub(dot(n,subp(pt[other][a],pt[owner][(e+2)%3]),unsafe),h).hi);
     possible=max(possible,upper);
    }
   }
   if(possible<0)return 4;
  }
 }
 return 0;
}
'''

SOURCE=r'''
uint row=thread_position_in_grid.x;if(row>=amount[0])return;
thread CI vars[10];bool unsafe=false;bool fixed[3]={false,false,false};
uint id=(uint)states[23*row],chart=(uint)states[23*row+1];
for(uint j=0;j<10;j++)vars[j]=CI{states[23*row+3+2*j],states[23*row+4+2*j]};
int result=certified_evaluate(vars,id,chart,wmasks,triples,angle_masks,constants_data,control[0],control[1]!=0,unsafe,fixed);
if(unsafe)result=9;
reasons[row]=(uint)result;
for(uint j=0;j<23;j++)updated[23*row+j]=states[23*row+j];
if(result==0){for(uint j=0;j<10;j++){updated[23*row+3+2*j]=(int)vars[j].lo;updated[23*row+4+2*j]=(int)vars[j].hi;}}
long best=-1;int variable=-1;
if(result==0||result==9)for(uint j=0;j<10;j++){
 long width=(long)updated[23*row+4+2*j]-(long)updated[23*row+3+2*j];
 if(j>=6&&j<9&&fixed[j-6])width=0;else if(j>=6)width*=2;
 if(width>best){best=width;variable=(int)j;}
}
split_variable[row]=variable;
'''

def build_kernel(header=HEADER,model_header=MODEL_HEADER,name='certified_spatial_model'):
 return mx.fast.metal_kernel(name=name,input_names=['states','wmasks','triples','angle_masks','constants_data','control','amount'],
              output_names=['updated','reasons','split_variable'],source=SOURCE,header=header+model_header)

def model_inputs(numerator=14783,denominator=10000):
 masks=mx.load(str(OUT/'wall_masks.npy')).astype(mx.uint32)
 triples=mx.load(str(OUT/'wall_triplets.npy')).astype(mx.uint32)
 allowed=mx.load(str(OUT/'wall_angles.npy'))
 angle_masks=mx.sum(allowed.astype(mx.uint32)*(mx.array(1,dtype=mx.uint32)<<mx.arange(12,dtype=mx.uint32)[None,:]),axis=1)
 return masks,triples,angle_masks,constants(numerator,denominator)

def evaluate(states,inputs,kernel=None,rounds=4,allow_sos=True):
 if kernel is None:kernel=build_kernel()
 n=states.shape[0]
 return kernel(inputs=[states,*inputs,mx.array([rounds,int(allow_sos)],dtype=mx.uint32),mx.array([n],dtype=mx.uint32)],
      grid=(n,1,1),threadgroup=(128,1,1),output_shapes=[states.shape,(n,),(n,)],output_dtypes=[mx.int32,mx.uint32,mx.int32])
