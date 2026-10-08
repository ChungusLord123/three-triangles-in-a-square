"""Exact interval coefficients for a strict packing-improvement velocity.

V=(u0x,u0y,u1x,u1y,u2x,u2y,omega0,omega1,omega2), side'=-1.
Wall rows enforce -n_wall dot (u+omega J local_vertex)>1/2.
Contact rows enforce derivative of an owner supporting-plane gap >0:
 n dot (u_other-u_owner) + omega_owner Jn dot (c_other+local-c_owner)
 - omega_other Jn dot local_other >0.
Noncontact vertices on each chosen axis have strictly positive initial gaps.
Unstated contacts/walls are strictly inactive in the FULL active profile.
Therefore a certified strict velocity gives a smaller feasible square for a
sufficiently small positive time, contradicting an attained global minimum.
Failure to find or validate such a velocity never excludes a region.
"""
import mlx.core as mx
from certified_precision30_gpu import HEADER,REFERENCE_HEADER,MODEL_HEADER

M=42
SOURCE=r'''
uint row=thread_position_in_grid.x;if(row>=amount[0])return;
for(uint j=0;j<42*18;j++)coefficients[row*42*18+j]=0;
for(uint j=0;j<42;j++)bias[row*42+j]=0;
for(uint j=0;j<3;j++)chosen_axes[3*row+j]=-1;
bool unsafe=false,ok=true;uint count=0;
CI radical={constants_data[0],constants_data[1]},h=quotient(radical,point(2*CQ),unsafe),r=quotient(radical,point(6*CQ),unsafe);
CI center[3][2],angles[3];uint id=(uint)states[23*row],chart=(uint)states[23*row+1],wi=id/50653u,m=id%50653u;
uint code[3]={m/1369u,(m/37u)%37u,m%37u},left[3]={0,0,1},right[3]={1,2,2},walls[3][3];
CP base[3]={CP{point(CQ/2),negative(r)},CP{point(0),times_int(r,2)},CP{point(-CQ/2),negative(r)}};
CP normbase[3]={CP{h,point(CQ/2)},CP{negative(h),point(CQ/2)},CP{point(0),point(-CQ)}};
CP wallnormal[4]={CP{point(-CQ),point(0)},CP{point(CQ),point(0)},CP{point(0),point(-CQ)},CP{point(0),point(CQ)}};
CP radialbase[3]={CP{h,point(-CQ/2)},CP{point(0),point(CQ)},CP{negative(h),point(-CQ/2)}};
CP local[3][3],normals[3][3],radial[3][3];
for(uint i=0;i<3;i++){
 center[i][0]={states[23*row+3+4*i],states[23*row+4+4*i]};
 center[i][1]={states[23*row+5+4*i],states[23*row+6+4*i]};
 angles[i]={states[23*row+15+2*i],states[23*row+16+2*i]};
 CI c,s,cl,sl,ch,sh;rational_rotation(angles[i],c,s,unsafe);
 rational_rotation(point(angles[i].lo),cl,sl,unsafe);rational_rotation(point(angles[i].hi),ch,sh,unsafe);
 if((chart>>i)&1u){c=negative(c);s=negative(s);cl=negative(cl);sl=negative(sl);ch=negative(ch);sh=negative(sh);}
 uint mask=wmasks[triples[3*wi+i]];
 for(uint a=0;a<3;a++){
  walls[i][a]=(mask>>(4*a))&15u;
  local[i][a]=bounded_rotate(c,s,cl,sl,ch,sh,base[a],2*r.hi,unsafe);
  normals[i][a]=bounded_rotate(c,s,cl,sl,ch,sh,normbase[a],CQ,unsafe);
  radial[i][a]=bounded_rotate(c,s,cl,sl,ch,sh,radialbase[a],CQ,unsafe);
 }
}
for(uint i=0;i<3;i++)for(uint a=0;a<3;a++)for(uint w=0;w<4;w++)if(walls[i][a]&(1u<<w)){
 CI vector[9];for(uint j=0;j<9;j++)vector[j]=point(0);
 CP n=wallnormal[w];vector[2*i]=negative(n.x);vector[2*i+1]=negative(n.y);
 vector[6+i]=ci_sub(times(n.x,local[i][a].y,unsafe),times(n.y,local[i][a].x,unsafe));
 for(uint j=0;j<9;j++){coefficients[(row*42+count)*18+2*j]=vector[j].lo;coefficients[(row*42+count)*18+2*j+1]=vector[j].hi;}
 bias[row*42+count]=CQ/2;count++;
}
for(uint p=0;p<3;p++)if(code[p]){
 uint i=left[p],j=right[p],c=code[p],owner=i,other=j,edge=0,vertices[2]={0,0},number=1;int axis=-1;bool vertex_axis=false;
 if(c<10){
  uint z=c-1,va=z/3,vb=z%3;int forced=axes[3*row+p];
  for(int candidate=0;candidate<12;candidate++){
   if(forced>=0&&candidate!=forced)continue;
   uint o=candidate<6?i:j,b=candidate<6?j:i,e=uint(candidate%6),vowner=candidate<6?va:vb,vother=candidate<6?vb:va;
   bool use_vertex_axis=e>=3;uint feature=use_vertex_axis?e-3:e;
   if(use_vertex_axis?feature!=vowner:(feature!=vowner&&(feature+1)%3!=vowner))continue;
   CP n=use_vertex_axis?radial[o][feature]:normals[o][feature];CI offset=use_vertex_axis?times_int(r,2):r;
   CP dc={ci_sub(center[b][0],center[o][0]),ci_sub(center[b][1],center[o][1])};bool good=true;
   for(uint v=0;v<3;v++)if(v!=vother&&ci_sub(dot(n,addp(dc,local[b][v]),unsafe),offset).lo<=0)good=false;
   if(good){axis=candidate;owner=o;other=b;edge=feature;vertex_axis=use_vertex_axis;vertices[0]=vother;break;}
  }
  if(axis<0){ok=false;continue;}
 }else if(c<28){
  uint z=c-10;owner=z/9==0?i:j;other=z/9==0?j:i;edge=(z%9)/3;vertices[0]=z%3;
  axis=(owner==i?0:6)+int(edge);
 }else{
  uint z=c-28;edge=z/3;vertices[0]=z%3;vertices[1]=(vertices[0]+1)%3;number=2;axis=int(edge);
 }
 chosen_axes[3*row+p]=axis;
 CP n=vertex_axis?radial[owner][edge]:normals[owner][edge],jn={negative(n.y),n.x};
 CI offset=vertex_axis?times_int(r,2):r;
 CP dc={ci_sub(center[other][0],center[owner][0]),ci_sub(center[other][1],center[owner][1])};
 for(uint v=0;v<3;v++){
  bool contact=false;for(uint q=0;q<number;q++)if(v==vertices[q])contact=true;
  if(!contact&&ci_sub(dot(n,addp(dc,local[other][v]),unsafe),offset).lo<=0)ok=false;
 }
 for(uint q=0;q<number;q++){
  uint v=vertices[q];CI vector[9];for(uint k=0;k<9;k++)vector[k]=point(0);
  vector[2*owner]=negative(n.x);vector[2*owner+1]=negative(n.y);vector[2*other]=n.x;vector[2*other+1]=n.y;
  vector[6+owner]=dot(jn,addp(dc,local[other][v]),unsafe);
  vector[6+other]=negative(dot(jn,local[other][v],unsafe));
  if(c<10){uint ownvertex=(c-1)/3;if(owner==j)ownvertex=(c-1)%3;vector[6+owner]=vertex_axis?point(0):point(ownvertex==edge?-CQ/2:CQ/2);}
  else if(c<28){if(!intersect(vector[6+owner],CI{-CQ/2,CQ/2}))ok=false;}
  else{
   if(!intersect(vector[6+owner],CI{-3*CQ/2,3*CQ/2}))ok=false;
   vector[6+other]=point(q==0?-CQ/2:CQ/2);
  }
  for(uint k=0;k<9;k++){coefficients[(row*42+count)*18+2*k]=vector[k].lo;coefficients[(row*42+count)*18+2*k+1]=vector[k].hi;}
  count++;
 }
}
counts[row]=count;available[row]=(ok&&!unsafe)?1u:0u;
'''

def coefficients(states,inputs,axes=None,reference=False):
    if axes is None:axes=mx.full((states.shape[0],3),-1,dtype=mx.int32)
    k=mx.fast.metal_kernel(name='descent_coefficients_reference' if reference else 'descent_coefficients_primary',
      input_names=['states','wmasks','triples','constants_data','axes','amount'],
      output_names=['coefficients','bias','counts','chosen_axes','available'],source=SOURCE,
      header=(REFERENCE_HEADER if reference else HEADER)+MODEL_HEADER)
    return k(inputs=[states,inputs[0],inputs[1],inputs[3],axes,mx.array([states.shape[0]],dtype=mx.uint32)],
      grid=(states.shape[0],1,1),threadgroup=(64,1,1),
      output_shapes=[(states.shape[0],42,9,2),(states.shape[0],42),(states.shape[0],),(states.shape[0],3),(states.shape[0],)],
      output_dtypes=[mx.int64,mx.int64,mx.uint32,mx.int32,mx.uint32])

def strict_certificate(c,b,count,available,velocity):
    # Velocity is an exact dyadic numerator over 1024. All products fit int64.
    selected=mx.where(velocity[:,None,:]>=0,c[:,:,:,0],c[:,:,:,1])
    lhs=mx.sum(selected*velocity[:,None,:].astype(mx.int64),axis=2)
    required=b*1024
    active=mx.arange(M,dtype=mx.uint32)[None,:]<count[:,None]
    return (available>0)&mx.all((lhs>required)|~active,axis=1)
