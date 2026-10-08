"""Stronger rotation enclosures, without modifying the archived model.

For X,Y=R(theta(t))*a, dX/dt=-2Y/(1+t*t), dY/dt=2X/(1+t*t).
When an interval establishes a derivative's sign, extremal coordinate values
occur at the corresponding endpoints. Bounds intersect the original interval
enclosure and the vector's known coordinate radius. No sampled-angle rule.
"""
import mlx.core as mx
from certified_model_gpu import MODEL_HEADER as BASE_MODEL,SOURCE
from certified_intervals_gpu import HEADER
from certified_reference_gpu import reference_header

ROTATION_HELPERS=r'''
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
'''
OLD='for(uint a=0;a<3;a++){local[i][a]=rotate(c,s,base[a],unsafe);norm[i][a]=rotate(c,s,normal[a],unsafe);rad[i][a]=rotate(c,s,radial[a],unsafe);}'
NEW=r'''
  CI cl,sl,ch,sh;rational_rotation(point(v[6+i].lo),cl,sl,unsafe);rational_rotation(point(v[6+i].hi),ch,sh,unsafe);
  if(phase){cl=negative(cl);sl=negative(sl);ch=negative(ch);sh=negative(sh);}
  for(uint a=0;a<3;a++){
   local[i][a]=bounded_rotate(c,s,cl,sl,ch,sh,base[a],2*r.hi,unsafe);
   norm[i][a]=bounded_rotate(c,s,cl,sl,ch,sh,normal[a],CQ,unsafe);
   rad[i][a]=bounded_rotate(c,s,cl,sl,ch,sh,radial[a],CQ,unsafe);
  }
'''
assert BASE_MODEL.count(OLD)==1
MODEL_HEADER=ROTATION_HELPERS+BASE_MODEL.replace(OLD,NEW)

def build_kernel(reference=False):
    header=reference_header() if reference else HEADER
    return mx.fast.metal_kernel(name='certified_spatial_v2_reference' if reference else 'certified_spatial_v2',
      input_names=['states','wmasks','triples','angle_masks','constants_data','control','amount'],
      output_names=['updated','reasons','split_variable'],source=SOURCE,header=header+MODEL_HEADER)
