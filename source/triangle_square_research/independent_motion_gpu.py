"""Motion checking extension to the standalone independent geometry model.

Imports only our independently written verifier. The extra wall-coordinate
intersection is a necessary equality before evaluating component spans.
"""
import hashlib
from pathlib import Path
import mlx.core as mx
from independent_geometry_gpu import Verifier,MOTION_SOURCE

class MotionVerifier(Verifier):
    def __init__(self,bits):
        super().__init__(bits)
        old='Vec point=plus_vec(Vec{domain[2*i],domain[2*i+1]},p[i].offset[v]);'
        new=r'''
            Vec point=plus_vec(Vec{domain[2*i],domain[2*i+1]},p[i].offset[v]);
            Range boundary=divide_small(domain[9],2);
            for(uint touched=0;touched<4;touched++)if(p[i].wall[v]&(1u<<touched)){
                Range coordinate=scale_int(boundary,(touched==0||touched==2)?-1:1);
                if(touched<2){if(!tighten(point.x,coordinate))return true;}
                else if(!tighten(point.y,coordinate))return true;
            }
        '''
        assert self.header.count(old)==1
        header=self.header.replace(old,new)
        self.motion=mx.fast.metal_kernel(name=f'independent_motion_with_wall_equalities_{bits}',
            input_names=['states','masks','triples','roots','velocities','control','amount'],output_names=['accepted'],
            header=header,source=MOTION_SOURCE)

def commitment():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
