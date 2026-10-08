"""Higher-precision dyadic intervals, preserving the archived Q=2^24 model.

Q=2^30 keeps all stored side/centroid/rotation bounds inside int32. Radical
constants are int64. Product operands are capped below sqrt(INT64_MAX), and
division numerators below floor(INT64_MAX/Q); uncertainty always retains a box.
"""
import mlx.core as mx
from certified_intervals_gpu import HEADER as OLD_HEADER,CONSTANT_SOURCE,TEST_SOURCE
from certified_reference_gpu import reference_header
from certified_model_v2_gpu import MODEL_HEADER as OLD_MODEL
from certified_model_gpu import SOURCE
from spatial_minimum_rotation_gpu import SOURCE as MINIMUM_SOURCE
from contact_catalog_gpu import OUT

Q=1073741824
SHIFT=6

def upgrade_header(h):
    h=h.replace('>1073741824L','>3037000499L')
    h=h.replace('>68719476736L','>8589934591L')
    h=h.replace('constant long CQ=16777216L;','constant long CQ=1073741824L;')
    assert 'constant long CQ=1073741824L;' in h
    return h

HEADER=upgrade_header(OLD_HEADER)
REFERENCE_HEADER=upgrade_header(reference_header())
MODEL_HEADER=OLD_MODEL.replace('device const int* constants_data','device const long* constants_data')
CONSTANTS_SOURCE=CONSTANT_SOURCE.replace('(int)','')

def constants(numerator,denominator):
    k=mx.fast.metal_kernel(name='precision30_constants',input_names=['fraction'],output_names=['out'],source=CONSTANTS_SOURCE,header=HEADER)
    a=k(inputs=[mx.array([numerator,denominator],dtype=mx.int64)],grid=(1,1,1),threadgroup=(1,1,1),
      output_shapes=[(10,)],output_dtypes=[mx.int64])[0]
    mx.eval(a);assert a[9].item()==0
    return a

def model_inputs(numerator,denominator):
    masks=mx.load(str(OUT/'wall_masks.npy')).astype(mx.uint32)
    triples=mx.load(str(OUT/'wall_triplets.npy')).astype(mx.uint32)
    allowed=mx.load(str(OUT/'wall_angles.npy'))
    angle_masks=mx.sum(allowed.astype(mx.uint32)*(mx.array(1,dtype=mx.uint32)<<mx.arange(12,dtype=mx.uint32)[None,:]),axis=1)
    return masks,triples,angle_masks,constants(numerator,denominator)

def build_kernel(reference=False):
    return mx.fast.metal_kernel(name='precision30_reference' if reference else 'precision30_primary',
      input_names=['states','wmasks','triples','angle_masks','constants_data','control','amount'],
      output_names=['updated','reasons','split_variable'],source=SOURCE,
      header=(REFERENCE_HEADER if reference else HEADER)+MODEL_HEADER)

def evaluate_minimum(states,inputs,reference=False):
    k=mx.fast.metal_kernel(name='precision30_minimum_reference' if reference else 'precision30_minimum_primary',
      input_names=['states','wmasks','triples','constants_data','amount'],output_names=['keep'],
      source=MINIMUM_SOURCE,header=(REFERENCE_HEADER if reference else HEADER)+MODEL_HEADER)
    return k(inputs=[states,inputs[0],inputs[1],inputs[3],mx.array([states.shape[0]],dtype=mx.uint32)],
      grid=(states.shape[0],1,1),threadgroup=(128,1,1),output_shapes=[(states.shape[0],)],output_dtypes=[mx.uint32])[0]
