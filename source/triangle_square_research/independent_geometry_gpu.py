"""Standalone GPU geometry verifier.

This module imports no search, contact-model, interval, motion or polynomial
code. It reads only the public row encoding and wall tables. A different
interval implementation and a separately written halfspace model verify
infeasibility, necessary contractions, rigid-component rotations and supplied
dyadic motion witnesses. A failed check is UNKNOWN, never proof by failure.
NumPy is used only to decode/copy file bytes before GPU transfer.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
import hashlib,json
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.gpu)
ROOT=Path(__file__).resolve().parent

ARITHMETIC=r'''
constant uint BITS=__BITS__;
constant long UNIT=1L<<BITS;
constant long BIG=1L<<40;
struct Range { long lower; long upper; };
struct Vec { Range x; Range y; };
struct Wide { ulong upper; ulong lower; };
struct Orientation { Range c;Range s;Range cl;Range sl;Range ch;Range sh; };
inline Range value(long a){return {a,a};}
inline bool impossible(Range a){return a.lower>a.upper;}
inline bool includes_zero(Range a){return a.lower<=0&&a.upper>=0;}
inline long floor_div(long n,long d){
    bool negative=(n<0)!=(d<0);ulong a=ulong(abs(n)),b=ulong(abs(d));
    ulong q=a/b;bool residual=(a-q*b)!=0;
    return negative?-long(q)-long(residual):long(q);
}
inline long ceil_div(long n,long d){return -floor_div(-n,d);}
inline Range add(Range a,Range b){return {a.lower+b.lower,a.upper+b.upper};}
inline Range subtract(Range a,Range b){return {a.lower-b.upper,a.upper-b.lower};}
inline Range negate(Range a){return {-a.upper,-a.lower};}
inline Range scale_int(Range a,long k){return k<0?Range{a.upper*k,a.lower*k}:Range{a.lower*k,a.upper*k};}
inline Range divide_small(Range a,long d){return {floor_div(a.lower,d),ceil_div(a.upper,d)};}
inline bool tighten(thread Range& a,Range b){a.lower=max(a.lower,b.lower);a.upper=min(a.upper,b.upper);return !impossible(a);}
inline Range multiply(Range a,Range b,thread bool& uncertain){
    if(max(abs(a.lower),abs(a.upper))>3037000499L||max(abs(b.lower),abs(b.upper))>3037000499L){uncertain=true;return {-BIG,BIG};}
    long products[4]={a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper};
    long small=products[0],large=products[0];for(uint i=1;i<4;i++){small=min(small,products[i]);large=max(large,products[i]);}
    return {small>>BITS,-((-large)>>BITS)};
}
inline Range quotient(Range a,Range b,thread bool& uncertain){
    if(includes_zero(b)||max(abs(a.lower),abs(a.upper))>(9223372036854775807L>>BITS)){uncertain=true;return {-BIG,BIG};}
    long endpoints[2]={a.lower*UNIT,a.upper*UNIT},den[2]={b.lower,b.upper};
    long small=9223372036854775807L;long large=-9223372036854775807L;
    for(uint i=0;i<2;i++)for(uint j=0;j<2;j++){small=min(small,floor_div(endpoints[i],den[j]));large=max(large,ceil_div(endpoints[i],den[j]));}
    return {small,large};
}
inline Range squared(Range a,thread bool& uncertain){
    if(max(abs(a.lower),abs(a.upper))>3037000499L){uncertain=true;return {0,BIG};}
    long first=a.lower*a.lower,last=a.upper*a.upper;
    return {includes_zero(a)?0:min(first,last)>>BITS,-((-max(first,last))>>BITS)};
}
inline Vec plus_vec(Vec a,Vec b){return {add(a.x,b.x),add(a.y,b.y)};}
inline Vec minus_vec(Vec a,Vec b){return {subtract(a.x,b.x),subtract(a.y,b.y)};}
inline Vec quarter_turn(Vec a){return {negate(a.y),a.x};}
inline Range inner(Vec a,Vec b,thread bool& uncertain){return add(multiply(a.x,b.x,uncertain),multiply(a.y,b.y,uncertain));}
inline Range distance2(Vec a,Vec b,thread bool& uncertain){Vec d=minus_vec(a,b);return add(squared(d.x,uncertain),squared(d.y,uncertain));}
inline Wide full_product(ulong a,ulong b){
    ulong mask=0xffffffffUL,a0=a&mask,a1=a>>32,b0=b&mask,b1=b>>32;
    ulong p=a0*b0,q=a0*b1,r=a1*b0,s=a1*b1;
    ulong middle=(p>>32)+(q&mask)+(r&mask);
    return {s+(q>>32)+(r>>32)+(middle>>32),(p&mask)|(middle<<32)};
}
inline bool at_most(Wide a,Wide b){return a.upper<b.upper||(a.upper==b.upper&&a.lower<=b.lower);}
inline bool identical(Wide a,Wide b){return a.upper==b.upper&&a.lower==b.lower;}
inline Range unit_ratio(Wide numerator,ulong denominator){
    ulong left=0,right=ulong(UNIT)+1;
    while(right-left>1){ulong mid=left+((right-left)>>1);if(at_most(full_product(mid,denominator),numerator))left=mid;else right=mid;}
    bool exact=identical(full_product(left,denominator),numerator);return {long(left),long(left+ulong(!exact))};
}
inline void rotation_endpoint(long t,thread Range& c,thread Range& s){
    ulong absolute=ulong(abs(t)),q=ulong(UNIT),qq=q*q,tt=absolute*absolute;
    c=unit_ratio(full_product(q,qq-tt),qq+tt);
    s=unit_ratio(full_product(2*absolute,qq),qq+tt);if(t<0)s=negate(s);
}
inline void rotation_range(Range t,thread Range& c,thread Range& s,thread bool& uncertain){
    if(t.lower<-UNIT||t.upper>UNIT){uncertain=true;c={-BIG,BIG};s=c;return;}
    long far=max(abs(t.lower),abs(t.upper)),near=includes_zero(t)?0:min(abs(t.lower),abs(t.upper));
    Range cf,sf,cn,sn,cl,sl,ch,sh;
    rotation_endpoint(far,cf,sf);rotation_endpoint(near,cn,sn);
    rotation_endpoint(t.lower,cl,sl);rotation_endpoint(t.upper,ch,sh);
    c={cf.lower,cn.upper};s={sl.lower,sh.upper};
}
inline Vec matrix_vec(Range c,Range s,Vec p,thread bool& uncertain){
    return {subtract(multiply(c,p.x,uncertain),multiply(s,p.y,uncertain)),add(multiply(s,p.x,uncertain),multiply(c,p.y,uncertain))};
}
inline Orientation orientation(Range angle,uint half_value,thread bool& uncertain){
    Orientation o;
    if(angle.lower<-UNIT||angle.upper>UNIT){uncertain=true;}
    rotation_endpoint(angle.lower,o.cl,o.sl);rotation_endpoint(angle.upper,o.ch,o.sh);
    o.c={min(o.cl.lower,o.ch.lower),includes_zero(angle)?UNIT:max(o.cl.upper,o.ch.upper)};
    o.s={o.sl.lower,o.sh.upper};
    if(half_value){o.c=negate(o.c);o.s=negate(o.s);o.cl=negate(o.cl);o.sl=negate(o.sl);o.ch=negate(o.ch);o.sh=negate(o.sh);}
    return o;
}
inline Vec projected_vector(Orientation o,Vec base,long radius,thread bool& uncertain){
    Vec enclosure=matrix_vec(o.c,o.s,base,uncertain),left=matrix_vec(o.cl,o.sl,base,uncertain),right=matrix_vec(o.ch,o.sh,base,uncertain);
    Range old_x=enclosure.x,old_y=enclosure.y;
    if(old_y.lower>=0)tighten(enclosure.x,{right.x.lower,left.x.upper});
    if(old_y.upper<=0)tighten(enclosure.x,{left.x.lower,right.x.upper});
    if(old_x.lower>=0)tighten(enclosure.y,{left.y.lower,right.y.upper});
    if(old_x.upper<=0)tighten(enclosure.y,{right.y.lower,left.y.upper});
    tighten(enclosure.x,{-radius,radius});tighten(enclosure.y,{-radius,radius});return enclosure;
}
inline ulong radical_floor(uint k){
    ulong q=ulong(UNIT),target=ulong(k)*q*q,left=0,right=3*q;
    while(right-left>1){ulong mid=left+((right-left)>>1);if(mid*mid<=target)left=mid;else right=mid;}
    return left;
}
'''

GEOMETRY=r'''
struct Piece { Vec center; Range angle; uint half_value; uint wall[3]; Vec offset[3]; Vec edge[3]; Vec radial[3]; int base_edge;int base_wall; };
inline uint support_angles(uint mask){
    int radial[3]={11,3,7},direction[4]={6,0,9,3};uint permitted=0;
    for(int turn=0;turn<12;turn++){
        bool valid=true;
        for(uint v=0;v<3;v++)for(uint w=0;w<4;w++)if(mask&(1u<<(4*v+w))){
            int difference=(direction[w]-radial[v]-turn+48)%12;difference=min(difference,12-difference);if(difference>2)valid=false;
        }
        if(valid)permitted|=1u<<turn;
    }return permitted;
}
inline Range tangent_endpoint(int index,Range root3,thread bool& uncertain){
    int absolute=abs(index);Range a=value(0);
    if(absolute==1)a=subtract(value(2*UNIT),root3);
    if(absolute==2)a=quotient(value(UNIT),root3,uncertain);
    if(absolute==3)a=value(UNIT);return index<0?negate(a):a;
}
inline Vec outward_wall(uint wall){return wall==0?Vec{value(-UNIT),value(0)}:wall==1?Vec{value(UNIT),value(0)}:wall==2?Vec{value(0),value(-UNIT)}:Vec{value(0),value(UNIT)};}
inline int initialise(thread Range* domain,uint id,uint chart,device const uint* masks,device const uint* triples,
    device const long* roots,thread Piece* pieces,thread bool& uncertain){
    uint tuple=id/50653u;Range root3={roots[0],roots[1]},h=divide_small(root3,2),r=divide_small(root3,6);
    Vec corners[3]={Vec{value(UNIT/2),negate(r)},Vec{value(0),scale_int(r,2)},Vec{value(-UNIT/2),negate(r)}};
    Vec edge_normals[3]={Vec{h,value(UNIT/2)},Vec{negate(h),value(UNIT/2)},Vec{value(0),value(-UNIT)}};
    Vec radial_normals[3]={Vec{h,value(-UNIT/2)},Vec{value(0),value(UNIT)},Vec{negate(h),value(-UNIT/2)}};
    for(uint i=0;i<3;i++){
        uint mask=masks[triples[3*tuple+i]],allow=support_angles(mask),half_value=(chart>>i)&1;
        int first=4,last=-4;
        for(int k=-3;k<=3;k++)if(allow&(1u<<uint((k+6*int(half_value)+12)%12))){first=min(first,k);last=max(last,k);}
        if(first>last)return 1;
        Range a=tangent_endpoint(first,root3,uncertain),b=tangent_endpoint(last,root3,uncertain);
        if(!tighten(domain[6+i],{a.lower,b.upper}))return 1;
        pieces[i].center={domain[2*i],domain[2*i+1]};pieces[i].half_value=half_value;
        pieces[i].base_edge=-1;pieces[i].base_wall=-1;
        for(uint v=0;v<3;v++)pieces[i].wall[v]=(mask>>(4*v))&15;
        for(uint e=0;e<3;e++)for(uint w=0;w<4;w++)if((pieces[i].wall[e]&(1u<<w))&&(pieces[i].wall[(e+1)%3]&(1u<<w))){pieces[i].base_edge=int(e);pieces[i].base_wall=int(w);}
        if(pieces[i].base_edge>=0){
            Vec n=edge_normals[pieces[i].base_edge],nw=outward_wall(uint(pieces[i].base_wall));
            Range c=inner(n,nw,uncertain),s=subtract(multiply(n.x,nw.y,uncertain),multiply(n.y,nw.x,uncertain));
            if((!half_value&&c.upper<0)||(half_value&&c.lower>0))return 1;
            if(half_value){c=negate(c);s=negate(s);}
            if(!tighten(domain[6+i],quotient(s,add(value(UNIT),c),uncertain)))return 1;
        }
        pieces[i].angle=domain[6+i];
        if(pieces[i].base_edge<0){
            Orientation cached=orientation(domain[6+i],half_value,uncertain);
            for(uint v=0;v<3;v++){
                pieces[i].offset[v]=projected_vector(cached,corners[v],2*r.upper,uncertain);
                pieces[i].edge[v]=projected_vector(cached,edge_normals[v],UNIT,uncertain);
                pieces[i].radial[v]=projected_vector(cached,radial_normals[v],UNIT,uncertain);
            }
        }else{
            uint e=uint(pieces[i].base_edge),next=(e+1)%3,opposite=(e+2)%3;
            Vec normal=outward_wall(uint(pieces[i].base_wall)),tangent=quarter_turn(normal);
            Vec mid={multiply(r,normal.x,uncertain),multiply(r,normal.y,uncertain)};
            Vec end={divide_small(tangent.x,2),divide_small(tangent.y,2)};
            pieces[i].offset[e]=minus_vec(mid,end);pieces[i].offset[next]=plus_vec(mid,end);
            pieces[i].offset[opposite]={scale_int(mid.x,-2),scale_int(mid.y,-2)};
            Vec half_normal={divide_small(normal.x,2),divide_small(normal.y,2)},height_tangent={multiply(h,tangent.x,uncertain),multiply(h,tangent.y,uncertain)};
            pieces[i].edge[e]=normal;
            pieces[i].edge[next]=plus_vec(Vec{negate(half_normal.x),negate(half_normal.y)},height_tangent);
            pieces[i].edge[opposite]=minus_vec(Vec{negate(half_normal.x),negate(half_normal.y)},height_tangent);
            pieces[i].radial[e]=minus_vec(half_normal,height_tangent);pieces[i].radial[next]=plus_vec(half_normal,height_tangent);
            pieces[i].radial[opposite]={negate(normal.x),negate(normal.y)};
        }
    }return 0;
}
inline bool plane_equation(thread Vec& a,thread Vec& b,Vec n,Range height,thread bool& uncertain){
    if(!includes_zero(subtract(inner(n,minus_vec(b,a),uncertain),height)))return false;
    for(uint coordinate=0;coordinate<2;coordinate++){
        Range nx=coordinate?n.y:n.x,ny=coordinate?n.x:n.y;
        if(includes_zero(nx)||min(abs(nx.lower),abs(nx.upper))<UNIT/64)continue;
        Range ax=coordinate?a.y:a.x,bx=coordinate?b.y:b.x;
        Range ay=coordinate?a.x:a.y,by=coordinate?b.x:b.y;
        Range difference=quotient(subtract(height,multiply(ny,subtract(by,ay),uncertain)),nx,uncertain);
        if(!tighten(bx,add(ax,difference))||!tighten(ax,subtract(bx,difference)))return false;
        if(coordinate){a.y=ax;b.y=bx;}else{a.x=ax;b.x=bx;}
    }return true;
}
inline bool outside_halfspace(Vec a,thread Vec& b,Vec n,Range height,thread bool& uncertain){
    if(inner(n,minus_vec(b,a),uncertain).upper<height.lower)return false;
    for(uint coordinate=0;coordinate<2;coordinate++){
        Range nx=coordinate?n.y:n.x,ny=coordinate?n.x:n.y;
        if(includes_zero(nx)||min(abs(nx.lower),abs(nx.upper))<UNIT/64)continue;
        Range ax=coordinate?a.y:a.x,bx=coordinate?b.y:b.x;
        Range ay=coordinate?a.x:a.y,by=coordinate?b.x:b.y;
        Range threshold=add(ax,quotient(subtract(height,multiply(ny,subtract(by,ay),uncertain)),nx,uncertain));
        if(nx.lower>0)bx.lower=max(bx.lower,threshold.lower);else bx.upper=min(bx.upper,threshold.upper);
        if(impossible(bx))return false;if(coordinate)b.y=bx;else b.x=bx;
    }return true;
}
inline int contract(thread Range* domain,uint id,uint chart,device const uint* masks,device const uint* triples,
    device const long* roots,uint iterations,thread bool& uncertain){
    Piece p[3];int setup=initialise(domain,id,chart,masks,triples,roots,p,uncertain);if(setup)return setup;
    Range h=divide_small(Range{roots[0],roots[1]},2);uint mode=id%50653u;
    uint contacts[3]={mode/1369u,(mode/37u)%37u,mode%37u},ii[3]={0,0,1},jj[3]={1,2,2};
    uint tuple=id/50653u;
    if(masks[triples[3*tuple]]==5&&masks[triples[3*tuple+1]]==34&&masks[triples[3*tuple+2]]==136&&
       contacts[0]==15&&contacts[1]==15&&contacts[2]==9&&domain[9].lower>h.upper&&domain[9].upper<roots[6])return 6;
    for(uint i=0;i<3;i++)for(uint v=0;v<3;v++)for(uint w=0;w<4;w++)if(p[i].wall[v]&(1u<<w))
        if(inner(p[i].radial[v],outward_wall(w),uncertain).upper<UNIT/2)return 2;
    for(uint pair=0;pair<3;pair++){
        uint a=ii[pair],b=jj[pair],c=contacts[pair];
        if(c>0&&c<10){uint z=c-1;if(inner(p[a].radial[z/3],p[b].radial[z%3],uncertain).lower>UNIT/2)return 2;}
        if(c>=10&&c<28){uint z=c-10,o=z<9?a:b,t=z<9?b:a,e=(z%9)/3,v=z%3;Vec n=p[o].edge[e];
            if(inner(Vec{negate(n.x),negate(n.y)},p[t].radial[v],uncertain).upper<UNIT/2)return 2;}
        if(c>=28){uint z=c-28;Vec total=plus_vec(p[a].edge[z/3],p[b].edge[z%3]);if(!includes_zero(total.x)||!includes_zero(total.y))return 2;}
    }
    for(uint pass=0;pass<iterations;pass++){
        if(scale_int(squared(domain[9],uncertain),2).upper<scale_int(h,3).lower)return 3;
        Range half_value=divide_small(domain[9],2);Vec points[3][3];
        for(uint i=0;i<3;i++)for(uint v=0;v<3;v++){
            points[i][v]=plus_vec(Vec{domain[2*i],domain[2*i+1]},p[i].offset[v]);
            if(!tighten(points[i][v].x,{-half_value.upper,half_value.upper})||!tighten(points[i][v].y,{-half_value.upper,half_value.upper}))return 4;
            for(uint w=0;w<4;w++)if(p[i].wall[v]&(1u<<w)){
                Range wall=scale_int(half_value,(w==0||w==2)?-1:1);
                if(w<2){if(!tighten(points[i][v].x,wall))return 4;}else if(!tighten(points[i][v].y,wall))return 4;
            }
            if(p[i].base_edge>=0){uint w=uint(p[i].base_wall),opposite=(uint(p[i].base_edge)+2)%3;
                Range coordinate=scale_int(v==opposite?subtract(half_value,h):half_value,(w==0||w==2)?-1:1);
                if(w<2){if(!tighten(points[i][v].x,coordinate))return 4;}else if(!tighten(points[i][v].y,coordinate))return 4;
            }
        }
        for(uint pair=0;pair<3;pair++){
            uint a=ii[pair],b=jj[pair],code=contacts[pair];
            if(code>0&&code<10){uint z=code-1,va=z/3,vb=z%3;
                Range x=points[a][va].x,y=points[a][va].y;
                if(!tighten(x,points[b][vb].x)||!tighten(y,points[b][vb].y))return 5;
                points[a][va]=Vec{x,y};points[b][vb]=Vec{x,y};
            }else if(code>=10&&code<28){
                uint z=code-10,o=z<9?a:b,t=z<9?b:a,e=(z%9)/3,v=z%3,opposite=(e+2)%3;Vec n=p[o].edge[e];
                if(!plane_equation(points[o][opposite],points[t][v],n,h,uncertain))return 5;
                Range position=inner(quarter_turn(n),minus_vec(points[t][v],points[o][e]),uncertain);
                if(position.upper<0||position.lower>UNIT)return 5;
                if(distance2(points[t][v],points[o][opposite],uncertain).upper<3*UNIT/4)return 5;
                for(uint k=0;k<3;k++)if(!outside_halfspace(points[o][opposite],points[t][k],n,h,uncertain))return 5;
            }else if(code>=28){
                uint z=code-28,e=z/3,f=z%3,oa=(e+2)%3,ob=(f+2)%3;Vec na=p[a].edge[e],nb=p[b].edge[f];
                if(!plane_equation(points[a][oa],points[b][ob],na,scale_int(h,2),uncertain))return 5;
                Range position=inner(quarter_turn(na),minus_vec(points[b][f],points[a][e]),uncertain);
                if(position.upper<0||position.lower>2*UNIT)return 5;
                if(distance2(points[b][ob],points[a][oa],uncertain).upper<3*UNIT)return 5;
                for(uint k=0;k<3;k++)if(!outside_halfspace(points[a][oa],points[b][k],na,h,uncertain)||
                    !outside_halfspace(points[b][ob],points[a][k],nb,h,uncertain))return 5;
            }
        }
        for(uint i=0;i<3;i++){
            long lx=BIG,hx=-BIG,ly=BIG,hy=-BIG;
            for(uint v=0;v<3;v++){lx=min(lx,points[i][v].x.lower);hx=max(hx,points[i][v].x.upper);ly=min(ly,points[i][v].y.lower);hy=max(hy,points[i][v].y.upper);}
            if(hx-lx<h.lower||hy-ly<h.lower)return 7;
            for(uint v=0;v<3;v++){
                Range length=distance2(points[i][v],points[i][(v+1)%3],uncertain);if(length.lower>UNIT||length.upper<UNIT)return 7;
                if(!tighten(domain[2*i],subtract(points[i][v].x,p[i].offset[v].x))||!tighten(domain[2*i+1],subtract(points[i][v].y,p[i].offset[v].y)))return 7;
                for(uint w=0;w<4;w++)if(p[i].wall[v]&(1u<<w)){
                    Range coordinate=w<2?points[i][v].x:points[i][v].y;
                    if(!tighten(domain[9],scale_int(coordinate,(w==0||w==2)?-2:2)))return 7;
                }
            }
        }
        for(uint pair=0;pair<3;pair++){
            uint a=ii[pair],b=jj[pair];bool possible=false;
            for(uint side=0;side<2;side++){
                uint o=side?a:b,t=side?b:a;
                for(uint e=0;e<3;e++){
                    long minimum=BIG;
                    for(uint v=0;v<3;v++)minimum=min(minimum,subtract(inner(p[o].edge[e],minus_vec(points[t][v],points[o][(e+2)%3]),uncertain),h).upper);
                    if(minimum>=0)possible=true;
                }
            }if(!possible)return 8;
        }
    }return 0;
}
inline bool rigid_decrease(thread Range* domain,uint id,uint chart,device const uint* masks,device const uint* triples,
    device const long* roots,thread bool& uncertain){
    Piece p[3];if(initialise(domain,id,chart,masks,triples,roots,p,uncertain))return true;
    bool graph[3][3];for(uint i=0;i<3;i++)for(uint j=0;j<3;j++)graph[i][j]=i==j;
    uint m=id%50653u,codes[3]={m/1369u,(m/37u)%37u,m%37u},ii[3]={0,0,1},jj[3]={1,2,2};
    for(uint k=0;k<3;k++)if(codes[k]){graph[ii[k]][jj[k]]=true;graph[jj[k]][ii[k]]=true;}
    for(uint k=0;k<3;k++)for(uint i=0;i<3;i++)for(uint j=0;j<3;j++)graph[i][j]=graph[i][j]||(graph[i][k]&&graph[k][j]);
    for(uint start=0;start<3;start++){
        long lower[4]={BIG,BIG,BIG,BIG},upper[4]={-BIG,-BIG,-BIG,-BIG};uint flags=0;
        for(uint i=0;i<3;i++)if(graph[start][i])for(uint v=0;v<3;v++){
            Vec point=plus_vec(Vec{domain[2*i],domain[2*i+1]},p[i].offset[v]);
            for(uint w=0;w<4;w++)if(p[i].wall[v]&(1u<<w)){
                Range coordinate=w<2?point.y:point.x;lower[w]=min(lower[w],coordinate.lower);upper[w]=max(upper[w],coordinate.upper);flags|=1u<<w;
            }
        }
        bool h=(flags&3)==3,v=(flags&12)==12;
        bool forward=(!h||upper[0]<=lower[1])&&(!v||upper[3]<=lower[2]);
        bool backward=(!h||lower[0]>=upper[1])&&(!v||lower[3]>=upper[2]);
        if(!forward&&!backward)return false;
    }return true;
}
inline Range motion_gap(Vec n,Vec delta,Vec other_local,long own_spin,long other_spin,uint kind,uint feature,uint owner_vertex,uint endpoint,
    thread bool& uncertain){
    Vec tangent=quarter_turn(n);Range own=inner(tangent,delta,uncertain),other=negate(inner(tangent,other_local,uncertain));
    if(kind==0)own=value(owner_vertex==feature?-UNIT/2:UNIT/2);
    if(kind==1)own=value(0);
    if(kind==2&&!tighten(own,{-UNIT/2,UNIT/2}))uncertain=true;
    if(kind==3){if(!tighten(own,{-3*UNIT/2,3*UNIT/2}))uncertain=true;other=value(endpoint?UNIT/2:-UNIT/2);}
    return add(scale_int(own,own_spin),scale_int(other,other_spin));
}
inline bool supplied_motion(thread Range* domain,uint id,uint chart,device const uint* masks,device const uint* triples,
    device const long* roots,thread long* velocity,thread bool& uncertain){
    Piece p[3];if(initialise(domain,id,chart,masks,triples,roots,p,uncertain))return true;
    Range r=divide_small(Range{roots[0],roots[1]},6);
    for(uint i=0;i<3;i++)for(uint v=0;v<3;v++)for(uint w=0;w<4;w++)if(p[i].wall[v]&(1u<<w)){
        Vec n=outward_wall(w),tangent=quarter_turn(p[i].offset[v]);
        Range wall=add(add(scale_int(n.x,velocity[2*i]),scale_int(n.y,velocity[2*i+1])),scale_int(inner(n,tangent,uncertain),velocity[6+i]));
        if(wall.upper>=-512*UNIT)return false;
    }
    uint m=id%50653u,codes[3]={m/1369u,(m/37u)%37u,m%37u},ii[3]={0,0,1},jj[3]={1,2,2};
    for(uint pair=0;pair<3;pair++)if(codes[pair]){
        uint a=ii[pair],b=jj[pair],code=codes[pair];bool pair_valid=false;
        for(uint candidate=0;candidate<12;candidate++){
            uint o=candidate<6?a:b,t=candidate<6?b:a,feature=candidate%6;bool radial=feature>=3;uint e=feature%3;
            uint contact_vertices[2]={0,0},count=1,kind=0,own_vertex=0;
            if(code<10){uint z=code-1;own_vertex=o==a?z/3:z%3;contact_vertices[0]=o==a?z%3:z/3;
                if(radial?e!=own_vertex:(e!=own_vertex&&(e+1)%3!=own_vertex))continue;kind=radial?1:0;
            }else if(code<28){uint z=code-10,expected=z<9?a:b;if(radial||o!=expected||e!=(z%9)/3)continue;
                contact_vertices[0]=z%3;kind=2;
            }else{uint z=code-28;if(radial||o!=a||e!=z/3)continue;contact_vertices[0]=z%3;contact_vertices[1]=(z%3+1)%3;count=2;kind=3;}
            Vec n=radial?p[o].radial[e]:p[o].edge[e],dc=minus_vec(Vec{domain[2*t],domain[2*t+1]},Vec{domain[2*o],domain[2*o+1]});
            Range height=radial?scale_int(r,2):r;bool valid=true;
            for(uint v=0;v<3;v++){
                bool contact=false;for(uint j=0;j<count;j++)contact=contact||v==contact_vertices[j];
                if(!contact&&subtract(inner(n,plus_vec(dc,p[t].offset[v]),uncertain),height).lower<=0)valid=false;
            }
            Range translation=add(scale_int(n.x,velocity[2*t]-velocity[2*o]),scale_int(n.y,velocity[2*t+1]-velocity[2*o+1]));
            for(uint j=0;j<count;j++){
                uint v=contact_vertices[j];Range turn=motion_gap(n,plus_vec(dc,p[t].offset[v]),p[t].offset[v],velocity[6+o],velocity[6+t],kind,e,own_vertex,j,uncertain);
                if(add(translation,turn).lower<=0)valid=false;
            }if(valid){pair_valid=true;break;}
        }if(!pair_valid)return false;
    }return true;
}
'''

PHYSICAL_SOURCE=r'''
uint row=thread_position_in_grid.x;if(row>=amount[0])return;
Range v[10];for(uint j=0;j<10;j++)v[j]={states[23*row+3+2*j],states[23*row+4+2*j]};
bool uncertain=false;int result=contract(v,uint(states[23*row]),uint(states[23*row+1]),masks,triples,roots,control[0],uncertain);
verdict[row]=uncertain?0u:uint(result);
for(uint j=0;j<23;j++)refined[23*row+j]=long(states[23*row+j]);
if(!uncertain&&result==0)for(uint j=0;j<10;j++){refined[23*row+3+2*j]=v[j].lower;refined[23*row+4+2*j]=v[j].upper;}
'''
MOTION_SOURCE=r'''
uint row=thread_position_in_grid.x;if(row>=amount[0])return;
Range v[10];for(uint j=0;j<10;j++)v[j]={states[23*row+3+2*j],states[23*row+4+2*j]};
bool uncertain=false;uint id=uint(states[23*row]),chart=uint(states[23*row+1]);bool valid=false;
if(control[0]==0)valid=rigid_decrease(v,id,chart,masks,triples,roots,uncertain);
else{long velocity[9];for(uint j=0;j<9;j++)velocity[j]=velocities[9*row+j];valid=supplied_motion(v,id,chart,masks,triples,roots,velocity,uncertain);}
accepted[row]=(valid&&!uncertain)?1u:0u;
'''
CONSTANT_SOURCE=r'''
ulong a=radical_floor(3),b=radical_floor(6),c=radical_floor(2);
out[0]=long(a);out[1]=long(a+1);out[2]=long(b);out[3]=long(b+1);out[4]=long(c);out[5]=long(c+1);
out[6]=long(a/2+b/4);out[7]=long((a+2)/2+(b+4)/4);
'''

def load_file(path):
    # File decoding/byte copy only. No NumPy coordinate arithmetic.
    return mx.array(np.load(path,allow_pickle=False).copy(order='C'))

class Verifier:
    def __init__(self,bits):
        assert bits in [24,30]
        self.bits=bits;self.header=ARITHMETIC.replace('__BITS__',str(bits))+GEOMETRY
        self.masks=load_file(ROOT/'contact_catalog/wall_masks.npy').astype(mx.uint32)
        self.triples=load_file(ROOT/'contact_catalog/wall_triplets.npy').astype(mx.uint32)
        constants=mx.fast.metal_kernel(name=f'independent_constants_{bits}',input_names=['unused'],output_names=['out'],header=self.header,source=CONSTANT_SOURCE)
        self.roots=constants(inputs=[mx.array([0],dtype=mx.int32)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(8,)],output_dtypes=[mx.int64])[0]
        self.physical=mx.fast.metal_kernel(name=f'independent_halfspace_model_{bits}',input_names=['states','masks','triples','roots','control','amount'],
            output_names=['refined','verdict'],header=self.header,source=PHYSICAL_SOURCE)
        self.motion=mx.fast.metal_kernel(name=f'independent_motion_model_{bits}',input_names=['states','masks','triples','roots','velocities','control','amount'],
            output_names=['accepted'],header=self.header,source=MOTION_SOURCE)
        mx.eval(self.roots)
    def contract(self,states,iterations=12):
        if states.shape[0]==0:return states.astype(mx.int64),mx.zeros((0,),dtype=mx.uint32)
        outputs=[]
        for first in range(0,states.shape[0],4096):
            block=states[first:first+4096]
            r=self.physical(inputs=[block,self.masks,self.triples,self.roots,mx.array([iterations],dtype=mx.uint32),mx.array([block.shape[0]],dtype=mx.uint32)],
                grid=(block.shape[0],1,1),threadgroup=(64,1,1),output_shapes=[block.shape,(block.shape[0],)],output_dtypes=[mx.int64,mx.uint32])
            mx.eval(*r);outputs.append(r)
        return tuple(mx.concatenate([r[i] for r in outputs]) for i in range(2))
    def motion_check(self,states,velocity=None):
        if states.shape[0]==0:return mx.zeros((0,),dtype=mx.uint32)
        supplied=velocity is not None
        if velocity is None:velocity=mx.zeros((states.shape[0],9),dtype=mx.int32)
        result=[]
        for first in range(0,states.shape[0],4096):
            block=states[first:first+4096]
            a=self.motion(inputs=[block,self.masks,self.triples,self.roots,velocity[first:first+4096],mx.array([int(supplied)],dtype=mx.uint32),mx.array([block.shape[0]],dtype=mx.uint32)],
                grid=(block.shape[0],1,1),threadgroup=(64,1,1),output_shapes=[(block.shape[0],)],output_dtypes=[mx.uint32])[0]
            mx.eval(a);result.append(a)
        return mx.concatenate(result)
    def verify_transition(self,states,claimed,reasons,iterations=12):
        refined,verdict=self.contract(states,iterations)
        rejected=(reasons!=0)&(reasons!=9)
        subset=mx.all((refined[:,3::2]>=claimed[:,3::2].astype(mx.int64))&(refined[:,4::2]<=claimed[:,4::2].astype(mx.int64)),axis=1)
        accepted=(verdict!=0)|(~rejected&subset)
        return accepted,refined,verdict

def source_commitment():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
