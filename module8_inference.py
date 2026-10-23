"""Module 8: GNN -> Recover(simple) -> Geometry Filter -> Quality Opt. v10."""
from __future__ import annotations
import math, os
from typing import Dict, List, Set, Tuple
import numpy as np
_EPS = 1e-12

def _ori(ax,ay,bx,by,cx,cy): return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
def _ek(i,j): return (min(i,j),max(i,j))
def _pt_seg(px,py,sx,sy,tx,ty):
    dx,dy=tx-sx,ty-sy;L2=dx*dx+dy*dy
    if L2<1e-30:return math.hypot(px-sx,py-sy)
    t=max(0.0,min(1.0,((px-sx)*dx+(py-sy)*dy)/L2))
    return math.hypot(px-(sx+t*dx),py-(sy+t*dy))
def _tri_min_angle(pts,i0,i1,i2):
    a,b,c=pts[i0],pts[i1],pts[i2];v1,v2,v3=b-a,c-a,c-b
    n1=np.linalg.norm(v1);n2=np.linalg.norm(v2);n3=np.linalg.norm(v3)
    if n1<_EPS or n2<_EPS or n3<_EPS:return 0.0
    a0=math.acos(max(-1,min(1,np.dot(v1,v2)/(n1*n2))))
    a1=math.acos(max(-1,min(1,np.dot(-v1,v3)/(n1*n3))))
    return min(a0,a1,math.pi-a0-a1)
def _seg_x(pa,pb,pc,pd):
    o1=_ori(pa[0],pa[1],pb[0],pb[1],pc[0],pc[1])
    o2=_ori(pa[0],pa[1],pb[0],pb[1],pd[0],pd[1])
    o3=_ori(pc[0],pc[1],pd[0],pd[1],pa[0],pa[1])
    o4=_ori(pc[0],pc[1],pd[0],pd[1],pb[0],pb[1])
    return (o1*o2<-_EPS) and (o3*o4<-_EPS)

# -------- GNN --------
def predict_edges(model,nf,ei,ef,threshold=0.5,device="cpu"):
    import torch
    x=torch.from_numpy(nf).float().to(device);e=torch.from_numpy(ei).long()
    if e.shape[0]!=2:e=e.T
    ea=torch.from_numpy(ef).float().to(device)
    with torch.no_grad():prob=model.predict_proba(x,e.to(device),ea).cpu().numpy()
    return (prob>=threshold).astype(np.int32)

# -------- Topology Repairs: cross-crack SEGMENT only --------
def topology_repair(pred_edges,ei,points,geometry=None):
    pred=pred_edges.copy();e=ei.T.copy()
    if geometry:
        pc=np.array(geometry.crack_start);pd=np.array(geometry.crack_tip)
        for k in range(len(pred)):
            if not pred[k]:continue
            i,j=e[0,k],e[1,k]
            if _seg_x(points[i],points[j],pc,pd):pred[k]=0
    return pred

# -------- PSLG helpers --------
def _boundary_edges(points,W,H):
    x,y=points[:,0],points[:,1];segs=set()
    for cond,ax in [(np.abs(y)<_EPS,0),(np.abs(x-W)<_EPS,1),(np.abs(y-H)<_EPS,0),(np.abs(x)<_EPS,1)]:
        idx=np.where(cond)[0]
        if len(idx)>1:
            srt=idx[np.argsort(points[idx,ax])]
            for k in range(len(srt)-1):segs.add(_ek(srt[k],srt[k+1]))
    return segs
def _crack_edges(points,sx,sy,tx,ty):
    on_c=np.array([_pt_seg(x,y,sx,sy,tx,ty)<_EPS for x,y in points]);idx=np.where(on_c)[0]
    if len(idx)<2:return set()
    dx,dy=tx-sx,ty-sy;L2=dx*dx+dy*dy
    if L2<1e-20:return set()
    t=((points[idx,0]-sx)*dx+(points[idx,1]-sy)*dy)/L2;srt=idx[np.argsort(t)]
    return {_ek(srt[k],srt[k+1]) for k in range(len(srt)-1)}
def _pred_edges_set(pred_edges,ei):
    if ei.shape[0]==2:return {_ek(int(ei[0,k]),int(ei[1,k])) for k in range(len(pred_edges)) if pred_edges[k]}
    else:return {_ek(int(ei[k,0]),int(ei[k,1])) for k in range(len(pred_edges)) if pred_edges[k]}

# -------- Recover (edge-flip with triangle neighbor map, O(N) not O(N^2)) --------
def _build_neighbor_map(tris):
    """Build (K,3) neighbor array: -1 = boundary. Edge (loc) -> triangle index."""
    K=len(tris);em={}
    for ti,t in enumerate(tris):
        for a,b in[(0,1),(0,2),(1,2)]:
            ek=_ek(int(t[a]),int(t[b]))
            em.setdefault(ek,[]).append(ti)
    ngh=np.full((K,3),-1,dtype=np.int32)
    for ti,t in enumerate(tris):
        for loc,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            ek=_ek(int(t[a]),int(t[b]))
            for other in em.get(ek,[]):
                if other!=ti:ngh[ti,loc]=other;break
    return ngh

def _new_edge_crosses_constraints(oa,ob,points,constraints):
    """True if new diagonal (oa,ob) intersects any constraint edge."""
    po,pb=points[oa],points[ob]
    for(a,b) in constraints:
        if oa in(a,b) or ob in(a,b):continue
        if _seg_x(po,pb,points[a],points[b]):return True
    return False

def _update_neighbors_after_flip(ngh,tris,ti,tj,a,b,oa,ob):
    """Local neighbor update after flipping edge (a,b) between triangles ti,tj.
    The new edge is (oa,ob) replacing (a,b)."""
    # clear old neighbors for both triangles
    ngh[ti]=[-1,-1,-1];ngh[tj]=[-1,-1,-1]
    em={}
    # rebuild only for affected triangles
    for idx in[ti,tj]:
        t=tris[idx]
        for loc,(u,v) in enumerate([(0,1),(0,2),(1,2)]):
            ek=_ek(int(t[u]),int(t[v]))
            em.setdefault(ek,[]).append(idx)
    # find neighbors of ti and tj
    for idx in[ti,tj]:
        t=tris[idx]
        for loc,(u,v) in enumerate([(0,1),(0,2),(1,2)]):
            ek=_ek(int(t[u]),int(t[v]))
            for other in em.get(ek,[]):
                if other!=idx:ngh[idx,loc]=other;break
    # also update neighbors of neighboring triangles that share the new/common edges
    for other_idx in range(len(tris)):
        if other_idx in(ti,tj):continue
        for loc,(u,v) in enumerate([(0,1),(0,2),(1,2)]):
            ek=_ek(int(tris[other_idx][u]),int(tris[other_idx][v]))
            if ek==_ek(oa,ob):
                ngh[other_idx,loc]=-1  # will be found below
                # try to find which of ti/tj shares this edge
                for cand in[ti,tj]:
                    ca=_ek(int(tris[cand][0]),int(tris[cand][1]))
                    cb=_ek(int(tris[cand][0]),int(tris[cand][2]))
                    cc=_ek(int(tris[cand][1]),int(tris[cand][2]))
                    if ek in(ca,cb,cc):ngh[other_idx,loc]=cand;break

def _recover_flip(tris,points,constraints,max_iter=10):
    tris=tris.copy()
    for it in range(max_iter):
        te=set()
        for t in tris:te.add(_ek(int(t[0]),int(t[1])));te.add(_ek(int(t[0]),int(t[2])));te.add(_ek(int(t[1]),int(t[2])))
        missing=[e for e in constraints if e not in te]
        if not missing:break
        ngh=_build_neighbor_map(tris)
        recovered=False
        for(a,b) in missing:
            pa,pb=points[a],points[b]
            # keep flipping until constraint (a,b) actually exists in the mesh
            stuck=0
            while _ek(a,b) not in {_ek(int(t[0]),int(t[1])) for t in tris}|{_ek(int(t[0]),int(t[2])) for t in tris}|{_ek(int(t[1]),int(t[2])) for t in tris}:
                flipped=False
                for ti,t in enumerate(tris):
                    for loc,(u,v) in enumerate([(0,1),(0,2),(1,2)]):
                        eu,ev=int(t[u]),int(t[v])
                        if _ek(eu,ev) in constraints:continue
                        if a in(eu,ev) or b in(eu,ev):continue
                        if not _seg_x(pa,pb,points[eu],points[ev]):continue
                        other=ngh[ti,loc]
                        if other<0:continue
                        oa=next((int(vx) for vx in tris[ti] if int(vx)not in(eu,ev)),None)
                        ob=next((int(vx) for vx in tris[other] if int(vx)not in(eu,ev)),None)
                        if oa is None or ob is None:continue
                        # reject if new diagonal crosses any constraint
                        if _new_edge_crosses_constraints(oa,ob,points,constraints):continue
                        # reject if new diagonal is >2.5x longer than old
                        old_len=math.hypot(points[eu,0]-points[ev,0],points[eu,1]-points[ev,1])
                        new_len=math.hypot(points[oa,0]-points[ob,0],points[oa,1]-points[ob,1])
                        if new_len>2.5*max(old_len,1e-6):continue
                        po,pq=points[oa],points[ob];pu,pv=points[eu],points[ev]
                        if _ori(po[0],po[1],pu[0],pu[1],pq[0],pq[1])*_ori(po[0],po[1],pq[0],pq[1],pv[0],pv[1])<=-_EPS:continue
                        tris[ti]=[oa,ob,eu];tris[other]=[oa,ob,ev];flipped=True;recovered=True
                        _update_neighbors_after_flip(ngh,tris,ti,other,eu,ev,oa,ob);stuck=0;break
                    if flipped:break
                if not flipped:stuck+=1
                if stuck>3:break  # cannot recover this constraint
        if not recovered:break
    return tris

def recover(pred_edges,ei,points,geometry):
    N=len(points)
    if N<3:return np.empty((0,3),dtype=np.int32)
    W,H=geometry.W,geometry.H;sx,sy=geometry.crack_start;tx,ty=geometry.crack_tip

    # Recover only the hard geometric constraints.  GNN-predicted edges remain
    # soft connectivity suggestions; treating all of them as constraints makes
    # recovery unnecessarily expensive and can create incompatible crossings.
    constraints:Set[Tuple[int,int]]=set()
    constraints|=_boundary_edges(points,W,H)
    constraints|=_crack_edges(points,sx,sy,tx,ty)
    try:
        import triangle as tr
        result=tr.triangulate(
            {"vertices":np.asarray(points,dtype=float),
             "segments":np.asarray(sorted(constraints),dtype=np.int32)},
            "pQ",
        )
        tris=np.asarray(result.get("triangles",[]),dtype=np.int32)
        out_points=np.asarray(result.get("vertices",points),dtype=float)
        if len(out_points)!=N or not np.allclose(out_points,points,rtol=0,atol=1e-12):
            raise ValueError("Constrained triangulation unexpectedly changed the point set.")
    except ImportError:
        from ground_truth_mesh import _Delaunay
        tris=_Delaunay(points).triangles()
        if len(tris)==0:return np.empty((0,3),dtype=np.int32)
        tris=_recover_flip(tris,points,constraints,max_iter=10)

    mesh_edges={_ek(int(t[a]),int(t[b])) for t in tris
                for a,b in ((0,1),(0,2),(1,2))}
    missing=constraints-mesh_edges
    if missing:
        raise ValueError(
            f"Failed to recover {len(missing)} hard boundary/crack edges. "
            f"First edges: {sorted(missing)[:10]}"
        )
    return tris

# -------- Geometry Filter (finite crack segment only) --------
def geometry_filter(triangles,points,geometry,eps=1e-10):
    """Remove triangles whose interior is cut by the finite crack segment."""
    triangles=np.asarray(triangles,dtype=np.int32)
    points=np.asarray(points,dtype=float)
    if len(triangles)==0:return np.empty((0,3),dtype=np.int32)

    pc=np.asarray(geometry.crack_start,dtype=float)
    pd=np.asarray(geometry.crack_tip,dtype=float)
    d=pd-pc;L2=float(np.dot(d,d))
    if L2<1e-30:raise ValueError("Crack start and tip are coincident.")
    parameter_tol=max(eps/math.sqrt(L2),1e-12)
    n=np.asarray(geometry.crack_normal,dtype=float)

    def cut_by_finite_crack(ids):
        p=points[np.asarray(ids,dtype=int)]
        sd=(p-pc)@n
        if not (np.any(sd>eps) and np.any(sd<-eps)):
            return False
        # Find intersections between triangle edges and the crack supporting line.
        # Remove the triangle only if at least one intersection lies on the
        # finite crack segment.
        for a,b in ((0,1),(1,2),(2,0)):
            sa,sb=float(sd[a]),float(sd[b])
            if sa*sb>=-eps*eps:
                continue
            alpha=sa/(sa-sb)
            q=p[a]+alpha*(p[b]-p[a])
            u=float(np.dot(q-pc,d)/L2)
            # Intersections at the crack mouth/tip only touch an endpoint and
            # must not delete the adjacent triangle.  Only an intersection in
            # the open interior of the finite crack is a genuine cut.
            if parameter_tol<u<1.0-parameter_tol:
                return True
        return False

    seen=set();uniq=[]
    for t in triangles:
        i0,i1,i2=map(int,t)
        p0,p1,p2=points[[i0,i1,i2]]
        area2=_ori(p0[0],p0[1],p1[0],p1[1],p2[0],p2[1])
        if abs(area2)<=eps or cut_by_finite_crack((i0,i1,i2)):
            continue
        key=tuple(sorted((i0,i1,i2)))
        if key not in seen:
            seen.add(key);uniq.append([i0,i1,i2])
    return np.asarray(uniq,dtype=np.int32) if uniq else np.empty((0,3),dtype=np.int32)

# -------- Laplacian Smoothing (boundary+crack nodes fixed) --------
def laplacian_smooth(triangles,points,geometry,n_iters=2):
    pts=points.copy();N=len(pts);W,H=geometry.W,geometry.H
    sx,sy=geometry.crack_start;tx,ty=geometry.crack_tip
    fixed=np.zeros(N,bool)
    fixed|=(np.abs(pts[:,0])<_EPS)|(np.abs(pts[:,0]-W)<_EPS)|(np.abs(pts[:,1])<_EPS)|(np.abs(pts[:,1]-H)<_EPS)
    for i in range(N):
        if _pt_seg(pts[i,0],pts[i,1],sx,sy,tx,ty)<_EPS:fixed[i]=True
    for _ in range(n_iters):
        adj_sum=np.zeros((N,2));adj_cnt=np.zeros(N)
        for t in triangles:
            for a,b in[(0,1),(1,2),(2,0)]:
                i,j=int(t[a]),int(t[b]);adj_sum[i]+=pts[j];adj_cnt[i]+=1
                adj_sum[j]+=pts[i];adj_cnt[j]+=1
        for i in range(N):
            if fixed[i] or adj_cnt[i]==0:continue
            new_pos=adj_sum[i]/adj_cnt[i]
            # collect incident edges for length check
            neighbors=set()
            for t in triangles:
                if i in(int(t[0]),int(t[1]),int(t[2])):
                    for v in(int(t[0]),int(t[1]),int(t[2])):
                        if v!=i:neighbors.add(v)
            # length rollback: reject if any edge grows >1.5x
            max_old=max([math.hypot(pts[i,0]-pts[nb,0],pts[i,1]-pts[nb,1]) for nb in neighbors]or[0])
            old_pos=pts[i].copy();pts[i]=new_pos
            max_new=max([math.hypot(pts[i,0]-pts[nb,0],pts[i,1]-pts[nb,1]) for nb in neighbors]or[0])
            if max_new>max_old*1.5+1e-8:pts[i]=old_pos;continue
            # angle rollback
            old_ang=min([_tri_min_angle(pts,int(t[0]),int(t[1]),int(t[2])) for t in triangles
                        if i in(int(t[0]),int(t[1]),int(t[2]))]or[math.pi/3])
            old_pos2=pts[i].copy();pts[i]=old_pos  # restore for angle baseline
            pts[i]=new_pos
            new_ang=min([_tri_min_angle(pts,int(t[0]),int(t[1]),int(t[2])) for t in triangles
                        if i in(int(t[0]),int(t[1]),int(t[2]))]or[math.pi/3])
            if new_ang<old_ang-1e-8:pts[i]=old_pos
    return pts

# -------- Quality Optimizer (edge flip with length-ratio guard) --------
def quality_optimize(triangles,points,locked=None,max_iters=2):
    tris=triangles.copy()
    for _ in range(max_iters):
        changed=False;em={}
        for ti,t in enumerate(tris):
            for ek in[_ek(int(t[0]),int(t[1])),_ek(int(t[0]),int(t[2])),_ek(int(t[1]),int(t[2]))]:
                if ek not in em:em[ek]=[]
                em[ek].append(ti)
        for ek_uv,tlist in em.items():
            if locked and ek_uv in locked:continue
            if len(tlist)!=2:continue
            ta=int(tlist[0]);tb=int(tlist[1]);u=ek_uv[0];v=ek_uv[1]
            oa=next((int(vx) for vx in tris[ta] if int(vx)not in(u,v)),None)
            ob=next((int(vx) for vx in tris[tb] if int(vx)not in(u,v)),None)
            if oa is None or ob is None:continue
            cur=min(_tri_min_angle(points,u,v,oa),_tri_min_angle(points,u,v,ob))
            cand=min(_tri_min_angle(points,oa,ob,u),_tri_min_angle(points,oa,ob,v))
            if cand>cur+1e-4:
                pa,pb=points[oa],points[ob];pu,pv=points[u],points[v]
                if _ori(pa[0],pa[1],pu[0],pu[1],pb[0],pb[1])*_ori(pa[0],pa[1],pb[0],pb[1],pv[0],pv[1])<=-_EPS:continue
                # reject if new diagonal crosses any locked edge
                if locked and _new_edge_crosses_constraints(oa,ob,points,locked):continue
                # reject if new diagonal is >3x longer than old (prevents spikes)
                old_len=math.hypot(pu[0]-pv[0],pu[1]-pv[1])
                new_len=math.hypot(pa[0]-pb[0],pa[1]-pb[1])
                if new_len/max(old_len,1e-6)>3.0:continue
                tris[ta]=[oa,ob,u];tris[tb]=[oa,ob,v];changed=True
        if not changed:break
    return tris

# -------- Mesh Export --------
def _point_sets_from_crack_info(crack_info):
    if not crack_info:return {}
    return {
        "CRACK_UPPER":np.asarray(sorted(crack_info.get("crack_upper_nodes",set())),dtype=int),
        "CRACK_LOWER":np.asarray(sorted(crack_info.get("crack_lower_nodes",set())),dtype=int),
        "CRACK_TIP":np.asarray(sorted(crack_info.get("crack_tip_nodes",{crack_info.get("tip_idx")})),dtype=int),
    }

def _meshiowrite(points,tri,path,file_format=None,crack_info=None):
    import numpy as _np, os
    try:import meshio
    except ImportError as exc:
        raise ImportError("meshio is required for this export format") from exc
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    pts=_np.column_stack([_np.asarray(points,dtype=float),_np.zeros(len(points))])
    point_sets={k:v for k,v in _point_sets_from_crack_info(crack_info).items() if len(v)>0}
    mesh=meshio.Mesh(pts,[("triangle",_np.asarray(tri,dtype=int))],point_sets=point_sets)
    kwargs={"file_format":file_format} if file_format else {}
    mesh.write(path,**kwargs)

def export_obj(points,tri,path,crack_info=None):_meshiowrite(points,tri,path,crack_info=crack_info)
def export_inp(points,tri,path,crack_info=None):_meshiowrite(points,tri,path,file_format="abaqus",crack_info=crack_info)
def export_msh(points,tri,path,crack_info=None):_meshiowrite(points,tri,path,file_format="gmsh",crack_info=crack_info)

def _write_node_component(f,name,node_ids):
    ids=sorted({int(v)+1 for v in node_ids})
    if not ids:return
    f.write(f"\n! Node component: {name}\nNSEL,NONE\n")
    f.write(f"NSEL,S,NODE,,{ids[0]}\n")
    for nid in ids[1:]:f.write(f"NSEL,A,NODE,,{nid}\n")
    f.write(f"CM,{name},NODE\nALLSEL,ALL\n")

def export_cdb(points,tri,path,thickness=1.0,E=1.0,nu=0.3,crack_info=None):
    import os;os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    with open(path,"w") as f:
        f.write("/PREP7\n! APDL mesh with crack node components\n\nET,1,PLANE182\nKEYOPT,1,3,2\n")
        f.write(f"R,1,{thickness:.12g}\nMP,EX,1,{E:.12g}\nMP,PRXY,1,{nu:.12g}\n\n")
        f.write("TYPE,1\nREAL,1\nMAT,1\n\n! Nodes\n")
        for i,p in enumerate(points,start=1):f.write(f"N,{i},{p[0]:.12g},{p[1]:.12g},0\n")
        f.write("\n! Elements\n")
        for t in tri:f.write(f"E,{int(t[0])+1},{int(t[1])+1},{int(t[2])+1}\n")
        if crack_info:
            _write_node_component(f,"CRACK_UPPER",crack_info.get("crack_upper_nodes",set()))
            _write_node_component(f,"CRACK_LOWER",crack_info.get("crack_lower_nodes",set()))
            _write_node_component(f,"CRACK_TIP",crack_info.get("crack_tip_nodes",{crack_info.get("tip_idx")}))
        f.write("\nALLSEL,ALL\nFINISH\n")

class InferencePipeline:
    def __init__(self,model=None,device="cpu"):
        self.model=model
        self.device=device
        if model is not None:
            model.eval()
    def run(self,geometry,size_field,threshold=0.5,seed=None,**sk):
        from adaptive_sampling import AdaptivePoissonSampler
        sampler=AdaptivePoissonSampler(geometry,size_field);points=sampler.sample(seed=seed,**sk)
        W,H=geometry.W,geometry.H;sx,sy=geometry.crack_start;tx,ty=geometry.crack_tip
        bnd=_boundary_edges(points,W,H);crk=_crack_edges(points,sx,sy,tx,ty)
        if self.model is None:
            # The constrained triangulator below determines connectivity.
            # In the continuous-size-field path, AI controls point density,
            # so an edge classifier and its search graph are unnecessary.
            ei=np.empty((2,0),dtype=np.int64)
            pred=np.empty(0,dtype=np.int32)
        else:
            from topology_search_space import TopologySearchSpace
            tss=TopologySearchSpace(geometry,size_field,points);graph=tss.build()
            nf=graph["node_features"].astype(np.float32);ei=graph["edges"].astype(np.int64);ef=graph["edge_features"].astype(np.float32)
            ei_pyg=ei.T if ei.shape[0]!=2 else ei
            pred=predict_edges(self.model,nf,ei_pyg,ef,threshold,self.device)
            pred=topology_repair(pred,ei,points,geometry)
            cf_flags=nf[:,6]
            for k in range(len(pred)):
                i=int(ei[0,k] if ei.shape[0]==2 else ei[k,0]);j=int(ei[1,k] if ei.shape[0]==2 else ei[k,1])
                ek=_ek(i,j)
                if ek in bnd or ek in crk:pred[k]=1;continue
                if cf_flags[i]>0.5 and cf_flags[j]>0.5:
                    if math.hypot(points[i,0]-points[j,0],points[i,1]-points[j,1])<size_field.h_min*2.5:pred[k]=1
        # Only geometric boundary and crack-chain edges are hard constraints.
        # Locking every predicted edge can freeze poor GNN connectivity.
        locked=bnd|crk
        tris=recover(pred,ei,points,geometry)
        if len(tris)>0:tris=geometry_filter(tris,points,geometry)

        # Do all coordinate/topology optimization BEFORE crack-node duplication.
        for _ in range(2):
            points=laplacian_smooth(tris,points,geometry,n_iters=2)
            tris=quality_optimize(tris,points,locked,max_iters=2)

        # Module 9 is the last topology-changing step. Do not remesh, smooth,
        # or edge-flip after this call.
        from module9_crack_nodes import duplicate_crack_nodes
        tris,points,crack_info=duplicate_crack_nodes(tris,points,geometry)
        q={}
        if len(tris)>0:
            a=np.array([_tri_min_angle(points,int(t[0]),int(t[1]),int(t[2])) for t in tris])
            q={"min_angle_deg":float(np.degrees(a.min())),"max_angle_deg":float(np.degrees(a.max())),"mean_angle_deg":float(np.degrees(a.mean())),"n_triangles":len(tris)}
        return {"points":points,"triangles":tris,"pred_edges":pred,"quality":q,
                "crack_info":crack_info,"n_nodes":len(points),"n_triangles":len(tris)}
