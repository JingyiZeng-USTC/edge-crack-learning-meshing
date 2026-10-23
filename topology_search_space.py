"""Module 4: Crack-aware Topology Search Space. v3."""
from __future__ import annotations
import math
from typing import Dict, List, Set, Tuple
import numpy as np
_EPS = 1e-12
def _ek(i,j): return (min(i,j), max(i,j))
def _ori(ax,ay,bx,by,cx,cy): return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
def _pt_seg(px,py,sx,sy,tx,ty):
    dx,dy=tx-sx,ty-sy;L2=dx*dx+dy*dy
    if L2<1e-30:return math.hypot(px-sx,py-sy)
    t=max(0.0,min(1.0,((px-sx)*dx+(py-sy)*dy)/L2))
    return math.hypot(px-(sx+t*dx),py-(sy+t*dy))

class _Delaunay2D:
    def __init__(self,pts):
        self._pts=np.asarray(pts,dtype=float);self._N=len(self._pts)
        xmin,xmax=self._pts[:,0].min(),self._pts[:,0].max();ymin,ymax=self._pts[:,1].min(),self._pts[:,1].max()
        m=max(max(xmax-xmin,ymax-ymin,1.0),1.0)*10.0;mx=(xmin+xmax)/2;my=(ymin+ymax)/2
        s0=(mx-m,my-m);s1=(mx+m,my-m);s2=(mx,my+m)
        self._all=np.vstack([self._pts,[s0,s1,s2]])
        self._triangles=[]
        self._add_triangle(self._N,self._N+1,self._N+2)
        for i in range(self._N):self._insert_point(i)
    def _cc(self,i0,i1,i2):
        a,b,c=self._all[i0],self._all[i1],self._all[i2]
        d=2.0*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
        if abs(d)<1e-20:mx=(a[0]+b[0]+c[0])/3;my=(a[1]+b[1]+c[1])/3;return(mx,my,1e20)
        a2=a[0]**2+a[1]**2;b2=b[0]**2+b[1]**2;c2=c[0]**2+c[1]**2
        cx=(a2*(b[1]-c[1])+b2*(c[1]-a[1])+c2*(a[1]-b[1]))/d
        cy=(a2*(c[0]-b[0])+b2*(a[0]-c[0])+c2*(b[0]-a[0]))/d
        r2=(cx-a[0])**2+(cy-a[1])**2;return(cx,cy,r2)
    def _add_triangle(self,i0,i1,i2):
        cx,cy,r2=self._cc(i0,i1,i2);self._triangles.append((i0,i1,i2,cx,cy,r2))
    def _in_circle(self,px,py,cx,cy,r2):return(px-cx)**2+(py-cy)**2<r2-1e-12
    def _insert_point(self,idx):
        px,py=self._all[idx];bad=[ti for ti,t in enumerate(self._triangles) if self._in_circle(px,py,t[3],t[4],t[5])]
        if not bad:return
        ec={}
        for ti in bad:
            i0,i1,i2=self._triangles[ti][:3]
            for e in[(i0,i1),(i1,i2),(i2,i0)]:ek=(min(e),max(e));ec[ek]=ec.get(ek,0)+1
        cavity=[e for e,c in ec.items() if c==1]
        for ti in sorted(bad,reverse=True):self._triangles.pop(ti)
        for e0,e1 in cavity:self._add_triangle(idx,e0,e1)
    def edges(self):
        s=set()
        for i0,i1,i2,_,_,_ in self._triangles:
            if i0>=self._N or i1>=self._N or i2>=self._N:continue
            for a,b in[(i0,i1),(i1,i2),(i2,i0)]:s.add((min(a,b),max(a,b)))
        return list(s)

class TopologySearchSpace:
    def __init__(self,geometry,size_field,points,K=10,radius_factor=2.5):
        self._geo=geometry;self._sf=size_field;self._pts=np.asarray(points,dtype=float);self._N=len(self._pts)
        self._K=int(K);self._rf=float(radius_factor)
        self._h_vals=None;self._d_tip=None;self._signed_crack=None

    def _crack_visible(self,i,j):
        """True if edge does NOT cross the FINITE crack segment."""
        pa=self._pts[i];pb=self._pts[j]
        pc=np.array(self._geo.crack_start);pd=np.array(self._geo.crack_tip)
        o1=_ori(pa[0],pa[1],pb[0],pb[1],pc[0],pc[1])
        o2=_ori(pa[0],pa[1],pb[0],pb[1],pd[0],pd[1])
        o3=_ori(pc[0],pc[1],pd[0],pd[1],pa[0],pa[1])
        o4=_ori(pc[0],pc[1],pd[0],pd[1],pb[0],pb[1])
        return not((o1*o2<-_EPS)and(o3*o4<-_EPS))

    def _filter_visible(self,edges):return[(i,j)for(i,j)in edges if self._crack_visible(i,j)]

    def _precompute(self):
        self._h_vals=self._sf.evaluate_bulk(self._pts)
        self._d_tip=self._geo.distance_to_tip_bulk(self._pts)
        sx,sy=self._geo.crack_start;nx,ny=self._geo.crack_normal
        self._signed_crack=np.array([(x-sx)*nx+(y-sy)*ny for x,y in self._pts])
        W,H=self._geo.W,self._geo.H;xv,yv=self._pts[:,0],self._pts[:,1]
        bf=((np.abs(xv)<_EPS)|(np.abs(xv-W)<_EPS)|(np.abs(yv)<_EPS)|(np.abs(yv-H)<_EPS)).astype(np.float32)
        cf=(np.abs(self._signed_crack)<_EPS).astype(np.float32)
        self._node_features=np.column_stack([xv,yv,self._d_tip,self._signed_crack,self._h_vals,bf,cf])

    def _edges_knn(self):
        h_max=self._sf.h_max;h_min=self._sf.h_min
        diff=self._pts[:,None,:]-self._pts[None,:,:];dist=np.sqrt(diff[:,:,0]**2+diff[:,:,1]**2)
        edges=[]
        for i in range(self._N):
            k_local=int(round(10*h_max/max(self._h_vals[i],h_min)));k_local=max(4,min(k_local,20))
            k=min(k_local,self._N-1)
            if k<1:continue
            order=np.argpartition(dist[i],k+1)[:k+1]
            for j in order:
                if j!=i and dist[i,j]>1e-12:edges.append((i,j))
                if len([e for e in edges if e[0]==i])>=k:break
        return self._filter_visible(edges)

    def _edges_radius(self):
        cell=self._sf.h_min/math.sqrt(2)
        from adaptive_sampling import _SpatialGrid
        grid=_SpatialGrid(cell,self._geo.W,self._geo.H)
        for i,(x,y)in enumerate(self._pts):grid.insert(i,x,y)
        edges=[]
        for i in range(self._N):
            ri=self._rf*self._h_vals[i]
            for j,_,_ in grid.query_radius(self._pts[i,0],self._pts[i,1],ri):
                if j>i:
                    rj=self._rf*self._h_vals[j]
                    if math.hypot(self._pts[i,0]-self._pts[j,0],self._pts[i,1]-self._pts[j,1])<max(ri,rj):
                        if self._crack_visible(i,j):edges.append((i,j))
        return edges

    def _edges_delaunay(self):
        return self._filter_visible(_Delaunay2D(self._pts).edges())

    def _compute_edge_features(self,edges):
        M=len(edges);ii=edges[:,0];jj=edges[:,1];pi,pj=self._pts[ii],self._pts[jj]
        dx,dy=pj[:,0]-pi[:,0],pj[:,1]-pi[:,1];length=np.sqrt(dx*dx+dy*dy);angle=np.arctan2(dy,dx)
        mx,my=(pi[:,0]+pj[:,0])*0.5,(pi[:,1]+pj[:,1])*0.5;mid_pts=np.column_stack([mx,my])
        h_mid=self._sf.evaluate_bulk(mid_pts);d_tip_mid=self._geo.distance_to_tip_bulk(mid_pts)
        d_crack_mid=np.array([_pt_seg(x,y,self._geo.crack_start[0],self._geo.crack_start[1],self._geo.crack_tip[0],self._geo.crack_tip[1])for x,y in mid_pts])
        signed_i=self._signed_crack[ii];signed_j=self._signed_crack[jj]
        return np.column_stack([length,angle,d_tip_mid,d_crack_mid,h_mid,signed_i,signed_j])

    def build(self):
        self._precompute()
        e_del=self._edges_delaunay();e_rad=self._edges_radius();e_knn=self._edges_knn()
        edge_set=set()
        for e in e_del+e_rad+e_knn:edge_set.add(_ek(e[0],e[1]))
        edges=np.array(sorted(edge_set),dtype=np.int32)
        ef=self._compute_edge_features(edges)
        return {"nodes":self._pts,"edges":edges,"node_features":self._node_features,"edge_features":ef}

    def stats(self,graph):
        N=len(graph["nodes"]);M=len(graph["edges"])
        if M>0:deg=np.bincount(graph["edges"].ravel(),minlength=N);avg=deg.mean();mx=deg.max()
        else:avg=mx=0
        return f"Nodes:{N} Edges:{M} Deg:avg={avg:.1f} max={mx}"
