"""Gmsh native Distance/Threshold local refinement versus learned sizing.

Uses the existing project's learned checkpoint and CST FEM implementation.
No training is performed. Outputs raw errors, including non-monotonic curves.
"""
from pathlib import Path
import json
import math
import sys
import time
import hashlib
import numpy as np
import common_pipeline as core


class LocalRefinement:
    def __init__(self, args, start, tip, scale=1.):
        self.a, self.start, self.tip, self.scale = args, start, tip, scale

    def evaluate_bulk(self, points):
        a = self.a
        v = self.tip-self.start
        u = np.clip((points-self.start) @ v / (v@v), 0., 1.)
        dc = np.linalg.norm(points-(self.start+u[:, None]*v), axis=1)
        dt = np.linalg.norm(points-self.tip, axis=1)
        ht = a.h_min+(a.h_max-a.h_min)*np.clip((dt-a.tip_inner)/(a.tip_outer-a.tip_inner), 0., 1.)
        hc = a.crack_size+(a.h_max-a.crack_size)*np.clip((dc-a.crack_inner)/(a.crack_outer-a.crack_inner), 0., 1.)
        return self.scale*np.minimum(ht, hc)

    def evaluate(self, x, y):
        return float(self.evaluate_bulk(np.array([[x, y]]))[0])

    def install_gmsh(self, line, tip):
        f = core.gmsh.model.mesh.field
        fields=[]
        for kind, entity, lo, hi, size in [
            ('PointsList', tip, self.a.tip_inner, self.a.tip_outer, self.a.h_min),
            ('CurvesList', line, self.a.crack_inner, self.a.crack_outer, self.a.crack_size),
        ]:
            dist=f.add('Distance'); f.setNumbers(dist,kind,[entity])
            if kind=='CurvesList': f.setNumber(dist,'Sampling',self.a.distance_sampling)
            threshold=f.add('Threshold')
            for key,val in {'InField':dist,'SizeMin':size*self.scale,
                            'SizeMax':self.a.h_max*self.scale,'DistMin':lo,'DistMax':hi,'Sigmoid':0}.items():
                f.setNumber(threshold,key,val)
            fields.append(threshold)
        minimum=f.add('Min'); f.setNumbers(minimum,'FieldsList',fields)
        f.setAsBackgroundMesh(minimum)


def crossing(rows, target):
    """No envelope, extrapolation, or fabricated crossing for censored curves."""
    rows=sorted(rows,key=lambda r:r['nodes'])
    if rows[0]['strain_energy_error_percent']<=target:
        return {'status':'already_below_at_coarsest','nodes':None,
                'upper_bound_nodes':rows[0]['nodes']}
    for left,right in zip(rows,rows[1:]):
        y0,y1=left['strain_energy_error_percent'],right['strain_energy_error_percent']
        if y0>target>=y1:
            n=left['nodes']+(right['nodes']-left['nodes'])*(y0-target)/(y0-y1)
            return {'status':'bracketed','nodes':n,'bracket_nodes':[left['nodes'],right['nodes']]}
    return {'status':'not_reached','nodes':None}


def main():
    p=core.build_parser()
    p.description=__doc__
    p.set_defaults(output=Path(__file__).resolve().parents[1] / 'artifacts/gmsh_local_vs_learned',reference_scale=.40)
    p.add_argument('--tip-inner',type=float,default=.008)
    p.add_argument('--tip-outer',type=float,default=.08)
    p.add_argument('--crack-inner',type=float,default=.004)
    p.add_argument('--crack-outer',type=float,default=.05)
    p.add_argument('--crack-size',type=float,default=.008)
    p.add_argument('--distance-sampling',type=int,default=1000)
    p.add_argument('--reference-refine',type=float,default=.7)
    p.add_argument('--reference-tolerance-percent',type=float,default=.10)
    a=p.parse_args()
    if not (0<a.h_min<=a.crack_size<=a.h_max): p.error('Require 0 < h-min <= crack-size <= h-max')
    if not (0<=a.tip_inner<a.tip_outer and 0<=a.crack_inner<a.crack_outer): p.error('Require 0 <= inner < outer')
    if not (len(a.scales)>=2 and 0<a.reference_scale<min(a.scales) and 0<a.reference_refine<1): p.error('Need >=2 scales, reference-scale < all scales and 0 < reference-refine < 1')
    if a.distance_sampling<2 or a.target_error_percent<=0: p.error('Invalid sampling or error target')
    if not a.checkpoint.is_file(): raise FileNotFoundError(a.checkpoint)
    sys.path.insert(0,str(a.project_root.resolve()))
    from crack_graph_geometry import CrackGraphGeometry
    from edge_crack_geometry import generate_crack
    from module30_learned_size_field.single_crack_local_field import SingleCrackLocalSizeField
    s,t=generate_crack('left',a.origin,a.crack_length,math.radians(a.angle_deg),a.width,a.height)
    s,t=np.array(s),np.array(t)
    if not (0<t[0]<a.width and 0<t[1]<a.height): p.error('Tip outside plate')
    learned=SingleCrackLocalSizeField(CrackGraphGeometry(a.width,a.height,[np.vstack([s,t])]),
        checkpoint=str(a.checkpoint),h_min=a.h_min,h_max=a.h_max,grid_nx=a.grid_nx,
        grid_ny=a.grid_ny,smooth_sigma=a.smooth_sigma,complexity_ratio=1.,device=a.device)
    local=LocalRefinement(a,s,t)
    output=a.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists(): raise FileExistsError(f'Use a new --output directory: {output}')
    mesh_dir=output/'meshes'; mesh_dir.mkdir(exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(gmsh_version=core.gmsh.__version__,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  baseline='native Gmsh Distance + Threshold + Min',seed=1,
                  ignored_legacy_options=['traditional-mode','L-tip','L-crack','L-entry'])
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    def evaluate(field,name):
        begin=time.perf_counter()
        points,triangles,q=core.generate_gmsh_mesh(field,width=a.width,height=a.height,start=s,tip=t,
            algorithm=a.gmsh_algorithm,optimize=not a.no_optimize,model_name=name)
        mesh_seconds=time.perf_counter()-begin
        if abs(q['domain_area']-a.width*a.height)>1e-8*a.width*a.height: raise RuntimeError('Area validation failed')
        if q['minimum_angle_deg']<=0: raise RuntimeError('Degenerate mesh')
        begin=time.perf_counter()
        energy,residual=core.solve_strain_energy(points,triangles,young_modulus=a.E,poisson_ratio=a.nu,
            thickness=a.thickness,line_load=a.top_line_load)
        if not np.isfinite(energy) or energy<=0 or not np.isfinite(residual) or residual>1e-7:
            raise RuntimeError(f'FEM validation failed: energy={energy}, residual={residual}')
        if a.save_meshes: np.savez_compressed(mesh_dir/f'{name}.npz',nodes=points,triangles=triangles,crack_start=s,crack_tip=t)
        print(f'{name}: N={len(points)}, U={energy:.8g} J',flush=True)
        return dict(nodes=len(points),geometric_nodes=len(points)-q['duplicated_crack_nodes'],triangles=len(triangles),
                    strain_energy_j=energy,relative_residual=residual,mesh_seconds=mesh_seconds,
                    fem_seconds=time.perf_counter()-begin,**q)
    core.gmsh.initialize()
    core.gmsh.option.setNumber('General.Terminal',0)
    core.gmsh.option.setNumber('General.NumThreads',1)
    core.gmsh.option.setNumber('Mesh.RandomSeed',1)
    try:
        refs=[]
        for i,scale in enumerate([a.reference_scale,a.reference_scale*a.reference_refine]):
            refs.append(evaluate(core.MinimumField(local,learned,scale),f'reference_{i}'))
        ref=refs[-1]['strain_energy_j']
        delta=100*abs(ref-refs[0]['strain_energy_j'])/abs(ref)
        for method in ['traditional','learned']:
            for level,scale in enumerate(sorted(set(a.scales),reverse=True),1):
                field=LocalRefinement(a,s,t,scale) if method=='traditional' else core.ScaledField(learned,scale)
                r=evaluate(field,f'{method}_{level}')
                r.update(method=method,level=level,size_scale=scale,reference_strain_energy_j=ref,
                         strain_energy_error_percent=100*abs(r['strain_energy_j']-ref)/abs(ref))
                rows.append(r); core._write_csv(output/'results.csv',rows)
    finally: core.gmsh.finalize()
    crossings={m:crossing([r for r in rows if r['method']==m],a.target_error_percent) for m in ['traditional','learned']}
    valid=delta<=a.reference_tolerance_percent
    nt,nl=crossings['traditional']['nodes'],crossings['learned']['nodes']
    saving=100*(nt-nl)/nt if valid and nt is not None and nl is not None else None
    summary={'reference_meshes':refs,'reference_change_percent':delta,'reference_check_passed':valid,
        'reference_check_note':'Two-level energy stability check, not a rigorous reference error bound.',
        'target_crossings':crossings,'node_saving_percent':saving,'target_error_percent':a.target_error_percent,
        'node_definition':'Mechanical nodes after duplicating crack faces; geometric_nodes also in CSV.',
        'claim_scope':'Comparison with the specified fixed-parameter local Gmsh sizing, not all Gmsh strategies.',
        'gnn_used':False,'fcnn_used':False}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    core.plot_curves(output,rows,a.target_error_percent,nt if valid else None,nl if valid else None,'Gmsh local refinement')
    core.validate_figure(output)
    if not valid: print('WARNING: Reference stability check failed. Refine reference before interpreting target savings.')
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print(f'Saved: {output}')

if __name__=='__main__': main()
