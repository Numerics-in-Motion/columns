# -*- coding: utf-8 -*-
"""Non-ranking mechanical checks for the published column solver.

Runs separately from the large result batches so a failed control is visible
even when every first-yield condition happened to return a number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

import column_fem as CF
import section_properties as SP
from column_brace import (equilibrium, lateral_spring, strut_resistance,
                          strut_stiffness)
from inverse_design import first_yield_load
from materials import Material
from brace_study import NAMES, ROOT, column, config, solve

OUT = ROOT / "data" / "validation.json"


def eigen_load(col, E, spring_k):
    n = col.n_nodes
    node = (n-1)//2
    with lateral_spring(3*node,spring_k):
        K0,cache=CF.assemble_global(col.nodes,col.members,E,col.area,col.inertia)
    Kg=np.zeros_like(K0)
    for kk,(i,j) in enumerate(col.members):
        T,kl,Le=cache[kk]
        g=T.T @ CF.geometric_stiffness_local(-1.0,Le) @ T
        d=[3*i,3*i+1,3*i+2,3*j,3*j+1,3*j+2]
        Kg[np.ix_(d,d)]+=g
    fixed=set(CF.column_constrained_dofs(col))
    lat=[d for d in range(3*n) if d not in fixed and d%3!=1]
    vals=eigh(K0[np.ix_(lat,lat)],-Kg[np.ix_(lat,lat)],eigvals_only=True)
    return float(vals[vals>0].min())


def spring_matrix(col,E,k):
    node=(col.n_nodes-1)//2
    original=CF.assemble_global
    Ka,_=CF.assemble_global(col.nodes,col.members,E,col.area,col.inertia)
    with lateral_spring(3*node,k):
        Kb,_=CF.assemble_global(col.nodes,col.members,E,col.area,col.inertia)
    D=Kb-Ka
    nz=np.argwhere(np.abs(D)>0)
    rng=np.random.default_rng(50)
    u=rng.standard_normal(len(D))
    err=abs(0.5*u@D@u-0.5*k*u[3*node]**2)
    ok=(CF.assemble_global is original and nz.shape==(1,2) and
        list(nz[0])==[3*node,3*node] and
        abs(D[3*node,3*node]-k) <= 1e-12*k and
        err <=1e-8*max(abs(0.5*k*u[3*node]**2),1))
    return dict(ok=bool(ok),energy_error=float(err),nonzeros=nz.tolist())


def event_mechanics(name,shape,mat,k):
    c=config(20,1/1000)
    col=column(name,20,1/1000,shape)
    node=10
    pe=CF.euler_buckling_load(mat.E,col.inertia,col.height,1.0)
    with lateral_spring(3*node,k):
        r=CF.run_nonlinear_column(col,mat,600,1_200_000/pe,40,1e-8)
    if r.failure_mode!='yield':
        return dict(ok=False,mode=r.failure_mode)
    P=np.asarray(r.load_history)
    S=np.asarray(r.max_combined_stress_history)
    ids=np.flatnonzero((S[1:]>=mat.failure_stress)&(S[:-1]<mat.failure_stress))
    if len(ids)!=1:
        return dict(ok=False,reason="no unique yield bracket")
    j=int(ids[0]+1)
    py=first_yield_load(r,mat.failure_stress)
    eq=[equilibrium(r,col,mat,node,k,i,1e-8) for i in (j-1,j)]
    force=abs(k*float(r.nodal_disp_history[j][3*node]))
    util=force/strut_resistance(mat.E,mat.failure_stress)
    ok=(P[j-1]<=py<=P[j] and all(x["ok"] for x in eq) and util<0.8)
    return dict(ok=bool(ok),first_yield_n=float(py),
                bracket_n=[float(P[j-1]),float(P[j])],
                equilibrium=eq,strut_util=float(util))


def run():
    mat=Material.from_config(config()["material"])
    k=strut_stiffness(mat.E)
    rows={}
    for name in NAMES:
        straight=column(name,20,0.0,"one_wave")
        pe=np.pi**2*mat.E*straight.inertia/straight.height**2
        bare=eigen_load(straight,mat.E,0.0)
        ideal=eigen_load(straight,mat.E,1e13)
        physical=eigen_load(straight,mat.E,k)
        spring=spring_matrix(straight,mat.E,k)
        mechanics={shape:event_mechanics(name,shape,mat,k)
                   for shape in ("one_wave","two_wave")}
        plus=solve(name,n=20,amp=1/1000,brace_factor=1.0,steps=600)
        minus=solve(name,n=20,amp=-1/1000,brace_factor=1.0,steps=600)
        mirror=abs(plus["first_yield_n"]-minus["first_yield_n"])/plus["first_yield_n"]
        rows[name]=dict(bare_over_euler=bare/pe,ideal_over_4euler=ideal/(4*pe),
                        physical_over_4euler=physical/(4*pe),spring=spring,
                        mechanics=mechanics,mirror_relative=mirror)
    # The one-plane model's square face is not the most distant fibre at 45°.
    box=SP.box_properties(0.004,0.170)
    diagonal=(box["outer_dim"]/np.sqrt(2))/(box["outer_dim"]/2)
    gates={
        "bare_euler":all(abs(x["bare_over_euler"]-1)<1e-3 for x in rows.values()),
        "ideal_brace_euler":all(abs(x["ideal_over_4euler"]-1)<1e-3 for x in rows.values()),
        "physical_brace_below_ideal":all(0.9<x["physical_over_4euler"]<1.001
                                         for x in rows.values()),
        "spring_matrix_energy":all(x["spring"]["ok"] for x in rows.values()),
        "yield_equilibrium":all(y["ok"] for x in rows.values()
                                for y in x["mechanics"].values()),
        "mirror":all(x["mirror_relative"]<1e-8 for x in rows.values()),
        "diagonal_scope":abs(diagonal-np.sqrt(2))<1e-12,
    }
    doc=dict(rows=rows,box_diagonal_fibre_ratio=diagonal,gates=gates,
             all_pass=all(gates.values()))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(doc,indent=2,
                              default=lambda x: x.item() if isinstance(x,np.generic)
                              else TypeError(type(x)))+"\n",encoding="utf-8")
    for n,x in gates.items():print("PASS" if x else "FAIL",n)
    print("wrote",OUT)
    return 0 if doc["all_pass"] else 2


if __name__=="__main__":
    raise SystemExit(run())
