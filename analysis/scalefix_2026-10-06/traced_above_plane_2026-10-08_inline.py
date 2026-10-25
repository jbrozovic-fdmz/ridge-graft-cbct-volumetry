import json, numpy as np, importlib.util
from pathlib import Path
from matplotlib.path import Path as MPath
spec=importlib.util.spec_from_file_location("qc","analysis/qc_outline_overlay_2026-09-16.py"); QC=importlib.util.module_from_spec(spec); spec.loader.exec_module(QC)
prov=json.load(open("JFB_REVISION_2026-09-13/03_figures/FIG_MAIN3_outline_pairs_v5_scalefix_metalctl_2026-10-06.provenance.json"))
def shoelace(P):
    P=np.asarray(P,float); return 0.5*float(np.dot(P[:,0],np.roll(P[:,1],-1))-np.dot(P[:,1],np.roll(P[:,0],-1)))
def clip(P,a,b,c):
    out=[];P=list(map(tuple,P));n=len(P);f=lambda q:a*q[0]+b*q[1]+c
    for i in range(n):
        p,q=P[i],P[(i+1)%n];fp,fq=f(p),f(q)
        if fp<=0: out.append(p)
        if (fp<0<fq) or (fq<0<fp):
            t=fp/(fp-fq); out.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
    return out
OUT={"_what":"",
     "method":{"net":"",
               "gain_loss_split":"",
               "metal":"",
               "section_pixel_mm":None},"cases":{}}
for c in ("S1","S3","S2","S4"):
    st=int(prov["per_case"][c]["station"]); r_,fr,arch=QC.case_geometry(c)
    O=json.load(open(f"cases/{c}_outlines.json"))["stations"][str(st)]
    V=np.array(O["pre_vertices_us_mm"],float); W=np.array(O["post_vertices_us_mm"],float)
    Q=QC.us_to_local(np.array([[0,0],[1,0],[0,1]],float),arch,st,fr); y0,gu,gs=Q[0,1],Q[1,1]-Q[0,1],Q[2,1]-Q[0,1]
    def ah(P,lo=None,hi=None):
        R=P
        if hi is not None: R=clip(R,gu,gs,y0-hi)
        if lo is not None and len(R): R=clip(R,-gu,-gs,lo-y0)
        return abs(shoelace(R)) if len(R)>=3 else 0.0
    net_ab=ah(W,hi=15)-ah(V,hi=15); net_bd=ah(W,lo=15,hi=25)-ah(V,lo=15,hi=25)
    Z=np.load(f"cache_fig4/{c}.npz"); img=Z["img_post"]; u=Z["u_post"]; s=Z["s_post"]
    du=float(np.median(np.diff(u))); ds=float(np.median(np.diff(s)))
    OUT["method"]["section_pixel_mm"]=OUT["method"]["section_pixel_mm"] or {}
    OUT["method"]["section_pixel_mm"][c]=[round(abs(du),4),round(abs(ds),4)]
    g=0.01; lo=np.minimum(V.min(0),W.min(0))-0.1; hi=np.maximum(V.max(0),W.max(0))+0.1
    uu=np.arange(lo[0],hi[0],g)+g/2; ss=np.arange(lo[1],hi[1],g)+g/2
    U,S=np.meshgrid(uu,ss); P=np.column_stack([U.ravel(),S.ravel()]); Y=y0+gu*P[:,0]+gs*P[:,1]
    iV=MPath(V).contains_points(P); iW=MPath(W).contains_points(P)
    iu=np.clip(np.rint((P[:,0]-u[0])/du).astype(int),0,len(u)-1); is_=np.clip(np.rint((P[:,1]-s[0])/ds).astype(int),0,len(s)-1)
    grey=img[is_,iu]
    ab=Y<15; bd=(Y>=15)&(Y<=25); gain=iW&~iV; loss=iV&~iW
    A=lambda m: round(float(m.sum())*g*g,2)
    rec={"station":st,
         "above_plane":{"net_exact_mm2":round(net_ab,2),"gain_mm2":A(gain&ab),"loss_mm2":A(loss&ab),
                        "gain_on_metal_gt3000_mm2":A(gain&ab&(grey>3000)),"gain_on_metal_gt7000_mm2":A(gain&ab&(grey>7000)),
                        "gain_bone_only_mm2_(gain_minus_gt3000)":round(A(gain&ab)-A(gain&ab&(grey>3000)),2)},
         "in_band_15_25":{"net_exact_mm2":round(net_bd,2),"gain_mm2":A(gain&bd),"loss_mm2":A(loss&bd),
                          "gain_on_metal_gt3000_mm2":A(gain&bd&(grey>3000)),"gain_on_metal_gt7000_mm2":A(gain&bd&(grey>7000))},
         "whole_section_net_exact_mm2":round(abs(shoelace(W))-abs(shoelace(V)),2),"outline_json_gain_area_mm2":O["gain_area_mm2"],
         "gain_whole_section_on_metal_gt3000_mm2":A(gain&(grey>3000)),"healed_section_max_grey_in_gain":float(grey[gain].max())}
    OUT["cases"][c]=rec; print(c, json.dumps(rec))
p=Path("../FINAL_scalefix_metalctl_2026-10-06/TRACED_ABOVE_PLANE_exact_2026-10-08.json")
assert not p.exists(); p.write_text(json.dumps(OUT,indent=1)+"\n"); print(OUT["method"]["section_pixel_mm"])

