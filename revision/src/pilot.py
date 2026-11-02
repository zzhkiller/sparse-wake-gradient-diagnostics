"""Exploratory LES pilot: same POD-MLP capacity, changed transverse-gradient loss.

Not a paper reproduction or an independent LES benchmark. The minute-long record
does not support robust temporal or cross-condition generalization claims.
"""
from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
import platform
import time
import numpy as np
from audit_data import load_data

ROOT = Path(__file__).resolve().parents[1]


def dy(a, spacing):
    return (a[...,2:]-a[...,:-2])/(2*spacing)


class MLP:
    def __init__(self, inputs, outputs, seed, hidden=32):
        rng=np.random.default_rng(seed)
        self.p=[rng.normal(0,np.sqrt(1/inputs),(inputs,hidden)),np.zeros(hidden),
                rng.normal(0,np.sqrt(1/hidden),(hidden,outputs)),np.zeros(outputs)]

    def forward(self,x):
        h=np.tanh(x@self.p[0]+self.p[1])
        return h@self.p[2]+self.p[3],h

    def loss_grad(self,x,bvalue,bgrad,qgrad,weight):
        c,h=self.forward(x)
        loss=np.mean(np.sum((c-bvalue)**2,axis=1)+weight*(np.sum((c@qgrad)*c,axis=1)-2*np.sum(c*bgrad,axis=1)))
        dc=2*(c-bvalue+weight*(c@qgrad-bgrad))/len(x)
        dh=(dc@self.p[2].T)*(1-h*h)
        return float(loss),[x.T@dh,dh.sum(0),h.T@dc,dc.sum(0)]


def fit_model(x,bvalue,bgrad,qgrad,weight,seed,score,epochs=400):
    net=MLP(x.shape[1],bvalue.shape[1],seed)
    rng=np.random.default_rng(seed+1000)
    m=[np.zeros_like(p) for p in net.p];v=copy.deepcopy(m)
    best=float('inf');best_p=None;best_epoch=0;step=0
    start=time.perf_counter()
    for epoch in range(epochs):
        indices=rng.permutation(len(x))
        for ids in np.array_split(indices,4):
            _,grad=net.loss_grad(x[ids],bvalue[ids],bgrad[ids],qgrad,weight)
            step+=1
            for j,g in enumerate(grad):
                m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
                net.p[j]-=.002*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
        if epoch%5==0:
            val=score(net)
            if val<best:
                best=val;best_p=copy.deepcopy(net.p);best_epoch=epoch
    net.p=best_p
    return net,{"best_epoch":best_epoch,"validation_score":best,"training_seconds":time.perf_counter()-start}


def metrics(pred,truth,mask,unobs,spacing):
    err=pred-truth
    ge=dy(pred,spacing)-dy(truth,spacing)
    return {"rmse_m_s":float(np.sqrt(np.mean(err**2))),
            "unobserved_rmse_m_s":float(np.sqrt(np.mean(err[:,unobs]**2))),
            "shear_region_rmse_m_s":float(np.sqrt(np.mean(err[:,mask]**2))),
            "gradient_rmse_s_inv":float(np.sqrt(np.mean(ge**2)))}


def main():
    x,y,t,u,_,_,_=load_data()
    diameter=126.
    ix=np.flatnonzero((x/diameter>=2-1e-8)&(x/diameter<=6+1e-8))
    iy=np.flatnonzero((y/diameter>=-1.5-1e-8)&(y/diameter<=1.5+1e-8))
    x,y=x[ix],y[iy]
    field=u[:,ix][:,:,iy]; del u
    nt,nx,ny=field.shape
    spacing=float(np.diff(y).mean()); p=nx*ny
    # Predeclared exploratory chronological split with two 4.8 s guard intervals.
    tr=np.arange(0,120);va=np.arange(140,180);te=np.arange(200,250)
    sx=np.rint(np.linspace(0,nx-1,4)).astype(int)
    sy=np.rint(np.linspace(0,ny-1,4)).astype(int)
    sensors=np.array([(i,j) for i in sx for j in sy])
    obs=np.array([i*ny+j for i,j in sensors])
    unobs=np.ones((nx,ny),bool);unobs[sensors[:,0],sensors[:,1]]=False
    mean=field[tr].mean(0)
    training_gradient=abs(dy(mean,spacing))
    shear=np.zeros((nx,ny),bool)
    shear[:,1:-1]=training_gradient>=np.quantile(training_gradient,.75)
    flat=field.reshape(nt,p);center=flat-mean.ravel()
    fscale=float(np.sqrt(np.mean(center[tr]**2)))
    # Method-of-snapshots POD, computed on training frames only.
    yy=center[tr]/fscale
    eig,ev=np.linalg.eigh(yy@yy.T/p)
    order=np.argsort(eig)[::-1];eig=eig[order];ev=ev[:,order]
    k=16
    assert eig[k-1]>1e-12
    basis=yy.T@ev[:,:k]/np.sqrt(eig[:k])[None,:]
    assert np.allclose(basis.T@basis/p,np.eye(k),atol=1e-9)
    bvalue=center@basis/(p*fscale)
    basis_grad=dy(basis.reshape(nx,ny,k).transpose(2,0,1),spacing).transpose(1,2,0).reshape(-1,k)
    actual_g=dy(field-mean,spacing).reshape(nt,-1)
    gscale=float(np.sqrt(np.mean(actual_g[tr]**2)))
    bg=fscale/gscale*basis_grad
    qgrad=bg.T@bg/len(bg)
    bgrad=(actual_g/gscale)@bg/len(bg)
    # A validation score containing both target metrics is shared by all weights.
    val_gnorm=float(np.mean((actual_g[va]/gscale)**2))
    val_vnorm=float(np.mean((center[va]/fscale)**2))
    def coefficient_score(c):
        lv=val_vnorm+np.mean(np.sum(c*c-2*c*bvalue[va],axis=1))
        lg=val_gnorm+np.mean(np.sum((c@qgrad)*c-2*c*bgrad[va],axis=1))
        return float(lv)
    rawobs=flat[:,obs]
    osmean=rawobs[tr].mean(0);osstd=rawobs[tr].std(0)
    xs=(rawobs-osmean)/np.maximum(osstd,1e-6)
    def decode(c):return (mean.ravel()+fscale*c@basis.T).reshape(-1,nx,ny)
    rows=[];predictions={};model_info=[]
    truth=field[te]
    coords=np.array(np.meshgrid(x,y,indexing='ij')).reshape(2,-1).T
    distances=((coords[:,None,:]-coords[obs][None,:,:])**2).sum(2)
    iw=1/np.maximum(distances,1e-12)
    iw/=iw.sum(1,keepdims=True)
    for j,index in enumerate(obs):iw[index]=0;iw[index,j]=1
    pred=(rawobs[te]@iw.T).reshape(-1,nx,ny)
    rows.append({"method":"IDW 16 sensors","seed":-1,"lambda":0,**metrics(pred,truth,shear,unobs,spacing)})
    predictions["IDW 16 sensors"]=pred
    for name,pred in [("Training mean",np.repeat(mean[None],len(te),axis=0)),
                      ("Persistence previous frame",field[te-1])]:
        predictions[name]=pred
        rows.append({"method":name,"seed":-1,"lambda":0,**metrics(pred,truth,shear,unobs,spacing)})
    # Persistence is an extra-information diagnostic, not a same-input competitor.
    bo=basis[obs,:8]
    coeff=np.linalg.lstsq(bo,((rawobs[te]-mean.ravel()[obs])/fscale).T,rcond=1e-5)[0].T
    pred=(mean.ravel()+fscale*coeff@basis[:,:8].T).reshape(-1,nx,ny)
    rows.append({"method":"POD-LSQ 8 modes","seed":-1,"lambda":0,**metrics(pred,truth,shear,unobs,spacing)})
    predictions["POD-LSQ 8 modes"]=pred
    oracle=decode(bvalue[te])
    rows.append({"method":"POD projection oracle","seed":-1,"lambda":0,**metrics(oracle,truth,shear,unobs,spacing)})
    predictions["POD projection oracle"]=oracle
    for seed in [11,22,33]:
        for weight in [0.,.1,1.]:
            score=lambda net:coefficient_score(net.forward(xs[va])[0])
            net,info=fit_model(xs[tr],bvalue[tr],bgrad[tr],qgrad,weight,seed,score)
            pred=decode(net.forward(xs[te])[0])
            checkpoints=ROOT/'results/checkpoints'
            checkpoints.mkdir(exist_ok=True)
            np.savez_compressed(checkpoints/f'seed{seed}_lambda{weight:g}.npz',
                                **{f'p{i}':p for i,p in enumerate(net.p)})
            name=f"POD-MLP lambda={weight:g}"
            row={"method":name,"seed":seed,"lambda":weight,**metrics(pred,truth,shear,unobs,spacing)}
            rows.append(row);model_info.append({**info,"seed":seed,"lambda":weight})
            if seed==11:predictions[name]=pred
            print(json.dumps(row),flush=True)
    out=ROOT/"results"
    with (out/"pilot_metrics.csv").open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    config={"data_kind":"author-provided actual LES subset; exploratory analysis only",
            "shape_time_x_y":list(field.shape),"crop_x_m":[float(x[0]),float(x[-1])],
            "crop_y_m":[float(y[0]),float(y[-1])],"diameter_m":diameter,
            "train_indices":tr.tolist(),"validation_indices":va.tolist(),"test_indices":te.tolist(),
            "train_time_range_s":t[tr[[0,-1]]].tolist(),"validation_time_range_s":t[va[[0,-1]]].tolist(),
            "test_time_range_s":t[te[[0,-1]]].tolist(),"guard_interval_frames":20,"guard_interval_s":4.8,
            "sensor_coordinates_m":[[float(x[i]),float(y[j])] for i,j in sensors],
            "model":"16 sensor inputs -> 32 tanh hidden units -> 16 POD coefficients; fixed POD decoder",
            "parameters":16*32+32+32*16+16,"pod_modes":k,
            "fixed_pod_decoder_entries":int(p*k),"fixed_mean_field_entries":int(p),
            "decoder_and_mean_float64_bytes":int((p*k+p)*8),
            "pod_training_variance_retained":float(eig[:k].sum()/np.maximum(eig,0).sum()),
            "value_fluctuation_scale_m_s":fscale,"gradient_fluctuation_scale_s_inv":gscale,
            "weights":[0,.1,1],"seeds":[11,22,33],"epochs":400,"learning_rate":.002,
            "checkpoint_rule":"minimum validation normalized velocity MSE only, checked every 5 epochs",
            "shear_mask":"top quartile of absolute gradient of training mean; same mask for every method",
            "fixed_layout_only":True,"grid_coarsening":False,
            "oracle_warning":"POD projection oracle uses test reference fields; not deployable",
            "persistence_warning":"uses previous full field as extra information; not a fair same-input baseline",
            "generalization_warning":"60 s total and one condition; guard interval does not establish statistical independence",
            "python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform(),
            "models":model_info}
    (out/"pilot_config.json").write_text(json.dumps(config,indent=2))
    summary=[]
    for name in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==name]
        summary.append({"method":name,"runs":len(rr),**{key:{"mean":float(np.mean([a[key] for a in rr])),
                       "std":float(np.std([a[key] for a in rr],ddof=1)) if len(rr)>1 else None}
                       for key in ['rmse_m_s','unobserved_rmse_m_s','shear_region_rmse_m_s','gradient_rmse_s_inv']}})
    (out/"pilot_summary.json").write_text(json.dumps(summary,indent=2))
    arrays={"x":x,"y":y,"t_test":t[te],"truth":truth,"mean":mean,"shear_mask":shear,
            "sensors":sensors,"basis":basis,"training_eigenvalues":eig}
    for name,pred in predictions.items():arrays[name]=pred
    np.savez_compressed(out/"pilot_arrays.npz",**arrays)
    from plotting import heatmaps,lines
    heatmaps(out/"pilot_diagnostics",[mean,field[tr].std(0),np.gradient(mean,spacing,axis=1)],x/diameter,y/diameter,
             ["Training mean","Training temporal fluctuations","Training mean shear"],['m/s','m/s','1/s'],
             "Actual LES pilot: fixed crop and 16 virtual sensors",sensors=sensors)
    section=int(np.argmin(abs(x/diameter-4)))
    series=[("LES test mean",y/diameter,truth.mean(0)[section])]
    for name in ["IDW 16 sensors","Training mean","POD-MLP lambda=0","POD-MLP lambda=0.1","POD-MLP lambda=1"]:
        series.append((name,y/diameter,predictions[name].mean(0)[section]))
    lines(out/"pilot_profiles",series,f"Actual LES pilot: mean profile at x/D={x[section]/diameter:.2f}",
          "y/D","u (m/s)","Seed 11 shown; test interval only 12 seconds; no cross-condition claim")
    series=[]
    for name in ["IDW 16 sensors","Training mean","POD-MLP lambda=0","POD-MLP lambda=0.1","POD-MLP lambda=1"]:
        per=np.sqrt(np.mean((predictions[name]-truth)**2,axis=(1,2)))
        series.append((name,t[te],per))
    lines(out/"pilot_error_time",series,"Actual LES pilot: framewise field error","Time (s)","RMSE (m/s)",
          "Chronological held-out segment; adjacent frames are strongly correlated")
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
