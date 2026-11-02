"""Training-only POD rank sensitivity and descriptive chronological shift checks."""
from pathlib import Path
import json
import numpy as np
from audit_data import load_data

ROOT=Path(__file__).resolve().parents[1]
x,y,t,u,*_=load_data()
ix=np.flatnonzero((x/126>=2-1e-8)&(x/126<=6+1e-8))
iy=np.flatnonzero((y/126>=-1.5-1e-8)&(y/126<=1.5+1e-8))
f=u[:,ix][:,:,iy];n=f.shape[1]*f.shape[2]
flat=f.reshape(len(f),-1);tr=np.arange(120);va=np.arange(140,180);te=np.arange(200,250)
mean=flat[tr].mean(0);a=flat-mean
ev,v=np.linalg.eigh(a[tr]@a[tr].T/n);order=np.argsort(ev)[::-1];ev=ev[order];v=v[:,order]
ranks=[1,2,4,8,16,32,64,96,119];rows=[]
for k in ranks:
 q=a[tr].T@v[:,:k]/np.sqrt(n*ev[:k])[None,:]
 assert np.allclose(q.T@q,np.eye(k),atol=1e-7)
 row={'modes':k,'retained_training_energy':float(ev[:k].sum()/np.maximum(ev,0).sum())}
 for name,ids in [('train',tr),('validation',va),('test',te)]:
  residual=a[ids]-(a[ids]@q)@q.T
  row[name+'_projection_rmse_m_s']=float(np.sqrt(np.mean(residual**2)))
 rows.append(row)
train_mean=f[tr].mean(0);test_mean=f[te].mean(0)
shift={'mean_field_change_rmse_m_s':float(np.sqrt(np.mean((test_mean-train_mean)**2))),
 'training_mean_speed_m_s':float(f[tr].mean()),'test_mean_speed_m_s':float(f[te].mean()),
 'train_temporal_rms_m_s':float(np.sqrt(np.mean((f[tr]-train_mean)**2))),
 'test_temporal_rms_m_s':float(np.sqrt(np.mean((f[te]-test_mean)**2)))}
sx=np.rint(np.linspace(0,f.shape[1]-1,4)).astype(int);sy=np.rint(np.linspace(0,f.shape[2]-1,4)).astype(int)
obs=f[:,sx][:,:,sy].reshape(len(f),16)
shift['out_of_training_sensor_range_fraction']=float(np.mean((obs[te]<obs[tr].min(0))|(obs[te]>obs[tr].max(0))))
q=a[tr].T@v[:,:16]/np.sqrt(n*ev[:16])[None,:]
mean_change=test_mean.ravel()-mean
unresolved_mean=mean_change-(mean_change@q)@q.T
fluct=f[te].reshape(len(te),-1)-test_mean.ravel()
unresolved_fluct=fluct-(fluct@q)@q.T
shift['projection_mean_shift_mse']=float(np.mean(unresolved_mean**2))
shift['projection_fluctuation_mse']=float(np.mean(unresolved_fluct**2))
shift['projection_total_mse']=float(np.mean((a[te]-(a[te]@q)@q.T)**2))
assert np.isclose(shift['projection_total_mse'],shift['projection_mean_shift_mse']+shift['projection_fluctuation_mse'])
out={'rank_sensitivity':rows,'shift':shift,'warning':'Descriptive same-record diagnostics; no independent stationarity or causal test. Ranks are not selected using test error.'}
(ROOT/'results/reviewer_diagnostics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
