"""Post-pilot failure analysis. These diagnostics must not tune the existing test."""
import json
from pathlib import Path
import numpy as np
from pilot import dy
from audit_data import load_data

ROOT=Path(__file__).resolve().parents[1]


def main():
    config=json.loads((ROOT/'results/pilot_config.json').read_text())
    a=np.load(ROOT/'results/pilot_arrays.npz')
    x,y,t,u,_,_,_=load_data()
    ix=np.flatnonzero((x>=a['x'][0]-1e-8)&(x<=a['x'][-1]+1e-8))
    iy=np.flatnonzero((y>=a['y'][0]-1e-8)&(y<=a['y'][-1]+1e-8))
    field=u[:,ix][:,:,iy];nt,nx,ny=field.shape
    tr=np.array(config['train_indices']);te=np.array(config['test_indices'])
    sensors=a['sensors'];obs=sensors[:,0]*ny+sensors[:,1]
    measured=field.reshape(nt,-1)[:,obs]
    z=(measured[te]-measured[tr].mean(0))/measured[tr].std(0)
    outside=(measured[te]<measured[tr].min(0))|(measured[te]>measured[tr].max(0))
    singular=np.linalg.svd(a['basis'][obs,:8],compute_uv=False)
    truth=a['truth'];mean=a['mean']
    average_error=float(np.mean((truth-mean)**2))
    oracle_error=float(np.mean((truth-a['POD projection oracle'])**2))
    gradients={}
    for name in ['truth','Training mean','IDW 16 sensors','POD-MLP lambda=0','POD-MLP lambda=0.1','POD-MLP lambda=1']:
        gradients[name]=float(np.sqrt(np.mean(dy(a[name],4.2)**2)))
    # Correlation uses training frames only. It is descriptive, not an independence guarantee.
    center=(field[tr]-field[tr].mean(0)).reshape(len(tr),-1)
    variance=np.mean(center**2)
    acf=[float(np.mean(center[:-lag]*center[lag:])/variance) for lag in range(1,31)]
    info={
        'analysis_status':'post-pilot diagnosis; not used to retune the reported test',
        'fraction_test_sensor_values_outside_training_minmax':float(outside.mean()),
        'test_sensor_zscore_abs_median':float(np.median(abs(z))),
        'test_sensor_zscore_abs_max':float(abs(z).max()),
        'pod8_sensor_matrix_singular_values':singular.tolist(),
        'pod8_sensor_matrix_condition_number':float(singular[0]/singular[-1]),
        'training_variance_explained_by_16_modes':config['pod_training_variance_retained'],
        'test_error_reduction_fraction_of_truth_projection_vs_training_mean':1-oracle_error/average_error,
        'gradient_rms_s_inv_seed11':gradients,
        'training_field_acf_lags_1_to_30':acf,
        'rotor_advection_scale_s_at_metadata_8_m_s':126/8,
        'crop_advection_scale_s_at_metadata_8_m_s':float((a['x'][-1]-a['x'][0])/8),
        'warning':'No z-score, correlation, or oracle figure above establishes a universal reconstruction limit.'}
    oracle=a['POD projection oracle']
    decompositions=[]
    for name in ['Training mean','POD-LSQ 8 modes','POD-MLP lambda=0','POD-MLP lambda=0.1','POD-MLP lambda=1']:
        fitting=a[name]-oracle
        residual=oracle-truth
        total=float(np.mean((a[name]-truth)**2))
        fitted=float(np.mean(fitting**2));floor=float(np.mean(residual**2))
        cross=float(2*np.mean(fitting*residual))
        assert abs(total-fitted-floor)<1e-9
        gf,gr=dy(fitting,4.2),dy(residual,4.2)
        decompositions.append({'method':name,'scope':'seed 11 for MLPs',
                               'velocity_mse_total':total,'coefficient_fit_mse':fitted,
                               'pod_truncation_mse':floor,'velocity_cross_term':cross,
                               'gradient_mse_total':float(np.mean((gf+gr)**2)),
                               'gradient_coefficient_term':float(np.mean(gf**2)),
                               'gradient_unresolved_term':float(np.mean(gr**2)),
                               'gradient_cross_term':float(2*np.mean(gf*gr))})
    info['error_decomposition']=decompositions
    (ROOT/'results/pilot_diagnosis.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
