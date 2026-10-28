"""Constrained academic ranking, explicit cycle effects, held-group calibration."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from model import ROOT, KINDS, PROGRAMS, datasets, canonical, sigmoid, metrics
from scipy.optimize import minimize, minimize_scalar

VERSION='learned-weight-rank-v6-2026-10-23'
TRADEOFF={'SAT':(50.,200.,125.),'Bocconi':(2.,8.,5.)}


def fingerprint(row):
    return f'{float(row.score):.8f}|{float(row.GPA):.8f}|{row.Session}'


def keys(frame):return [fingerprint(r) for r in frame.itertuples()]


def w_for_k(kind,k):
    factor=120. if kind=='SAT' else 5.
    return factor/(factor+k)


def tradeoff(model):
    w=model['parameters'][0]
    return (120. if model['kind']=='SAT' else 5.)*(1-w)/w if w>0 else None


def academic(score,gpa,kind,w):
    lower,upper=KINDS[kind][2:]
    return w*(np.asarray(score)-lower)/(upper-lower)+(1-w)*np.asarray(gpa)/10


def logit_index(s):
    s=np.clip(np.asarray(s,dtype=float),1e-12,1-1e-12)
    return np.log(s)-np.log1p(-s)


def design(frame,programs,rounds,years):
    return np.array([[1.]+[float(r.Course==p) for p in programs]
                     +[float(r.Session==t) for t in rounds]
                     +[float(r.Academic_Year==y) for y in years]
                     for r in frame.itertuples()])


def fit_rank(frame,kind,grouped=False,weight_prior=False):
    if frame.empty or frame.outcome.nunique()<2:raise ValueError('Rank fit needs both outcomes.')
    programs=sorted(frame.Course.unique());rounds=sorted(frame.Session.unique());years=sorted(frame.Academic_Year.unique())
    kmin,kmax,kref=TRADEOFF[kind]
    wmin,wmax=w_for_k(kind,kmax),w_for_k(kind,kmin);wref=w_for_k(kind,kref)
    reference=logit_index(academic(frame.score.to_numpy(),frame.GPA.to_numpy(),kind,wref))
    mean,sd=float(reference.mean()),float(reference.std())
    if sd<=0:raise ValueError('No academic variation.')
    x=design(frame,programs,rounds,years)
    y=frame.outcome.to_numpy(dtype=float)
    wt=np.ones(len(frame))
    if grouped:
        counts=frame.groupby(['score','GPA','Academic_Year','Session']).score.transform('size')
        wt=1/counts.to_numpy(dtype=float)
    wt/=wt.sum()

    def objective(theta):
        w,slope=theta[:2]
        s=academic(frame.score.to_numpy(),frame.GPA.to_numpy(),kind,w)
        z=(logit_index(s)-mean)/sd
        logits=slope*z+x@theta[2:]
        loss=np.sum(wt*(np.logaddexp(0,logits)-y*logits))
        return float(loss+.02*np.sum(theta[3:]**2)+.005*(slope-1)**2
                     +(.02*((w-wref)/(wmax-wmin))**2 if weight_prior else 0.))

    wb=(0.,1.)
    bounds=[wb,(.01,20),(-30,30)]+[(-6,6)]*(x.shape[1]-1)
    starts=[.1,.25,wmin,wref,wmax,.75,.95]
    candidates=[]
    for w in starts:
        initial=np.array([w,1.,math.log((y.sum()+1)/(len(y)-y.sum()+1))]+[0.]*(x.shape[1]-1))
        result=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,
                        options={'maxiter':1500,'ftol':1e-12,'gtol':1e-7})
        if result.success and np.isfinite(result.fun):candidates.append(result)
    if not candidates:raise RuntimeError('Rank optimisation failed to converge.')
    best=min(candidates,key=lambda r:r.fun)
    return {'kind':kind,'programs':programs,'rounds':rounds,'years':years,
            'parameters':best.x.tolist(),'normalizer':{'mean':mean,'sd':sd},
            'reference_cycle':years[-1],'calibration_shift':0.,'calibration_rows':0,
            'fit_rows':len(frame),'cycles':years,'group_weighting':bool(grouped),
            'objective':float(best.fun),'optimizer_success':True,
            'weight_prior':bool(weight_prior),'weight_bounds':[0.,1.],
            'boundary_weight':bool(best.x[0] in (0.,1.))}


def rank_logit(model,score,gpa,program,session,year=None):
    if program not in model['programs'] or session not in model['rounds']:return None
    w,slope,*co=model['parameters']
    s=float(academic(score,gpa,model['kind'],w))
    if s<=0:return -math.inf
    if s>=1:return math.inf
    z=(float(logit_index(s))-model['normalizer']['mean'])/model['normalizer']['sd']
    ref=year if year in model['years'] else model['reference_cycle']
    d=[1.]+[float(program==p) for p in model['programs']]+[float(session==r) for r in model['rounds']]+[float(ref==y) for y in model['years']]
    return float(slope*z+np.dot(d,co))


def predict(model,score,gpa,program,session,calibrated=True):
    z=rank_logit(model,score,gpa,program,session)
    if z is None:return None
    if math.isinf(z):return 1. if z>0 else 0.
    shift=model['calibration_shift'] if calibrated else 0.
    return float(sigmoid(z+shift))


def predictions(model,frame,calibrated=True):
    return np.array([predict(model,float(r.score),float(r.GPA),r.Course,r.Session,calibrated)
                     for r in frame.itertuples()],dtype=float)


def calibrate(model,frame,raw_logits=None):
    """Only a one-parameter recent competition shift; no endpoint renormalising."""
    z=np.array([rank_logit(model,float(r.score),float(r.GPA),r.Course,r.Session)
                for r in frame.itertuples()],dtype=float) if raw_logits is None else np.asarray(raw_logits,dtype=float)
    mask=np.isfinite(z)
    if mask.sum()<4 or frame.outcome.to_numpy()[mask].sum() in (0,mask.sum()):
        raise ValueError('Recent calibration needs at least four rows and both classes.')
    z=z[mask];y=frame.outcome.to_numpy(dtype=float)[mask]
    def loss(delta):return float(np.mean(np.logaddexp(0,z+delta)-y*(z+delta))+.002*delta*delta)
    result=minimize_scalar(loss,bounds=(-10,10),method='bounded',options={'xatol':1e-10})
    if not result.success:raise RuntimeError('Calibration failed.')
    return float(result.x),int(mask.sum())


def fold_assignment(frame):
    groups=sorted(set(keys(frame)),key=lambda k:hashlib.sha256(('v5|'+k).encode()).hexdigest())
    mapping={k:i%5 for i,k in enumerate(groups)}
    return np.array([mapping[k] for k in keys(frame)]),mapping


def v3_fit(train,kind):
    from legacy.experimental import _hinge_design,_fit_hinge,STANDARDIZED_ALPHA
    column=KINDS[kind][1]
    points=train[[column,'GPA']].to_numpy(dtype=float)
    mean,sd=points.mean(axis=0),points.std(axis=0);knots=np.median((points-mean)/sd,axis=0)
    programs,rounds=sorted(train.Course.unique()),sorted(train.Session.unique())
    x=_hinge_design(train,column,programs,rounds,mean,sd,knots)
    theta=_fit_hinge(x,train.outcome.to_numpy(dtype=float),STANDARDIZED_ALPHA[kind])
    def logits(frame):
        a=_hinge_design(frame,column,programs,rounds,mean,sd,knots)@theta
        mask=frame.Course.isin(programs)&frame.Session.isin(rounds)
        return np.where(mask,a,np.nan)
    return logits


def bootstrap_difference(frame,ys,p,baseline):
    rng=np.random.default_rng(20261023)
    ks=keys(frame);groups=sorted(set(ks));indices={k:np.flatnonzero(np.array(ks)==k) for k in groups}
    diffs=[]
    for _ in range(1000):
        sample=np.concatenate([indices[k] for k in rng.choice(groups,len(groups),replace=True)])
        a=metrics(ys[sample],p[sample]);b=metrics(ys[sample],baseline[sample])
        diffs.append([a['brier']-b['brier'],a['log_loss']-b['log_loss']])
    return {'method':'1000 fingerprint-cluster bootstrap resamples; exploratory development only',
            'brier_difference_95pct':np.quantile(np.array(diffs)[:,0],[.025,.975]).tolist(),
            'log_loss_difference_95pct':np.quantile(np.array(diffs)[:,1],[.025,.975]).tolist()}


def resample_models(older,recent,kind,count=100):
    rng=np.random.default_rng(20261023)
    def index(frame):
        ks=np.array(keys(frame));groups=sorted(set(ks))
        return groups,{k:np.flatnonzero(ks==k) for k in groups}
    og,oi=index(older);rg,ri=index(recent);models=[]
    for i in range(count):
        oldidx=np.concatenate([oi[k] for k in rng.choice(og,len(og),replace=True)])
        newidx=np.concatenate([ri[k] for k in rng.choice(rg,len(rg),replace=True)])
        rank=fit_rank(older.iloc[oldidx].copy(),kind)
        delta,n=calibrate(rank,recent.iloc[newidx].copy())
        rank.update(calibration_shift=delta,calibration_rows=n,calibration_cycle='26-27')
        models.append(rank)
        if (i+1)%25==0:print(kind,'resample',i+1,flush=True)
    return models


def validate_kind(frame,kind):
    older=frame[frame.Academic_Year.isin(['24-25','25-26'])].copy()
    recent=frame[frame.Academic_Year=='26-27'].copy().reset_index(drop=True)
    folds,mapping=fold_assignment(recent)
    old_report=json.loads((ROOT/'research/validation-v5.json').read_text())
    previous=next(e for e in old_report['evaluations'] if e['kind']==kind)
    old_rows={r['id']:r for r in previous['oof']}
    for i,row in enumerate(recent.itertuples()):
        old=old_rows[row.id]
        if old['fingerprint']!=fingerprint(row) or old['fold']!=folds[i] or old['outcome']!=row.outcome:raise ValueError('V5 baseline rows differ')
    oof={name:np.full(len(recent),np.nan) for name in ['candidate','fixed50','recent_prior','older_only','v3_adapted','previous_v5']}
    details=[]
    for fold in range(5):
        held=recent.iloc[np.flatnonzero(folds==fold)].copy()
        cal=recent.iloc[np.flatnonzero(folds!=fold)].copy()
        hold_keys=set(keys(held))
        train=older.loc[[k not in hold_keys for k in keys(older)]].copy()
        rank=fit_rank(train,kind)
        rank['calibration_shift'],rank['calibration_rows']=calibrate(rank,cal)
        idx=np.flatnonzero(folds==fold)
        oof['candidate'][idx]=predictions(rank,held)
        oof['older_only'][idx]=predictions(rank,held,False)
        oof['fixed50'][idx]=.5
        oof['recent_prior'][idx]=(float(cal.outcome.sum())+1)/(len(cal)+2)
        oof['v3_adapted'][idx]=[old_rows[row.id]['v3_adapted'] for row in held.itertuples()]
        oof['previous_v5'][idx]=[old_rows[row.id]['candidate'] for row in held.itertuples()]
        details.append({'fold':fold,'heldout_rows':len(held),'calibration_rows':len(cal),'older_training_rows':len(train),
                        'heldout_fingerprints':sorted(hold_keys),'calibration_fingerprints':sorted(set(keys(cal))),
                        'older_training_fingerprints':sorted(set(keys(train))),'tradeoff':tradeoff(rank),
                        'weight':rank['parameters'][0],'calibration_shift':rank['calibration_shift']})
    if not all(np.isfinite(v).all() for v in oof.values()):raise ValueError('Incomplete same-row scoring')
    y=recent.outcome.to_numpy();results={name:metrics(y,p) for name,p in oof.items()}
    gate=all(results['candidate'][metric]<results[baseline][metric] for metric in ['brier','log_loss'] for baseline in ['fixed50','recent_prior'])
    rank=fit_rank(older,kind)
    stress=metrics(y,predictions(rank,recent,False))
    delta,n=calibrate(rank,recent)
    rank.update(calibration_shift=delta,calibration_rows=n,calibration_cycle='26-27')
    variants={}
    for label,kwargs in [('group_weighted',{'grouped':True}),('legacy_weight_prior',{'weight_prior':True})]:
        alternate=fit_rank(older,kind,**kwargs)
        shift,n=calibrate(alternate,recent)
        alternate.update(calibration_shift=shift,calibration_rows=n,calibration_cycle='26-27')
        variants[label]=alternate
    report={'kind':kind,'rows':len(recent),'shared_rows':len(recent),'groups':len(mapping),'metrics':results,
            'gate_passed':bool(gate),'folds':details,'forward_without_recent_calibration':stress,
            'evaluation':'retrospective held-group adaptation within26-27, NOT future-year validation',
            'oof':[{'id':row.id,'fingerprint':fingerprint(row),'outcome':int(row.outcome),'fold':int(folds[i]),
                    **{name:float(p[i]) for name,p in oof.items()}} for i,row in enumerate(recent.itertuples())]}
    print(kind,'V6 gate',gate,'tradeoff',tradeoff(rank),{name:{k:round(v,6) for k,v in stat.items() if k in ['brier','log_loss']} for name,stat in results.items()},flush=True)
    return rank,variants,resample_models(older,recent,kind),report


def export():
    bundle={'version':VERSION,'models':{},'sensitivities':{},'resamples':{},'source_sha256':{}}
    report={'version':VERSION,'label':'V6','protocol':'research/protocol-v6.md','evaluations':[],
            'scope_note':'Retrospective development on previously inspected cycles. Not future2027/28 validation.'}
    for kind,(frame,screening) in datasets().items():
        rank,variants,boot,evaluation=validate_kind(frame,kind)
        evaluation['screening']=screening
        bundle['models'][kind]=rank;bundle['sensitivities'][kind]=variants;bundle['resamples'][kind]=boot
        name=KINDS[kind][0];bundle['source_sha256'][name]=hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()
        report['evaluations'].append(evaluation)
    raw=canonical(bundle)
    (ROOT/'artifacts/model-v6.json').write_bytes(raw)
    (ROOT/'artifacts/manifest-v6.json').write_bytes(canonical({'version':VERSION,'sha256':hashlib.sha256(raw).hexdigest()}))
    (ROOT/'research/validation-v6.json').write_bytes(canonical(report))


def load():
    raw=(ROOT/'artifacts/model-v6.json').read_bytes()
    manifest=json.loads((ROOT/'artifacts/manifest-v6.json').read_text())
    if manifest.get('version')!=VERSION or hashlib.sha256(raw).hexdigest()!=manifest.get('sha256'):raise ValueError('V6 artifact mismatch')
    bundle=json.loads(raw)
    if bundle.get('version')!=VERSION or set(bundle.get('models',{}))!=set(KINDS):raise ValueError('V6 version/kinds')
    if set(bundle.get('source_sha256',{}))!={v[0] for v in KINDS.values()}:raise ValueError('V6 datasets')
    for name,sha in bundle['source_sha256'].items():
        if hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()!=sha:raise ValueError('V6 data mismatch')
    for kind in KINDS:
        if set(bundle['sensitivities'][kind])!={'group_weighted','legacy_weight_prior'} or len(bundle['resamples'][kind])!=100:raise ValueError('V6 sensitivity schema')
        models=[bundle['models'][kind]]+list(bundle['sensitivities'][kind].values())+bundle['resamples'][kind]
        for model in models:
            ps,rs,ys=model.get('programs',[]),model.get('rounds',[]),model.get('years',[])
            if model.get('kind')!=kind or not ps or ps!=sorted(set(ps)) or not set(ps)<=PROGRAMS:raise ValueError('V6 programmes')
            if not rs or not set(rs)<={'Early','Winter'} or not ys or not set(ys)<={'24-25','25-26'}:raise ValueError('V6 temporal scope')
            theta=model.get('parameters',[])
            if len(theta)!=3+len(ps)+len(rs)+len(ys) or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in theta):raise ValueError('V6 parameters')
            if not 0<=theta[0]<=1 or not .01<=theta[1]<=20:raise ValueError('V6 weight/slope')
            if model.get('boundary_weight')!=(theta[0] in (0.,1.)):raise ValueError('V6 boundary flag')
            if not all(math.isfinite(model['normalizer'][x]) for x in ['mean','sd']) or model['normalizer']['sd']<=0:raise ValueError('V6 normalizer')
            if not math.isfinite(model['calibration_shift']) or abs(model['calibration_shift'])>10 or model.get('calibration_cycle')!='26-27':raise ValueError('V6 calibration')
        if bundle['models'][kind].get('weight_prior') or bundle['models'][kind].get('group_weighting'):raise ValueError('V6 primary policy')
    return bundle


if __name__=='__main__':export()
