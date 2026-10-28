"""Offline-fitted, bounded random academic-cutoff experiment. Not admission odds."""
from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.deps'))
from legacy.data import load_profiles
from legacy.programs import PROGRAM_CONTRACTS

VERSION = 'bounded-academic-cutoff-v4-experiment-2026-10-23'
KINDS = {'SAT': ('bocconi_master_dataset.csv', 'SAT', 400., 1600.),
         'Bocconi': ('bocconitest.csv', 'Bocconi_Test', 0., 50.)}
PROGRAMS = {c['historical_proxy'] for c in PROGRAM_CONTRACTS.values()}


def datasets():
    out = {}
    for kind, (name, column, lower, upper) in KINDS.items():
        frame, screening = load_profiles(name, column, ROOT / 'legacy')
        valid = (frame.Course.isin(PROGRAMS) & frame.Session.isin(['Early', 'Winter'])
                 & frame[column].between(lower, upper))
        selected = frame.loc[valid].copy()
        selected['score'] = selected[column].astype(float)
        selected['outcome'] = (selected.Decision == 'accept').astype(int)
        selected['id'] = [f'{kind}:{int(r)}' for r in selected.source_row]
        out[kind] = (selected, dict(screening, screened_rows=len(frame),
                                     experiment_rows=len(selected), excluded_scope=int((~valid).sum())))
    return out


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


def cutoff_probability(s, cutoff, tau):
    """Stable log form of truncated logistic CDF, exact endpoints by definition."""
    s, cutoff = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(cutoff, dtype=float))
    interior = np.clip(s, 1e-14, 1 - 1e-14)
    logp = (np.log(np.expm1(interior / tau)) - np.log(np.expm1(1 / tau))
            + np.logaddexp(0, (1-cutoff)/tau) - np.logaddexp(0, (interior-cutoff)/tau))
    return np.where(s <= 0, 0., np.where(s >= 1, 1., np.clip(np.exp(logp), 0, 1)))


def weights(frame, grouped=True):
    if not grouped:
        return np.ones(len(frame))
    count = frame.groupby(['score', 'GPA', 'Academic_Year', 'Session']).score.transform('size')
    return 1 / count.to_numpy(dtype=float)


def fit(frame, kind, grouped=True):
    from scipy.optimize import minimize
    if frame.empty or frame.outcome.nunique() < 2:
        raise ValueError('Training requires both outcomes.')
    programs = sorted(frame.Course.unique())
    rounds = sorted(frame.Session.unique())
    design = np.array([[1.] + [float(p == r.Course) for p in programs]
                       + [float(t == r.Session) for t in rounds] for r in frame.itertuples()])
    lower, upper = KINDS[kind][2:]
    t = (frame.score.to_numpy()-lower)/(upper-lower)
    g = frame.GPA.to_numpy()/10
    y = frame.outcome.to_numpy()
    wt = weights(frame, grouped); wt /= wt.sum()

    def objective(params):
        w, tau = params[:2]
        s = w*t + (1-w)*g
        cutoff = design @ params[2:]
        p = np.clip(cutoff_probability(s, cutoff, tau), 1e-12, 1-1e-12)
        loss = -np.sum(wt*(y*np.log(p)+(1-y)*np.log1p(-p)))
        return float(loss + .2*np.sum(params[3:]**2) + .001*((w-.55)/.4)**2)

    bounds = [(.05, .95), (.015, .35), (.05, .95)] + [(-.35, .35)]*(design.shape[1]-1)
    candidates=[]
    for w in (.35, .55, .75):
        initial=np.array([w, .06, float(np.median(w*t+(1-w)*g))]+[0.]*(design.shape[1]-1))
        initial[2]=np.clip(initial[2],.05,.95)
        result=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,
                        options={'maxiter':1500,'ftol':1e-12,'gtol':1e-7})
        if not np.isfinite(result.fun):
            continue
        candidates.append(result)
    if not candidates:
        raise RuntimeError('No finite offline model.')
    best=min(candidates,key=lambda r:r.fun)
    # Unsuccessful numerical fits are not silently accepted as a model.
    if not best.success:
        raise RuntimeError('Offline optimizer did not converge: '+str(best.message))
    return {'kind':kind,'programs':programs,'rounds':rounds,
            'parameters':best.x.tolist(),'fit_rows':len(frame),
            'cycles':sorted(frame.Academic_Year.unique()),'group_weighting':bool(grouped),
            'objective':float(best.fun),'optimizer_success':bool(best.success)}


def predict(model, score, gpa, program, session):
    if program not in model['programs'] or session not in model['rounds']:
        return None
    w,tau,*coeff=model['parameters']
    lower,upper=KINDS[model['kind']][2:]
    academic=w*(score-lower)/(upper-lower)+(1-w)*gpa/10
    design=[1.]+[float(program==p) for p in model['programs']]+[float(session==r) for r in model['rounds']]
    c=float(np.dot(design,coeff))
    return float(cutoff_probability(academic,c,tau))


def predictions(model, frame):
    return np.array([predict(model,float(r.score),float(r.GPA),r.Course,r.Session)
                     for r in frame.itertuples()],dtype=float)


def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    mask=np.isfinite(p); y,p=y[mask],p[mask]
    if not len(y):
        return {'n':0}
    clipped=np.clip(p,1e-12,1-1e-12)
    positive=p[y==1];negative=p[y==0]
    auc=float(np.mean((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:]))) if len(positive) and len(negative) else None
    bins=[]
    for left in np.arange(0,1,.2):
        m=(p>=left)&(p<left+.2 if left<.8 else p<=1)
        if m.any():bins.append({'from':round(float(left),1),'to':round(float(left+.2),1),
                                'n':int(m.sum()),'predicted':float(p[m].mean()),'reported_admitted':float(y[m].mean())})
    return {'n':len(y),'brier':float(np.mean((y-p)**2)),
            'log_loss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
            'bias':float(np.mean(p-y)),'auc':auc,'reliability_bins':bins}


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def export():
    from legacy.experimental import _hinge_design, _fit_hinge, STANDARDIZED_ALPHA
    frames=datasets()
    report={'version':VERSION,'protocol':'research/protocol.md',
            'evaluation_status':'retrospective development, not independent future validation',
            'folds':[],'screening':{},'same_row_comparisons':True}
    bundle={'version':VERSION,'models':{},'row_weighted':{},'source_sha256':{}}
    for kind,(frame,screening) in frames.items():
        report['screening'][kind]=screening
        for fold,train_cycles,test_cycle in [('A',['24-25'],'25-26'),('B',['24-25','25-26'],'26-27')]:
            train=frame[frame.Academic_Year.isin(train_cycles)].copy()
            test=frame[frame.Academic_Year==test_cycle].copy()
            fitted=fit(train,kind)
            p=predictions(fitted,test)
            prior=float((train.outcome.sum()+1)/(len(train)+2))
            # Refit the old V3 academic FAMILY on identical train rows. No test
            # labels, no post-hoc intercept correction, no output guard censoring.
            column=KINDS[kind][1]
            academic=train[[column,'GPA']].to_numpy(dtype=float)
            mean,sd=academic.mean(axis=0),academic.std(axis=0)
            knots=np.median((academic-mean)/sd,axis=0)
            programs,rounds=sorted(train.Course.unique()),sorted(train.Session.unique())
            x=_hinge_design(train,column,programs,rounds,mean,sd,knots)
            theta=_fit_hinge(x,train.outcome.to_numpy(dtype=float),STANDARDIZED_ALPHA[kind])
            old_p=[]
            for i,r in enumerate(test.itertuples()):
                if r.Course not in programs or r.Session not in rounds:
                    old_p.append(np.nan)
                else:
                    z=_hinge_design(test.iloc[[i]],column,programs,rounds,mean,sd,knots)
                    old_p.append(float(sigmoid(z@theta)[0]))
            old_p=np.array(old_p,dtype=float)
            shared=np.isfinite(p)&np.isfinite(old_p)
            scores={
                'candidate_all_rows':metrics(test.outcome.to_numpy(),p),
                'candidate_shared':metrics(test.outcome.to_numpy()[shared],p[shared]),
                'v3_uncalibrated_shared':metrics(test.outcome.to_numpy()[shared],old_p[shared]),
                'prior_shared':metrics(test.outcome.to_numpy()[shared],np.full(int(shared.sum()),prior)),
                'fixed50_shared':metrics(test.outcome.to_numpy()[shared],np.full(int(shared.sum()),.5))}
            groups=[]
            for (course,session),group in test.groupby(['Course','Session']):
                group_p=predictions(fitted,group)
                groups.append({'program':course,'session':session,**metrics(group.outcome.to_numpy(),group_p)})
            report['folds'].append({'kind':kind,'fold':fold,'train':train_cycles,'test':test_cycle,
                                    'test_rows':len(test),'shared_rows':int(shared.sum()),
                                    'candidate_missing':int((~np.isfinite(p)).sum()),
                                    'v3_missing':int((~np.isfinite(old_p)).sum()),
                                    'parameters':fitted,'metrics':scores,'contexts':groups})
        bundle['models'][kind]=fit(frame,kind)
        bundle['row_weighted'][kind]=fit(frame,kind,False)
        name=KINDS[kind][0]
        bundle['source_sha256'][name]=hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()
    artifact=canonical(bundle)
    (ROOT/'artifacts'/'model.json').write_bytes(artifact)
    (ROOT/'artifacts'/'manifest.json').write_bytes(canonical({'version':VERSION,'sha256':hashlib.sha256(artifact).hexdigest()}))
    (ROOT/'research'/'validation.json').write_bytes(canonical(report))
    for f in report['folds']:
        print(f['kind'],f['fold'],{k:{m:round(v,5) for m,v in val.items() if m in ['brier','log_loss','bias']}
                                   for k,val in f['metrics'].items()},'shared',f['shared_rows'],'/',f['test_rows'])


def load():
    manifest=json.loads((ROOT/'artifacts'/'manifest.json').read_text())
    raw=(ROOT/'artifacts'/'model.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:
        raise ValueError('Model hash mismatch')
    bundle=json.loads(raw)
    if bundle['version']!=VERSION or manifest.get('version')!=VERSION:
        raise ValueError('Model version mismatch')
    if set(bundle['source_sha256'])!={v[0] for v in KINDS.values()}:
        raise ValueError('Missing dataset identities')
    for name,sha in bundle['source_sha256'].items():
        if hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Data hash mismatch')
    for family in ['models','row_weighted']:
        if set(bundle[family])!=set(KINDS):raise ValueError('Missing model kind')
        for kind,model in bundle[family].items():
            if model.get('kind')!=kind or model.get('programs')!=sorted(set(model.get('programs',[]))) or not model.get('programs'):
                raise ValueError('Invalid model identity')
            if not set(model['programs'])<=PROGRAMS or not set(model.get('rounds',[]))<={'Early','Winter'} or not model.get('rounds'):
                raise ValueError('Invalid model scope')
            if len(model['parameters'])!=3+len(model['programs'])+len(model['rounds']):raise ValueError('Invalid parameter count')
            w,tau,*_ = model['parameters']
            if not .05<=w<=.95 or not .015<=tau<=.35 or not all(math.isfinite(x) for x in model['parameters']):
                raise ValueError('Invalid model parameters')
            if not .05<=model['parameters'][2]<=.95 or any(abs(x)>.35 for x in model['parameters'][3:]):raise ValueError('Invalid cutoff bounds')
    return bundle


if __name__=='__main__':
    export()
