"""Frozen shared ordered calibration experiment; no runtime fit."""
from __future__ import annotations
import hashlib,json,math
import numpy as np
import model_v6 as old
import pooled_v7 as core
ROOT=old.ROOT;KINDS=old.KINDS;PROGRAMS=old.PROGRAMS;datasets=old.datasets;academic=old.academic;tradeoff=old.tradeoff
VERSION=core.VERSION


def predict(model,score,gpa,program,session,calibrated=True):
    if not calibrated:return old.predict(model,score,gpa,program,session,False)
    if model.get('calibrator') is None:return None
    return core.predict({model['kind']:model},model['calibrator'],model['kind'],score,gpa,program,session)


def export():
    audit=json.loads((ROOT/'research/pooled-audit-v7.json').read_text());previous=old.load()
    report={'version':VERSION,'label':'V7','scope_note':audit['scope'],'evaluations':[]}
    bundle={'version':VERSION,'models':{},'sensitivities':{},'resamples':{},'resample_requested':100,'resample_failures':[],
            'source_sha256':previous['source_sha256'],'order_prior':core.ORDER}
    cal=audit['calibrator']
    for kind in KINDS:
        rank=dict(audit['ranks'][kind],calibrator=cal)
        bundle['models'][kind]=rank
        # Frozen comparison policies remain explicitly labelled V6 assumptions.
        bundle['sensitivities'][kind]={}
        bundle['resamples'][kind]=[]
        evaluation=next(e for e in audit['evaluations'] if e['kind']==kind)
        evaluation=dict(evaluation,rows=evaluation['shared_rows'],groups=len(set(r['fingerprint'] for r in evaluation['oof'])),folds=audit['folds'],evaluation='retrospective nested-group development, NOT future-year validation')
        report['evaluations'].append(evaluation)
    frame=core.combined();recent=frame[frame.Academic_Year=='26-27'];keys=sorted(set(recent.key));indices={k:np.flatnonzero(recent.key.to_numpy()==k) for k in keys};rng=np.random.default_rng(202610237)
    for i in range(100):
        ranks={kind:previous['resamples'][kind][i] for kind in KINDS}
        idx=np.concatenate([indices[k] for k in rng.choice(keys,len(keys),replace=True)]);sample=recent.iloc[idx].copy()
        try:
            fitted=core.fit(ranks,sample,tuple(cal['penalties']))
        except (ValueError,RuntimeError) as exc:
            bundle['resample_failures'].append({'index':i,'reason':str(exc),'boundary_kinds':[k for k,m in ranks.items() if m['boundary_weight']]})
            continue
        for kind in KINDS:bundle['resamples'][kind].append(dict(ranks[kind],calibrator=fitted))
    raw=old.canonical(bundle)
    (ROOT/'artifacts/model-v7.json').write_bytes(raw)
    (ROOT/'artifacts/manifest-v7.json').write_bytes(old.canonical({'version':VERSION,'sha256':hashlib.sha256(raw).hexdigest()}))
    (ROOT/'research/validation-v7.json').write_bytes(old.canonical(report))
    print('V7 frozen; bootstrapidentified',len(bundle['resamples']['SAT']),'unidentified',len(bundle['resample_failures']),flush=True)


def load():
    raw=(ROOT/'artifacts/model-v7.json').read_bytes();manifest=json.loads((ROOT/'artifacts/manifest-v7.json').read_text())
    if manifest.get('version')!=VERSION or hashlib.sha256(raw).hexdigest()!=manifest.get('sha256'):raise ValueError('V7 artifact mismatch')
    bundle=json.loads(raw)
    if bundle.get('version')!=VERSION or set(bundle.get('models',{}))!=set(KINDS):raise ValueError('V7 kinds/version')
    if set(bundle.get('source_sha256',{}))!={v[0] for v in KINDS.values()}:raise ValueError('V7 sources')
    for name,sha in bundle['source_sha256'].items():
        if hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()!=sha:raise ValueError('V7 data identity')
    n=bundle.get('resample_requested');failures=bundle.get('resample_failures',[])
    if n!=100 or not isinstance(failures,list):raise ValueError('V7 sampling metadata')
    for kind in KINDS:
        samples=bundle['resamples'][kind]
        if len(samples)+len(failures)!=100:raise ValueError('V7 sample coverage')
        for model in [bundle['models'][kind]]+samples:
            ps,rs,ys=model.get('programs',[]),model.get('rounds',[]),model.get('years',[]);theta=model.get('parameters',[])
            if model.get('kind')!=kind or not ps or not set(ps)<=PROGRAMS or not set(rs)<={'Early','Winter'} or not set(ys)<={'24-25','25-26'}:raise ValueError('V7 rank scope')
            if len(theta)!=3+len(ps)+len(rs)+len(ys) or not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in theta):raise ValueError('V7 rank parameters')
            if not 0<=theta[0]<=1 or not .01<=theta[1]<=20 or model['normalizer']['sd']<=0:raise ValueError('V7 rank limits')
            c=model['calibrator'];co=c['parameters'];nk=len(c['kinds'])
            if c['kinds']!=core.KINDS or c['programs']!=core.PROGRAMS or c['rounds']!=core.ROUNDS:raise ValueError('V7 calibration scope')
            if len(co)!=2*nk+len(c['programs'])+len(c['rounds']) or not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in co):raise ValueError('V7 calibration parameters')
            if any(not .02<=b<=5 for b in co[nk:2*nk]):raise ValueError('V7 positive slopes')
            for easy,hard in core.ORDER:
                if co[2*nk+c['programs'].index(easy)]<co[2*nk+c['programs'].index(hard)]-1e-7:raise ValueError('V7 order prior')
    return bundle


if __name__=='__main__':export()
