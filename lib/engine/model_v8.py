"""Fixed academic exchange, bounded linear historical-outcome model.

Offline fitting is explicit. Serving loads a hash-checked artifact only.
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from model import ROOT, KINDS, PROGRAMS, datasets as original_datasets, canonical, metrics
from scipy.optimize import minimize, minimize_scalar

VERSION='fixed-exchange-linear-historical-v8-2026-10-23'
PROGRAM_ORDER=[('BEMACC','BIEM'),('BIEM','BIEF'),('BIEF','BEMACS'),('BIEF','BAI')]
PS=sorted(PROGRAMS)
RS=['Early','Winter']
CENTRES={'SAT':1500.,'Bocconi':35.}
SCALES={'SAT':200.,'Bocconi':10.}
BT_K=[2.5,5.,7.5,10.,12.5,15.,20.]
SAT_K=[100.,150.,200.,250.,300.,400.,600.]
LAMBDA=.001


def datasets():
    overlay=json.loads((ROOT/'research/data-overlay-v8.json').read_text())
    corrections={r['id']:r for r in overlay['label_corrections']}
    quarantine=set(overlay['quarantine_unverified_rescaling'])
    output={}
    for kind,(frame,screening) in original_datasets().items():
        frame=frame.copy();before=len(frame);corrected=0
        for identifier,change in corrections.items():
            mask=frame.id==identifier
            if mask.any():
                if mask.sum()!=1 or frame.loc[mask,'Decision'].iloc[0]!=change['original']:raise ValueError('Overlay label identity')
                row=frame.loc[mask].iloc[0]
                if (row.Course!=change['program'] or row.score!=change['score'] or row.GPA!=change['gpa'] or row.Academic_Year!=change['cycle'] or row.Session!=change['session']):raise ValueError('Overlay full profile identity')
                frame.loc[mask,'Decision']=change['corrected'];frame.loc[mask,'outcome']=int(change['corrected']=='accept');corrected+=1
        removed=frame[frame.id.isin(quarantine)]
        frame=frame.loc[~frame.id.isin(quarantine)].copy()
        output[kind]=(frame,dict(screening,before_quality=before,quality_quarantined=len(removed),label_corrected=corrected,
                                final_fit_rows=len(frame),unverified_rescaling_ids=removed.id.tolist()))
    return output


def keys(frame):
    return [f'{float(r.score):.8f}|{float(r.GPA):.8f}|{r.Session}' for r in frame.itertuples()]


def folds(frame,n=5,salt='v8-outer'):
    ks=keys(frame)
    unique=sorted(set(ks),key=lambda k:hashlib.sha256((salt+'|'+k).encode()).hexdigest())
    lookup={k:i%n for i,k in enumerate(unique)}
    return np.array([lookup[k] for k in ks])


def design(frame,kind,k):
    a=((frame.score.to_numpy(float)-CENTRES[kind])+k*(frame.GPA.to_numpy(float)-9.))/SCALES[kind]
    return np.column_stack([np.ones(len(frame)),a]+[(frame.Course==p).to_numpy(float) for p in PS]+[(frame.Session==r).to_numpy(float) for r in RS])


def fit(frame,kind,k,grouped=False,penalty=LAMBDA):
    if frame.empty or frame.outcome.nunique()<2:raise ValueError('Both reported outcomes required')
    x=design(frame,kind,k);y=frame.outcome.to_numpy(float)
    wt=np.ones(len(frame))
    if grouped:
        sizes=frame.groupby(['score','GPA','Academic_Year','Session']).score.transform('size')
        wt=1./sizes.to_numpy(float)
    wt/=wt.sum()
    epsilon=1e-9
    def objective(t):
        p=np.clip(x@t,epsilon,1-epsilon)
        return float(-np.sum(wt*(y*np.log(p)+(1-y)*np.log1p(-p)))+penalty*np.sum(t[2:]**2))
    def jac(t):
        p=np.clip(x@t,epsilon,1-epsilon)
        g=x.T@(wt*(p-y)/(p*(1-p)))
        g[2:]+=2*penalty*t[2:]
        return g
    a=[]
    for easy,hard in PROGRAM_ORDER:
        v=np.zeros(x.shape[1]);v[2+PS.index(easy)]=1;v[2+PS.index(hard)]=-1;a.append(v)
    a=np.array(a)
    maximum=[];minimum=[]
    for pr in PS:
        for se in RS:
            indicators=[float(pr==p) for p in PS]+[float(se==r) for r in RS]
            maximum.append([1.,(KINDS[kind][3]-CENTRES[kind]+k)/SCALES[kind]]+indicators)
            minimum.append([1.,((1040. if kind=='SAT' else 17.)-CENTRES[kind]-3*k)/SCALES[kind]]+indicators)
    maximum=np.array(maximum);minimum=np.array(minimum)
    bounds=[(-5,5),(0,2)]+[(-2,2)]*(x.shape[1]-2)
    candidates=[]
    low_test=1040. if kind=='SAT' else 17.
    anchored_slope=SCALES[kind]/(KINDS[kind][3]-low_test+4*k)
    anchored_intercept=anchored_slope*(CENTRES[kind]-low_test+3*k)/SCALES[kind]
    for intercept,slope in [(anchored_intercept,anchored_slope),(float(wt@y),.1),(float(wt@y),.3)]:
        start=np.zeros(x.shape[1]);start[0]=intercept;start[1]=slope
        result=minimize(objective,start,jac=jac,method='SLSQP',bounds=bounds,
                        constraints=[{'type':'ineq','fun':lambda t:a@t,'jac':lambda t:a},
                                     {'type':'ineq','fun':lambda t:x@t-epsilon,'jac':lambda t:x},
                                     {'type':'ineq','fun':lambda t:1-epsilon-x@t,'jac':lambda t:-x},
                                     {'type':'ineq','fun':lambda t:maximum@t-1.,'jac':lambda t:maximum},
                                     {'type':'ineq','fun':lambda t:-minimum@t,'jac':lambda t:-minimum}],
                        options={'maxiter':1500,'ftol':1e-10})
        if (result.success and np.isfinite(result.fun) and np.min(a@result.x)>=-1e-7
            and np.min(x@result.x)>=-1e-7 and np.max(x@result.x)<=1+1e-7
            and np.min(maximum@result.x)>=1-1e-7 and np.max(minimum@result.x)<=1e-7):candidates.append(result)
    if not candidates:raise RuntimeError('Linear fit did not converge')
    best=min(candidates,key=lambda r:r.fun)
    return {'kind':kind,'k':k,'centre':CENTRES[kind],'scale':SCALES[kind],
            'programs':PS,'rounds':RS,'parameters':best.x.tolist(),
            'fit_rows':len(frame),'cycles':sorted(frame.Academic_Year.unique()),
            'group_weighting':grouped,'penalty':penalty,'objective':float(best.fun),
            'optimizer_success':True,'order_prior':PROGRAM_ORDER,
            'objective_kind':'identity_link_bernoulli','conditional_rank_endpoints':True}


def raw_probability(model,score,gpa,program,session):
    if program not in model['programs'] or session not in model['rounds']:raise ValueError('Unsupported programme/round')
    t=model['parameters']
    a=(np.asarray(score,dtype=float)-model['centre']+model['k']*(np.asarray(gpa,dtype=float)-9.))/model['scale']
    return t[0]+t[1]*a+t[2+model['programs'].index(program)]+t[2+len(model['programs'])+model['rounds'].index(session)]


def predict(model,score,gpa,program,session):
    value=np.clip(raw_probability(model,score,gpa,program,session),0,1)
    return float(value) if np.ndim(value)==0 else value


def predictions(model,frame):
    return np.array([predict(model,float(r.score),float(r.GPA),r.Course,r.Session) for r in frame.itertuples()])


def select_regularisation(frame,kind):
    assigned=folds(frame,3,'v8-regularisation-inner');scores=[]
    for k in (SAT_K if kind=='SAT' else BT_K):
        p=np.full(len(frame),np.nan)
        prior=np.full(len(frame),np.nan)
        for i in range(3):
            train=frame.iloc[np.flatnonzero(assigned!=i)];test=frame.iloc[np.flatnonzero(assigned==i)]
            assert not set(keys(train))&set(keys(test))
            p[assigned==i]=predictions(fit(train,kind,k),test)
            prior[assigned==i]=float(train.outcome.mean())
        y=frame.outcome.to_numpy(float)
        def loss(gamma):
            q=prior+gamma*(p-prior)
            if np.any((q<=0)&(y==1)) or np.any((q>=1)&(y==0)):return math.inf
            return float(-np.mean(np.log(np.where(y==1,q,1-q))))
        result=minimize_scalar(loss,bounds=(0.,1.),method='bounded',options={'xatol':1e-10})
        if not result.success:raise RuntimeError('Shrinkage optimisation failed')
        gamma=min([0.,float(result.x),1.],key=loss)
        scores.append({'k':k,'gamma':gamma,'log_loss':loss(gamma),'unshrunk_metrics':metrics(y,p)})
    # Primary is the requested straight linear score, without global shrinkage.
    # Learned gamma remains an explicitly rejected overconfidence sensitivity.
    criterion='log_loss' if kind=='SAT' else 'brier'
    best=min(s['unshrunk_metrics'][criterion] for s in scores)
    selected=min((s for s in scores if s['unshrunk_metrics'][criterion]<=best+1e-6),key=lambda s:s['k'])
    return selected,scores


def fit_pipeline(frame,kind):
    selected,selection=select_regularisation(frame,kind)
    m=fit(frame,kind,selected['k']);m['gamma']=selected['gamma'];m['prevalence']=float(frame.outcome.mean())
    m['tradeoff_selection']=selection
    return m


def validate(frame,kind):
    assigned=folds(frame);out={n:np.full(len(frame),np.nan) for n in ['candidate','fixed50','recent_prior']}
    details=[]
    for i in range(5):
        train=frame.iloc[np.flatnonzero(assigned!=i)];test=frame.iloc[np.flatnonzero(assigned==i)]
        m=fit_pipeline(train,kind);k=m['k'];selection=m['tradeoff_selection']
        assert not set(keys(train))&set(keys(test))
        out['candidate'][assigned==i]=predictions(m,test);out['fixed50'][assigned==i]=.5;out['recent_prior'][assigned==i]=train.outcome.mean()
        details.append({'fold':i,'train_rows':len(train),'test_rows':len(test),'k':k,'gamma':m['gamma'],'selection':selection,
                        'train_keys':sorted(set(keys(train))),'held_keys':sorted(set(keys(test)))})
    oof=[{'id':r.id,'fingerprint':key,'cycle':r.Academic_Year,'program':r.Course,'round':r.Session,
          'score':float(r.score),'gpa':float(r.GPA),'outcome':int(r.outcome),'fold':int(assigned[i]),
          **{n:float(p[i]) for n,p in out.items()}} for i,(r,key) in enumerate(zip(frame.itertuples(),keys(frame)))]
    by_year={year:metrics(frame.loc[frame.Academic_Year==year,'outcome'],out['candidate'][frame.Academic_Year.to_numpy()==year]) for year in sorted(frame.Academic_Year.unique())}
    train=frame[frame.Academic_Year!='26-27'];test=frame[frame.Academic_Year=='26-27']
    # Same-profile repeats removed from historical training in chronological check.
    train=train.loc[~np.isin(keys(train),keys(test))]
    m=fit_pipeline(train,kind);pp=predictions(m,test)
    transfer={name:metrics(test.outcome,p) for name,p in [('candidate',pp),('fixed50',np.full(len(test),.5)),('training_prior',np.full(len(test),train.outcome.mean()))]}
    transfer['note']='Train older cycles only, no recent calibration; same-profile old records removed. Not equivalent to mixed-cycle CV.'
    transfer['training_rows']=len(train)
    return {'kind':kind,'shared_rows':len(frame),'groups':len(set(keys(frame))),'folds':details,
            'metrics':{n:metrics(frame.outcome,p) for n,p in out.items()},'by_year':by_year,'oof':oof,'transfer':transfer,
            'scope_note':'Mixed-cycle grouped retrospective interpolation; NOT independent future validation.'}


def export():
    bundle={'version':VERSION,'models':{},'source_sha256':{},'sensitivities':{},'order_prior':PROGRAM_ORDER,
            'overlay_sha256':hashlib.sha256((ROOT/'research/data-overlay-v8.json').read_bytes()).hexdigest()}
    report={'version':VERSION,'label':'V8','protocol':'research/protocol-v8.md','evaluations':[],
            'scope_note':'User-requested fixed-exchange linear historical model; retrospective mixed-cycle CV, not future admission odds.'}
    for kind,(frame,screening) in datasets().items():
        print('Evaluating',kind,flush=True)
        ev=validate(frame,kind);report['evaluations'].append(ev)
        m=fit_pipeline(frame,kind);k=m['k'];bundle['models'][kind]=m
        alternatives=[]
        if kind=='SAT':
            assigned=folds(frame)
            for alt in [100.,150.,200.,250.,300.,400.,600.]:
                pp=np.full(len(frame),np.nan)
                for i in range(5):
                    train=frame.iloc[np.flatnonzero(assigned!=i)];held=frame.iloc[np.flatnonzero(assigned==i)]
                    pp[assigned==i]=predictions(fit(train,kind,alt),held)
                alternatives.append({'k':alt,'metrics':metrics(frame.outcome,pp)})
        bundle['sensitivities'][kind]={}
        for name,grouped,penalty in [('grouped',True,LAMBDA),('weak_penalty',False,.0001),('strong_penalty',False,.01)]:
            variant=fit(frame,kind,k,grouped=grouped,penalty=penalty)
            variant.update(gamma=m['gamma'],prevalence=m['prevalence'])
            bundle['sensitivities'][kind][name]=variant
        ev['fixed_tradeoff_sensitivity']=alternatives;ev['screening']=screening
        name=KINDS[kind][0];bundle['source_sha256'][name]=hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()
        print(kind,'k',k,'metrics',ev['metrics'],'transfer',ev['transfer'],flush=True)
    raw=canonical(bundle)
    (ROOT/'artifacts/model-v8.json').write_bytes(raw)
    (ROOT/'artifacts/manifest-v8.json').write_bytes(canonical({'version':VERSION,'sha256':hashlib.sha256(raw).hexdigest()}))
    (ROOT/'research/validation-v8.json').write_bytes(canonical(report))
    print('Frozen V8 exported',flush=True)


def load():
    raw=(ROOT/'artifacts/model-v8.json').read_bytes();manifest=json.loads((ROOT/'artifacts/manifest-v8.json').read_text());bundle=json.loads(raw)
    if manifest.get('version')!=VERSION or bundle.get('version')!=VERSION or hashlib.sha256(raw).hexdigest()!=manifest.get('sha256'):raise ValueError('V8 artifact identity')
    if set(bundle['models'])!=set(KINDS):raise ValueError('V8 kind identity')
    if hashlib.sha256((ROOT/'research/data-overlay-v8.json').read_bytes()).hexdigest()!=bundle.get('overlay_sha256'):raise ValueError('V8 overlay identity')
    if set(bundle['source_sha256'])!={x[0] for x in KINDS.values()}:raise ValueError('V8 source set')
    for name,sha in bundle['source_sha256'].items():
        if hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()!=sha:raise ValueError('V8 source changed')
    for kind,m in bundle['models'].items():
        t=m['parameters']
        if m['kind']!=kind or m['programs']!=PS or m['rounds']!=RS or len(t)!=2+len(PS)+len(RS):raise ValueError('V8 parameter dimensions')
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in t+[m['k'],m['scale'],m['centre']]):raise ValueError('V8 nonfinite parameters')
        if t[1]<0 or m['k']<=0 or m['scale']<=0:raise ValueError('V8 monotone coefficients')
        if kind=='SAT' and m['k'] not in SAT_K:raise ValueError('V8 fixed SAT exchange')
        if kind=='Bocconi' and m['k'] not in BT_K:raise ValueError('V8 fixed BT exchange')
        if not 0<=m['gamma']<=1 or not 0<m['prevalence']<1:raise ValueError('V8 shrinkage range')
        for easy,hard in PROGRAM_ORDER:
            if t[2+PS.index(easy)]<t[2+PS.index(hard)]-1e-7:raise ValueError('V8 programme order')
    return bundle


if __name__=='__main__':export()
