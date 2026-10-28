"""Fixed academic exchange, positive domain-ordered programme thresholds."""
from __future__ import annotations
import hashlib,json,math
import numpy as np
import model_v8 as base
from scipy.optimize import minimize,linprog
ROOT=base.ROOT;KINDS=base.KINDS
PS=base.PS;RS=base.RS
VERSION='fixed20-ordered-programme-thresholds-2026-10-24'
ALPHAS=[.0001,.001,.01]
FIXED_K={'SAT':200.,'Bocconi':2.5}
STRICT=[('BIEM','BIEF'),('BIEF','BEMACS'),('BIEF','BAI')]
EPS=1e-9
predict=base.predict;raw_probability=base.raw_probability


def datasets():
    frames=base.datasets()
    overlay=json.loads((ROOT/'research/data-overlay-programmes.json').read_text())
    original_ids={r['id'] for r in json.loads((ROOT/'research/data-overlay-v8.json').read_text())['label_corrections']}
    for kind,(frame,screening) in frames.items():
        for correction in overlay['label_corrections']:
            if correction['id'] in original_ids:continue
            mask=frame.id==correction['id']
            if mask.any():
                row=frame.loc[mask].iloc[0]
                if mask.sum()!=1 or row.Decision!=correction['original'] or row.Course!=correction['program'] or row.score!=correction['score'] or row.GPA!=correction['gpa'] or row.Academic_Year!=correction['cycle'] or row.Session!=correction['session']:raise ValueError('Programme correction identity')
                frame.loc[mask,'Decision']=correction['corrected'];frame.loc[mask,'outcome']=int(correction['corrected']=='accept');screening['label_corrected']+=1
    return frames


def matched_scale(frame):
    pairs=[]
    for key,group in frame.groupby(['score','GPA','Academic_Year','Session']):
        outcomes={p:int(g.outcome.iloc[0]) for p,g in group.groupby('Course') if g.outcome.nunique()==1}
        if 'BIEM' in outcomes and 'BIEF' in outcomes:pairs.append({'key':list(key),'a':outcomes['BIEM'],'b':outcomes['BIEF']})
    ar=sum(p['a']==1 and p['b']==0 for p in pairs);ra=sum(p['a']==0 and p['b']==1 for p in pairs)
    return {'scale':(max(ar-ra,0)+.5)/(len(pairs)+1) if pairs else .1,
            'scale_source':'training_SAT_matched_labels' if pairs else 'declared_no_match_reference_0.1',
            'n':len(pairs),'AR':ar,'RA':ra,'pairs':pairs}


def matrices(frame,kind,strict=True):
    k=FIXED_K[kind];x=base.design(frame,kind,k);order=[];lower=[];d=[]
    for easy,hard in base.PROGRAM_ORDER:
        row=np.zeros(x.shape[1]);row[2+PS.index(easy)]=1;row[2+PS.index(hard)]=-1
        order.append(row);lower.append(1e-8 if strict and (easy,hard) in STRICT else 0.)
    for easy,hard in STRICT:
        row=np.zeros(x.shape[1]);row[2+PS.index(easy)]=1;row[2+PS.index(hard)]=-1;d.append(row)
    maximum=[];minimum=[]
    for p in PS:
        for r in RS:
            indicators=[float(p==q) for q in PS]+[float(r==s) for s in RS]
            maximum.append([1.,(KINDS[kind][3]-base.CENTRES[kind]+k)/base.SCALES[kind]]+indicators)
            minimum.append([1.,((1040. if kind=='SAT' else 17.)-base.CENTRES[kind]-3*k)/base.SCALES[kind]]+indicators)
    order=np.array(order);d=np.array(d);maximum=np.array(maximum);minimum=np.array(minimum)
    accepted=x if kind=='Bocconi' else x[frame.outcome.to_numpy()==1];rejected=x if kind=='Bocconi' else x[frame.outcome.to_numpy()==0]
    a=np.vstack([order,accepted,-rejected,maximum,-minimum]);lb=np.concatenate([lower,np.full(len(accepted),EPS),np.full(len(rejected),-1+EPS),np.ones(len(maximum)),np.zeros(len(minimum))])
    return x,d,a,lb


def fit(frame,kind,scale_info,alpha,start_model=None):
    x,d,a,lb=matrices(frame,kind,strict=alpha>0);y=frame.outcome.to_numpy(float);scale=scale_info['scale']
    start_model=start_model or base.fit(frame,kind,FIXED_K[kind]);old=np.array(start_model['parameters'])
    def objective(t):
        p=np.clip(x@t,EPS,1-EPS);g=np.maximum(d@t,1e-12)
        prior=alpha*np.sum(g/scale-np.log(g/scale)) if alpha>0 else 0.
        return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))+.001*np.sum(t[2:]**2)+prior)
    def jac(t):
        p=np.clip(x@t,EPS,1-EPS);g=np.maximum(d@t,1e-12)
        raw=x@t;active=np.where(y==1,raw<1,raw>0)
        gradient=x.T@((p-y)/(p*(1-p))*active)/len(y);gradient[2:]+=.002*t[2:]
        return gradient+alpha*d.T@((1/scale-1/g)*((d@t)>1e-12)) if alpha>0 else gradient
    bounds=[(-5,5),(.000001,2)]+[(-2,2)]*(x.shape[1]-2)
    initials=[old]
    if alpha>0:
        phase_lb=lb.copy();phase_lb[:len(base.PROGRAM_ORDER)]=[.0001 if p in STRICT else 0 for p in base.PROGRAM_ORDER]
        phase=linprog(np.zeros(len(old)),A_ub=-a,b_ub=-phase_lb,bounds=bounds,method='highs')
        if not phase.success:raise RuntimeError('Strict programme order infeasible')
        initials=[(1-weight)*old+weight*phase.x for weight in [.15,.5]]
    candidates=[]
    for initial in initials:
        result=minimize(objective,initial,jac=jac,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':lambda t:a@t-lb,'jac':lambda t:a}],options={'maxiter':1500,'ftol':1e-10})
        if result.success and np.isfinite(result.fun) and np.min(a@result.x-lb)>=-1e-7 and (alpha==0 or np.min(d@result.x)>0):candidates.append(result)
    if not candidates:raise RuntimeError('Ordered threshold optimisation failed')
    best=min(candidates,key=lambda r:r.fun)
    return dict(start_model,parameters=best.x.tolist(),objective=float(best.fun),programme_alpha=alpha,
                programme_prior_scale=scale_info['scale'],programme_prior_counts={k:scale_info[k] for k in ['n','AR','RA']},
                programme_prior='positive_Gamma_mode_regularisation_user_order' if alpha>0 else 'nonstrict_order_baseline',strict_programme_order=STRICT if alpha>0 else [],objective_kind='clipped_linear_bernoulli',training_domain='both_sided_stability_guard' if kind=='Bocconi' else 'outcome_halfspaces')


def inner_parts(frames):
    assignments={kind:base.folds(f,3,'programme-inner-20261024') for kind,f in frames.items()}
    parts=[]
    for fold in range(3):
        train={k:f.iloc[np.flatnonzero(assignments[k]!=fold)] for k,f in frames.items()};held={k:f.iloc[np.flatnonzero(assignments[k]==fold)] for k,f in frames.items()}
        for kind in frames:assert not set(base.keys(train[kind]))&set(base.keys(held[kind]))
        scale=matched_scale(train['SAT']);starts={k:base.fit(f,k,FIXED_K[k]) for k,f in train.items()}
        parts.append((train,held,scale,starts))
    return parts


def select(frames):
    parts=inner_parts(frames);scores={k:[] for k in frames};proof=[]
    for train,held,scale,starts in parts:
        proof.append({'scale_info':scale,'training':{k:sorted(set(base.keys(f))) for k,f in train.items()},'heldout':{k:sorted(set(base.keys(f))) for k,f in held.items()}})
    for kind in frames:
        for alpha in ALPHAS:
            ys=[];ps=[]
            for train,held,scale,starts in parts:
                m=fit(train[kind],kind,scale,alpha,starts[kind]);ps.extend(base.predictions(m,held[kind]).tolist());ys.extend(held[kind].outcome.tolist())
            scores[kind].append({'alpha':alpha,'metrics':base.metrics(ys,ps)})
    chosen={}
    for kind in frames:
        best=min(s['metrics']['log_loss'] for s in scores[kind]);chosen[kind]=max(s['alpha'] for s in scores[kind] if s['metrics']['log_loss']<=best+1e-6)
    return chosen,scores,proof


def export():
    frames={k:f.reset_index(drop=True) for k,(f,_) in datasets().items()};assign={k:base.folds(f) for k,f in frames.items()}
    rows={k:[] for k in frames};details=[]
    for fold in range(5):
        print('Programme outer fold',fold,flush=True)
        train={k:f.iloc[np.flatnonzero(assign[k]!=fold)] for k,f in frames.items()};held={k:f.iloc[np.flatnonzero(assign[k]==fold)] for k,f in frames.items()}
        chosen,scores,proof=select(train);scale=matched_scale(train['SAT'])
        for kind in frames:
            assert not set(base.keys(train[kind]))&set(base.keys(held[kind]))
            baseline=fit(train[kind],kind,scale,0);m=fit(train[kind],kind,scale,chosen[kind],baseline)
            p=base.predictions(m,held[kind]);old=base.predictions(baseline,held[kind]);prior=float(train[kind].outcome.mean())
            for r,value,previous,key in zip(held[kind].itertuples(),p,old,base.keys(held[kind])):
                rows[kind].append({'id':r.id,'fingerprint':key,'outcome':int(r.outcome),'cycle':r.Academic_Year,'program':r.Course,'round':r.Session,'score':float(r.score),'gpa':float(r.GPA),'fold':fold,'candidate':float(value),'fixed20_v8':float(previous),'fixed50':.5,'recent_prior':prior})
        details.append({'fold':fold,'chosen':chosen,'scores':scores,'inner_proof':proof,'scale_info':scale,
                        'training':{k:sorted(set(base.keys(f))) for k,f in train.items()},'heldout':{k:sorted(set(base.keys(f))) for k,f in held.items()}})
    chosen,scores,proof=select(frames);scale=matched_scale(frames['SAT']);models={k:fit(f,k,scale,chosen[k]) for k,f in frames.items()}
    previous=base.load();bundle={'version':VERSION,'models':models,'source_sha256':previous['source_sha256'],
        'base_overlay_sha256':previous['overlay_sha256'],
        'overlay_sha256':hashlib.sha256((ROOT/'research/data-overlay-programmes.json').read_bytes()).hexdigest(),
        'baseline_version':base.VERSION,'strict_order':STRICT,'matched_scale':scale,'selection':scores}
    report={'version':VERSION,'label':'V8 · progi kierunków','protocol':'research/programme-protocol-2026-10-24.md','evaluations':[],
            'scope_note':'Retrospective grouped development. Strict order is a user prior; CS/AI gap magnitudes depend on positive regularisation, not verified population cutoffs.','outer_proof':details,'final_inner_proof':proof}
    for kind,f in frames.items():
        ys=[r['outcome'] for r in rows[kind]]
        ev={'kind':kind,'shared_rows':len(f),'groups':len(set(base.keys(f))),'folds':details,'oof':rows[kind],
            'metrics':{n:base.metrics(ys,[r[n] for r in rows[kind]]) for n in ['candidate','fixed20_v8','fixed50','recent_prior']},
            'scope_note':report['scope_note'],'selected_alpha':chosen[kind],'prior_scale':scale}
        report['evaluations'].append(ev)
        print(kind,'alpha',chosen[kind],'metrics',{n:(round(v['brier'],5),round(v['log_loss'],5)) for n,v in ev['metrics'].items()},flush=True)
    raw=base.canonical(bundle);report_raw=base.canonical(report)
    (ROOT/'artifacts/model-programmes.json').write_bytes(raw);(ROOT/'research/validation-programmes.json').write_bytes(report_raw)
    (ROOT/'artifacts/manifest-programmes.json').write_bytes(base.canonical({'version':VERSION,'sha256':hashlib.sha256(raw).hexdigest(),'report_sha256':hashlib.sha256(report_raw).hexdigest()}))
    print('Frozen programme thresholds exported',flush=True)


def load():
    raw=(ROOT/'artifacts/model-programmes.json').read_bytes();manifest=json.loads((ROOT/'artifacts/manifest-programmes.json').read_text());bundle=json.loads(raw)
    if bundle.get('version')!=VERSION or manifest.get('version')!=VERSION or hashlib.sha256(raw).hexdigest()!=manifest.get('sha256'):raise ValueError('Programme artifact identity')
    if hashlib.sha256((ROOT/'research/validation-programmes.json').read_bytes()).hexdigest()!=manifest.get('report_sha256'):raise ValueError('Programme report identity')
    if hashlib.sha256((ROOT/'research/data-overlay-v8.json').read_bytes()).hexdigest()!=bundle['base_overlay_sha256']:raise ValueError('Base overlay identity')
    if hashlib.sha256((ROOT/'research/data-overlay-programmes.json').read_bytes()).hexdigest()!=bundle['overlay_sha256']:raise ValueError('Programme overlay identity')
    for name,sha in bundle['source_sha256'].items():
        if hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()!=sha:raise ValueError('Programme source identity')
    if set(bundle['models'])!=set(KINDS):raise ValueError('Programme model kinds')
    for kind,m in bundle['models'].items():
        t=np.array(m['parameters'])
        if m['kind']!=kind or m['k']!=FIXED_K[kind] or m['programs']!=PS or m['rounds']!=RS or len(t)!=2+len(PS)+len(RS) or not np.isfinite(t).all() or t[1]<=0:raise ValueError('Programme parameters')
        x,d,a,lb=matrices(datasets()[kind][0],kind)
        if np.min(a@t-lb)<-1e-7 or np.min(d@t)<=0:raise ValueError('Programme constraints')
    return bundle


if __name__=='__main__':export()
