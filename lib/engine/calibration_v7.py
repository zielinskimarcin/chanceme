"""Nested group-CV audit of global calibration; partial pooling and user order."""
from __future__ import annotations
import hashlib,json,math,itertools
import numpy as np
import model_v6 as base
from scipy.optimize import minimize

ROOT=base.ROOT
VERSION='pooled-ordered-calibration-v7-2026-10-23'
ORDER=[('BEMACC','BIEM'),('BIEM','BIEF'),('BIEF','BEMACS'),('BIEF','BAI')]
FAMILIES=['global','slope','context','joint','ordered']


def inner_folds(frame):
    groups=sorted(set(base.keys(frame)),key=lambda k:hashlib.sha256(('v7-inner|'+k).encode()).hexdigest())
    mapping={k:i%3 for i,k in enumerate(groups)}
    return np.array([mapping[k] for k in base.keys(frame)])


def configs(family):
    if family=='global':return [(0.,0.)]
    if family=='slope':return [(0.,b) for b in [.02,.2]]
    if family=='context':return [(c,0.) for c in [.01,.1,1.]]
    return list(itertools.product([.01,.1,1.],[.02,.2]))


def old_programme(rank):
    return dict(zip(rank['programs'],rank['parameters'][3:3+len(rank['programs'])]))


def matrix(rank,frame,family):
    z=np.array([base.rank_logit(rank,float(r.score),float(r.GPA),r.Course,r.Session) for r in frame.itertuples()],float)
    if not np.isfinite(z).all():raise ValueError('Invalid calibration rank/support')
    parts=[np.ones(len(frame)),z]
    if family in ['context','joint','ordered']:
        parts += [(frame.Course.to_numpy()==p).astype(float) for p in rank['programs']]
        parts += [(frame.Session.to_numpy()==t).astype(float) for t in rank['rounds']]
    return np.column_stack(parts)


def fit_calibrator(rank,frame,family,penalties):
    lc,lb=penalties
    if family=='global':
        delta,n=base.calibrate(rank,frame)
        return {'family':family,'parameters':[delta,1.],'programs':rank['programs'],'rounds':rank['rounds'],'penalties':[lc,lb],'fit_rows':n,'success':True}
    x=matrix(rank,frame,family);y=frame.outcome.to_numpy(float)
    free_slope=family in ['slope','joint','ordered']
    def loss(theta):
        logits=x@theta
        return float(np.mean(np.logaddexp(0,logits)-y*logits)+.002*theta[0]**2+lb*(theta[1]-1)**2+lc*np.sum(theta[2:]**2))
    def jac(theta):
        grad=x.T@(base.sigmoid(x@theta)-y)/len(y)
        grad[0]+=.004*theta[0];grad[1]+=2*lb*(theta[1]-1);grad[2:]+=2*lc*theta[2:]
        return grad
    bounds=[(-10,10),(.02,5.) if free_slope else (1.,1.)]+[(-10,10)]*(x.shape[1]-2)
    initial=np.zeros(x.shape[1]);initial[1]=1.
    constraints=[]
    if family=='ordered':
        old=old_programme(rank);ps=rank['programs'];rows=[]
        for easier,harder in ORDER:
            if easier in ps and harder in ps:
                row=np.zeros(x.shape[1]);row[1]=old[easier]-old[harder]
                row[2+ps.index(easier)]=1.;row[2+ps.index(harder)]=-1.;rows.append(row)
        if rows:
            a=np.array(rows)
            constraints=[{'type':'ineq','fun':lambda theta:a@theta,'jac':lambda theta:a}]
    result=minimize(loss,initial,jac=jac,bounds=bounds,constraints=constraints,
                    method='SLSQP' if constraints else 'L-BFGS-B',options={'maxiter':1000,'ftol':1e-11})
    if not result.success or not np.isfinite(result.fun):raise RuntimeError(f'Calibrator {family} failed: {result.message}')
    if constraints and np.min(a@result.x)<-1e-7:raise ValueError('Order not satisfied')
    return {'family':family,'parameters':result.x.tolist(),'programs':rank['programs'],'rounds':rank['rounds'],
            'penalties':[lc,lb],'fit_rows':len(frame),'success':True,'objective':float(result.fun)}


def cal_logit(cal,z,program,session):
    a,b,*offsets=cal['parameters'];extra=0.
    if offsets:
        ps,rs=cal['programs'],cal['rounds']
        if program not in ps or session not in rs:return None
        extra=offsets[ps.index(program)]+offsets[len(ps)+rs.index(session)]
    return float(a+b*z+extra)


def predict(rank,cal,score,gpa,program,session):
    z=base.rank_logit(rank,score,gpa,program,session)
    if z is None:return None
    if math.isinf(z):return float(z>0)
    value=cal_logit(cal,z,program,session)
    return None if value is None else float(base.sigmoid(value))


def predictions(rank,cal,frame):
    return np.array([predict(rank,cal,float(r.score),float(r.GPA),r.Course,r.Session) for r in frame.itertuples()],float)


def prepare_inner(older,recent,kind,outer_keys):
    folds=inner_folds(recent);parts=[]
    for fold in range(3):
        held=recent.iloc[np.flatnonzero(folds==fold)].copy();train=recent.iloc[np.flatnonzero(folds!=fold)].copy()
        heldkeys=set(base.keys(held));exclude=outer_keys|heldkeys
        oldtrain=older.loc[[k not in exclude for k in base.keys(older)]].copy()
        rank=base.fit_rank(oldtrain,kind)
        if heldkeys&set(base.keys(train)) or exclude&set(base.keys(oldtrain)):raise ValueError('Inner leakage')
        parts.append((rank,train,held))
    return parts


def select(parts,family):
    rows=[]
    for config in configs(family):
        losses=[];sizes=[]
        for rank,train,held in parts:
            cal=fit_calibrator(rank,train,family,config);p=predictions(rank,cal,held)
            losses.append(base.metrics(held.outcome.to_numpy(),p)['log_loss']);sizes.append(len(held))
        value=float(np.average(losses,weights=sizes));rows.append({'penalties':list(config),'logloss':value,'fold_losses':losses})
    best=min(rows,key=lambda row:(round(row['logloss'],6),-row['penalties'][0],-row['penalties'][1]))
    return tuple(best['penalties']),rows


def audit(kind,frame):
    older=frame[frame.Academic_Year!='26-27'].copy();recent=frame[frame.Academic_Year=='26-27'].copy().reset_index(drop=True)
    folds,_=base.fold_assignment(recent);oof={f:np.full(len(recent),np.nan) for f in FAMILIES};details=[]
    for fold in range(5):
        held=recent.iloc[np.flatnonzero(folds==fold)].copy();train=recent.iloc[np.flatnonzero(folds!=fold)].copy();heldkeys=set(base.keys(held))
        oldtrain=older.loc[[k not in heldkeys for k in base.keys(older)]].copy();rank=base.fit_rank(oldtrain,kind)
        parts=prepare_inner(older,train,kind,heldkeys)
        decisions={}
        for family in FAMILIES:
            cfg,search=select(parts,family);cal=fit_calibrator(rank,train,family,cfg)
            oof[family][folds==fold]=predictions(rank,cal,held)
            decisions[family]={'selected':list(cfg),'inner':search,'calibrator':cal}
        details.append({'fold':fold,'heldout_ids':held.id.tolist(),'heldout_fingerprints':sorted(heldkeys),'calibration_fingerprints':sorted(set(base.keys(train))),
                        'older_training_fingerprints':sorted(set(base.keys(oldtrain))),
                        'inner_audit':[{'heldout':sorted(set(base.keys(h))),'training':sorted(set(base.keys(t)))} for _,t,h in parts],
                        'decisions':decisions})
        print(kind,'outer',fold,'done',flush=True)
    if not all(np.isfinite(p).all() for p in oof.values()):raise ValueError('Incomplete OOF')
    stats={name:base.metrics(recent.outcome.to_numpy(),p) for name,p in oof.items()}
    contexts=[]
    for (p,t),part in recent.groupby(['Course','Session']):
        idx=part.index.to_numpy();contexts.append({'program':p,'session':t,'rows':len(part),'admitted':int(part.outcome.sum()),
            'metrics':{name:base.metrics(part.outcome.to_numpy(),values[idx]) for name,values in oof.items()},
            'mean_predictions':{name:float(values[idx].mean()) for name,values in oof.items()}})
    strong=(recent.score>= (1500 if kind=='SAT' else 40))&(recent.GPA>=8)
    strata={'strong':{'rows':int(strong.sum()),'metrics':{name:base.metrics(recent.loc[strong].outcome.to_numpy(),p[strong]) for name,p in oof.items()}}}
    parts=prepare_inner(older,recent,kind,set());rank=base.fit_rank(older,kind);final={};selection={}
    for family in FAMILIES:
        cfg,search=select(parts,family);final[family]=fit_calibrator(rank,recent,family,cfg);selection[family]={'selected':list(cfg),'inner':search}
    return {'kind':kind,'shared_rows':len(recent),'groups':len(set(base.keys(recent))),'metrics':stats,'contexts':contexts,'strata':strata,'folds':details,
            'oof':[{'id':r.id,'outcome':int(r.outcome),'fingerprint':base.fingerprint(r),'fold':int(folds[i]),**{name:float(p[i]) for name,p in oof.items()}} for i,r in enumerate(recent.itertuples())],
            'final_rank':rank,'final_calibrators':final,'final_selection':selection}


def run():
    report={'version':VERSION,'primary':'ordered','protocol':'research/protocol-v7.md','evaluations':[]}
    for kind,(frame,_) in base.datasets().items():
        result=audit(kind,frame);report['evaluations'].append(result)
        print(kind,'RESULT',{name:{k:round(v,6) for k,v in stat.items() if k in ['brier','log_loss','bias','auc']} for name,stat in result['metrics'].items()},flush=True)
        (ROOT/'research/calibration-audit-v7.json').write_bytes(base.canonical(report))


if __name__=='__main__':run()
