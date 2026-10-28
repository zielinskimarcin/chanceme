"""Shared programme/round calibration across tests; fixed domain programme order."""
from __future__ import annotations
import json,hashlib,math,itertools
import numpy as np
import model_v6 as base
from calibration_v7 import ORDER
from scipy.optimize import minimize
ROOT=base.ROOT
VERSION='pooled-programme-calibration-v7-2026-10-23'
KINDS=list(base.KINDS)
PROGRAMS=sorted(base.PROGRAMS)
ROUNDS=['Early','Winter']
CONFIGS=list(itertools.product([.01,.1,1.],[.02,.2]))


def combined():
    import pandas as pd
    frames=[]
    for kind,(f,_) in base.datasets().items():
        f=f.copy();f['kind']=kind;f['key']=[kind+'|'+k for k in base.keys(f)];frames.append(f)
    return pd.concat(frames,ignore_index=True)


def fit_ranks(older,excluded):
    out={}
    for kind in KINDS:
        part=older[(older.kind==kind)&~older.key.isin(excluded)].copy()
        out[kind]=base.fit_rank(part,kind)
    return out


def academic_logit(rank,score,gpa,p,t):
    # Shared programme/round effects are learned downstream. Their absence in an
    # older resample must not discard the otherwise valid academic curve.
    theta=rank['parameters'];s=float(base.academic(score,gpa,rank['kind'],theta[0]))
    if s<=0:return -math.inf
    if s>=1:return math.inf
    ref=rank['years'].index(rank['reference_cycle'])
    year=theta[3+len(rank['programs'])+len(rank['rounds'])+ref]
    return float(theta[1]*(float(base.logit_index(s))-rank['normalizer']['mean'])/rank['normalizer']['sd']+theta[2]+year)


def design(ranks,frame):
    rows=[]
    for r in frame.itertuples():
        z=academic_logit(ranks[r.kind],float(r.score),float(r.GPA),r.Course,r.Session)
        if z is None or not math.isfinite(z):raise ValueError('Pooled rank has unsupported calibration row')
        rows.append([float(r.kind==k) for k in KINDS]+[z*float(r.kind==k) for k in KINDS]+[float(r.Course==p) for p in PROGRAMS]+[float(r.Session==t) for t in ROUNDS])
    return np.array(rows)


def fit(ranks,frame,config):
    lc,lb=config;x=design(ranks,frame);y=frame.outcome.to_numpy(float);nk=len(KINDS)
    def fun(theta):
        z=x@theta
        return float(np.mean(np.logaddexp(0,z)-y*z)+.002*np.sum(theta[:nk]**2)+lb*np.sum((theta[nk:2*nk]-1)**2)+lc*np.sum(theta[2*nk:]**2))
    def jac(theta):
        g=x.T@(base.sigmoid(x@theta)-y)/len(y);g[:nk]+=.004*theta[:nk];g[nk:2*nk]+=2*lb*(theta[nk:2*nk]-1);g[2*nk:]+=2*lc*theta[2*nk:];return g
    a=[]
    for easy,hard in ORDER:
        row=np.zeros(x.shape[1]);row[2*nk+PROGRAMS.index(easy)]=1;row[2*nk+PROGRAMS.index(hard)]=-1;a.append(row)
    a=np.array(a);initial=np.zeros(x.shape[1]);initial[nk:2*nk]=1
    bounds=[(-10,10)]*nk+[(.02,5.)]*nk+[(-10,10)]*(x.shape[1]-2*nk)
    result=minimize(fun,initial,jac=jac,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':lambda th:a@th,'jac':lambda th:a}],options={'maxiter':1000,'ftol':1e-11})
    if not result.success or np.min(a@result.x)<-1e-7:raise RuntimeError('Pooled calibrator failed '+result.message)
    return {'parameters':result.x.tolist(),'kinds':KINDS,'programs':PROGRAMS,'rounds':ROUNDS,'penalties':list(config),'fit_rows':len(frame),'family':'shared_ordered','objective':float(result.fun)}


def predict(ranks,cal,kind,score,gpa,p,t):
    z=academic_logit(ranks[kind],score,gpa,p,t)
    if z is None:return None
    if math.isinf(z):return float(z>0)
    kinds=cal['kinds'];ps=cal['programs'];rs=cal['rounds'];th=cal['parameters'];nk=len(kinds);i=kinds.index(kind)
    return float(base.sigmoid(th[i]+th[nk+i]*z+th[2*nk+ps.index(p)]+th[2*nk+len(ps)+rs.index(t)]))


def predictions(ranks,cal,frame):return np.array([predict(ranks,cal,r.kind,float(r.score),float(r.GPA),r.Course,r.Session) for r in frame.itertuples()],float)


def inner_assignment(frame):
    groups=sorted(set(frame.key),key=lambda k:hashlib.sha256(('pooled-v7-inner|'+k).encode()).hexdigest());mp={k:i%3 for i,k in enumerate(groups)}
    return np.array([mp[k] for k in frame.key])


def prepare(older,recent,excluded):
    fold=inner_assignment(recent);parts=[]
    for i in range(3):
        held=recent.iloc[np.flatnonzero(fold==i)];train=recent.iloc[np.flatnonzero(fold!=i)];excluded2=excluded|set(held.key)
        ranks=fit_ranks(older,excluded2)
        audit={'heldout':sorted(set(held.key)),'training':sorted(set(train.key)),'older_training':sorted(set(older.loc[~older.key.isin(excluded2),'key']))}
        if set(audit['heldout'])&set(audit['training']) or excluded2&set(audit['older_training']):raise ValueError('Nested leakage')
        parts.append((ranks,train,held,audit))
    return parts


def select(parts):
    choices=[]
    for config in CONFIGS:
        losses=[];ns=[]
        for ranks,train,held,_ in parts:
            cal=fit(ranks,train,config);p=predictions(ranks,cal,held)
            losses.append(base.metrics(held.outcome.to_numpy(),p)['log_loss']);ns.append(len(held))
        choices.append({'penalties':list(config),'logloss':float(np.average(losses,weights=ns)),'folds':losses})
    best=min(choices,key=lambda x:(round(x['logloss'],6),-x['penalties'][0],-x['penalties'][1]))
    return tuple(best['penalties']),choices


def run():
    frame=combined();older=frame[frame.Academic_Year!='26-27'];recent=frame[frame.Academic_Year=='26-27'].copy().reset_index(drop=True)
    recent['fold']=-1
    for kind in KINDS:
        part=recent[recent.kind==kind];fold,_=base.fold_assignment(part);recent.loc[part.index,'fold']=fold
    p=np.full(len(recent),np.nan);details=[]
    for i in range(5):
        held=recent[recent.fold==i];train=recent[recent.fold!=i];exclude=set(held.key)
        ranks=fit_ranks(older,exclude);parts=prepare(older,train,exclude);config,selection=select(parts);cal=fit(ranks,train,config)
        p[held.index]=predictions(ranks,cal,held)
        details.append({'fold':i,'heldout':sorted(set(held.key)),'calibration':sorted(set(train.key)),
                        'older_training':sorted(set(older.loc[~older.key.isin(exclude),'key'])),'inner_audit':[a for _,_,_,a in parts],'selection':selection,'selected':list(config),'calibrator':cal})
        print('pooled outer',i,'done',flush=True)
    ranks=fit_ranks(older,set());config,selection=select(prepare(older,recent,set()));cal=fit(ranks,recent,config)
    report={'version':VERSION,'scope':'disclosed second retrospective development iteration; notfuturevalidation','ranks':ranks,'calibrator':cal,'final_selection':selection,'selected':list(config),'folds':details,'evaluations':[]}
    frozen=json.loads((ROOT/'research/validation-v6.json').read_text())
    for kind in KINDS:
        idx=np.flatnonzero(recent.kind.to_numpy()==kind);part=recent.iloc[idx];y=part.outcome.to_numpy();pred=p[idx]
        old=next(e for e in frozen['evaluations'] if e['kind']==kind);previous={r['id']:r for r in old['oof']}
        for row in part.itertuples():
            prev=previous[row.id]
            if prev['fingerprint']!=base.fingerprint(row) or prev['fold']!=row.fold or prev['outcome']!=row.outcome:raise ValueError('Rows differ')
        globalp=np.array([previous[r.id]['candidate'] for r in part.itertuples()])
        stats={'candidate':base.metrics(y,pred),'global_v6':base.metrics(y,globalp),'fixed50':base.metrics(y,np.full(len(y),.5))}
        contexts=[]
        for (program,session),c in part.groupby(['Course','Session']):
            local=c.index.to_numpy();contextpred=p[local];actual=c.outcome.to_numpy();oldpred=np.array([previous[r.id]['candidate'] for r in c.itertuples()])
            contexts.append({'program':program,'session':session,'rows':len(c),'admitted':int(c.outcome.sum()),'candidate':base.metrics(actual,contextpred),'global_v6':base.metrics(actual,oldpred),'predicted_mean':float(contextpred.mean())})
        strong=(part.score>=(1500 if kind=='SAT' else 40))&(part.GPA>=8)
        report['evaluations'].append({'kind':kind,'shared_rows':len(part),'metrics':stats,'contexts':contexts,
             'strong':{'rows':int(strong.sum()),'candidate':base.metrics(part.loc[strong].outcome.to_numpy(),pred[strong]),'global_v6':base.metrics(part.loc[strong].outcome.to_numpy(),globalp[strong])},
             'oof':[{'id':row.id,'fingerprint':base.fingerprint(row),'fold':int(row.fold),'outcome':int(row.outcome),'candidate':float(p[row.Index]),'global_v6':float(previous[row.id]['candidate'])} for row in part.itertuples()]})
        print(kind,stats,flush=True)
    (ROOT/'research/pooled-audit-v7.json').write_bytes(base.canonical(report))
    print('final',cal,flush=True)


if __name__=='__main__':run()
