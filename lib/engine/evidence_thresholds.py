"""V10: fixed SAT/GPA exchange, grouped threshold likelihood, cohort effects.

Exact score/GPA/year/round matches are profile proxies, not verified identities.
Comparable outcomes in a proxy share a latent admission rank. Contradictory or
incomparable sets use a normalized Bernoulli composite likelihood, never relabel.
No forced perfect endpoints or cross-test maximum equivalence.
"""
from __future__ import annotations
import json,hashlib
import numpy as np
import pandas as pd
import coherent_thresholds as previous
from scipy.optimize import minimize
from scipy.special import expit
ROOT=previous.ROOT;KINDS=previous.KINDS;PS=previous.PS;RS=previous.RS
datasets=previous.datasets
VERSION='fixed20-profile-thresholds-v10-2026-10-24'
LAMBDAS=[.001,.01,.1]

def design(f,cycles):
 sat=f.test_kind.to_numpy()=='SAT';g=f.GPA.to_numpy(float)-9;s=f.score.to_numpy(float)
 cy=f.Academic_Year.to_numpy();known=np.array([v if v in cycles else cycles[-1] for v in cy])
 return np.column_stack([np.ones(len(f)),g+np.where(sat,(s-1500)/200,0),~sat,np.where(sat,0,(s-35)/10)]+[(f.Course==p).to_numpy(float) for p in PS]+[(f.Session=='Winter').to_numpy(float)]+[(known==c).astype(float) for c in cycles])

def units(f,grouped=True):
 # Both CS/AI or BIG cannot be ordered from the declared partial hierarchy.
 out=[];fallback=0
 for _,g in f.groupby(['test_kind','score','GPA','Academic_Year','Session'],sort=False):
  idx=g.index.to_numpy();courses=set(g.Course)
  comparable=not ('BIG' in courses and len(courses)>1) and not {'BEMACS','BAI'}<=courses
  contradictory=False
  rank={'BEMACC':0,'BIEM':1,'BIEF':2,'BEMACS':3,'BAI':3,'BIG':0}
  for a in g.itertuples():
   for b in g.itertuples():
    if a.outcome==0 and b.outcome==1 and rank[a.Course]<=rank[b.Course]:contradictory=True
  if grouped and comparable and not contradictory:
   out.append((idx[g.outcome.to_numpy()==1],idx[g.outcome.to_numpy()==0],False))
  else:
   fallback+=1
   out.append((idx[g.outcome.to_numpy()==1],idx[g.outcome.to_numpy()==0],True))
 return out,fallback

def fit(frames,lam=.01,grouped=True,cohort=True):
 f=previous.join(frames).reset_index(drop=True);cycles=sorted(f.Academic_Year.unique()) if cohort else ['pooled'];x=design(f,cycles);y=f.outcome.to_numpy(float)
 uu,fallback=units(f,grouped);n=x.shape[1]
 penalty=np.full(n,lam);penalty[:4]=[0,.001,.01,.001];penalty[4+len(PS)]=.01
 order=[]
 for easy,hard in previous.base.PROGRAM_ORDER:
  r=np.zeros(n);r[4+PS.index(easy)]=1;r[4+PS.index(hard)]=-1;order.append(r)
 a=np.array(order);lower=np.full(len(a),1e-5)
 gap_rows=np.array([a[i] for i,pair in enumerate(previous.base.PROGRAM_ORDER) if pair in previous.STRICT]);gap_center=gap_rows-gap_rows.mean(axis=0)
 def objective(t):
  z=x@t;p=expit(z);grad=np.zeros(len(f));loss=0.
  for aa,rr,independent in uu:
   ids=np.r_[aa,rr]
   if independent:
    loss+=np.mean(np.logaddexp(0,z[ids])-y[ids]*z[ids]);grad[ids]+=(p[ids]-y[ids])/len(ids)
   elif len(aa) and len(rr):
    ia=aa[np.argmin(z[aa])];ir=rr[np.argmax(z[rr])];gap=z[ia]-z[ir]
    if gap<=0:return 1e10+1e5*abs(gap),np.zeros(n)
    loss+=np.logaddexp(0,-z[ia])+np.logaddexp(0,z[ir])-np.log(-np.expm1(-gap))
    inverse=1/np.expm1(min(gap,700));grad[ia]+=-expit(-z[ia])-inverse;grad[ir]+=expit(z[ir])+inverse
   elif len(aa):
    ia=aa[np.argmin(z[aa])];loss+=np.logaddexp(0,-z[ia]);grad[ia]+=-expit(-z[ia])
   else:
    ir=rr[np.argmax(z[rr])];loss+=np.logaddexp(0,z[ir]);grad[ir]+=expit(z[ir])
  return loss/len(uu)+np.sum(penalty*t*t)+lam*np.sum((gap_center@t)**2),x.T@grad/len(uu)+2*penalty*t+2*lam*gap_center.T@(gap_center@t)
 start=np.zeros(n);start[0]=.5;start[1]=2;start[3]=2
 for p,v in {'BEMACC':.5,'BIEM':.3,'BIEF':0,'BEMACS':-.4,'BAI':-.5}.items():start[4+PS.index(p)]=v
 r=minimize(objective,start,jac=True,method='SLSQP',bounds=[(-20,20),(1e-5,20),(-20,20),(1e-5,20)]+[(-20,20)]*len(PS)+[(0,20)]+[(-20,20)]*len(cycles),constraints=[{'type':'ineq','fun':lambda t:a@t-lower,'jac':lambda t:a}],options={'ftol':1e-9,'maxiter':800})
 if not r.success or np.min(a@r.x-lower)<-1e-7:raise RuntimeError(str(r.message))
 return dict(parameters=r.x.tolist(),cycles=cycles,lambda_=lam,grouped=grouped,cohort=cohort,fit_rows=len(f),profile_units=len(uu),fallback_units=fallback,objective=float(r.fun))

def frame_predictions(m,frames):
 return {k:expit(design(f.assign(test_kind=k),m['cycles'])@np.array(m['parameters'])) for k,f in frames.items()}

def predict(m,score,gpa,program,session,cycle=None):
 s,g=np.broadcast_arrays(np.asarray(score,float),np.asarray(gpa,float));t=m['parameters'];cy=cycle if cycle in m['cycles'] else m['cycles'][-1]
 z=t[0]+t[1]*(g-9)+t[4+PS.index(program)]+t[4+len(PS)]*(session=='Winter')+t[5+len(PS)+m['cycles'].index(cy)]
 z+=t[1]*(s-1500)/200 if m['kind']=='SAT' else t[2]+t[3]*(s-35)/10
 return expit(z)

def select(frames):
 assign=previous.fold_assign(frames,3,'evidence-inner-v10');scores=[]
 for lam in LAMBDAS:
  ys=[];pp=[]
  for fold in range(3):
   tr={k:f.iloc[np.flatnonzero(assign[k]!=fold)] for k,f in frames.items()};he={k:f.iloc[np.flatnonzero(assign[k]==fold)] for k,f in frames.items()}
   mm=fit(tr,lam);pred=frame_predictions(mm,he)
   for k,f in he.items():ys.extend(f.outcome);pp.extend(pred[k])
  scores.append(dict(lambda_=lam,metrics=previous.base.metrics(ys,pp)))
 chosen=min(scores,key=lambda r:r['metrics']['log_loss'])['lambda_']
 return chosen,scores

def export():
 frames={k:f.reset_index(drop=True) for k,(f,_) in datasets().items()};old=json.loads((ROOT/'research/validation-coherent.json').read_text());assigned=previous.fold_assign(frames);oof={k:[] for k in frames};proof=[]
 for fold in range(5):
  print('V10 outer',fold,flush=True);tr={k:f.iloc[np.flatnonzero(assigned[k]!=fold)] for k,f in frames.items()};he={k:f.iloc[np.flatnonzero(assigned[k]==fold)] for k,f in frames.items()}
  lam,scores=select(tr);m=fit(tr,lam);pred=frame_predictions(m,he)
  proof.append(dict(fold=fold,lambda_=lam,selection=scores,train={k:f.id.tolist() for k,f in tr.items()},held={k:f.id.tolist() for k,f in he.items()}))
  for k,f in he.items():
   lookup={r['id']:r for r in next(e for e in old['evaluations'] if e['kind']==k)['oof']}
   for r,p in zip(f.itertuples(),pred[k]):oof[k].append(dict(id=r.id,outcome=int(r.outcome),cycle=r.Academic_Year,candidate=float(p),previous=lookup[r.id]['candidate'],fixed50=.5))
 older={};latest={}
 for k,f in frames.items():
  latest[k]=f[f.Academic_Year=='26-27'];tr=f[f.Academic_Year!='26-27'];older[k]=tr.loc[~np.isin(previous.base.keys(tr),previous.base.keys(latest[k]))]
 lam,selection=select(older);cm=fit(older,lam);cp=frame_predictions(cm,latest)
 final_lam,final_selection=select(frames);final=fit(frames,final_lam)
 report=dict(version=VERSION,evaluations=[],outer_proof=proof,chronological_model=cm,chronological_selection=selection,final_selection=final_selection,limitations=['Exact profile matches are proxy identities; preferences unknown.','26-27 has been inspected in development; not an untouched test.','27-28 forecast holds latest observed cohort effect; not a fitted future cohort.','Ordered programme hierarchy and nonnegative Winter effect remain user priors.','No forced SAT1600=BT50 or perfect-profile certainty.'])
 for k,f in frames.items():
  rows=oof[k];ev=dict(kind=k,shared_rows=len(f),oof=rows,metrics={name:previous.base.metrics([r['outcome'] for r in rows],[r[name] for r in rows]) for name in ['candidate','previous','fixed50']},transfer={'candidate':previous.base.metrics(latest[k].outcome,cp[k]),'previous':next(e for e in old['evaluations'] if e['kind']==k)['transfer']['candidate'],'fixed50':previous.base.metrics(latest[k].outcome,np.full(len(latest[k]),.5))})
  report['evaluations'].append(ev);print(k,{n:(v['brier'],v['log_loss']) for n,v in ev['metrics'].items()},'transfer',{n:(v['brier'],v['log_loss']) for n,v in ev['transfer'].items()},flush=True)
 # Sensitivity to the unverified profile linkage, evaluated with same training protocol.
 independent=fit(frames,final_lam,grouped=False)
 final.update(version=VERSION,models={},independent_sensitivity=independent,assumptions=report['limitations'])
 for k in frames:
  final['models'][k]={**{n:v for n,v in final.items() if n not in ['models','independent_sensitivity']},'kind':k,'k':200 if k=='SAT' else 10*final['parameters'][1]/final['parameters'][3],'centre':1500 if k=='SAT' else 35,'scale':200 if k=='SAT' else 10,'programs':PS,'rounds':RS}
 source_paths=['artifacts/model-coherent.json','research/validation-coherent.json','research/data-overlay-v8.json','research/data-overlay-programmes.json']
 final['source_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
 raw=previous.base.canonical(final);rr=previous.base.canonical(report)
 (ROOT/'artifacts/model-evidence.json').write_bytes(raw);(ROOT/'research/validation-evidence.json').write_bytes(rr)
 (ROOT/'artifacts/manifest-evidence.json').write_bytes(previous.base.canonical(dict(version=VERSION,sha256=hashlib.sha256(raw).hexdigest(),report_sha256=hashlib.sha256(rr).hexdigest())))
 print('Final',final_lam,np.round(final['parameters'],4),'profiles',final['profile_units'],'fallback',final['fallback_units'],flush=True)

def load():
 raw=(ROOT/'artifacts/model-evidence.json').read_bytes();man=json.loads((ROOT/'artifacts/manifest-evidence.json').read_text());b=json.loads(raw)
 if b['version']!=VERSION or man['version']!=VERSION or hashlib.sha256(raw).hexdigest()!=man['sha256']:raise ValueError('V10 model hash')
 if hashlib.sha256((ROOT/'research/validation-evidence.json').read_bytes()).hexdigest()!=man['report_sha256']:raise ValueError('V10 report hash')
 for path,sha in b['source_hashes'].items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('V10 source hash')
 for kind,m in b['models'].items():
  if kind not in KINDS or m['parameters'][1]<=0 or m['parameters'][3]<=0 or (kind=='SAT' and m['k']!=200):raise ValueError('V10 parameters')
 return b
if __name__=='__main__':export()
