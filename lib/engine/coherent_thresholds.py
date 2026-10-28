"""Joint SAT/Bocconi clipped-linear model. Offline export only, no serving fit.

Declared domain assumptions: same maximum at SAT1600/BT50; Winter no harder
than Early; positive user-ordered programme gaps. Academic SAT/GPA exchange fixed.
"""
from __future__ import annotations
import hashlib,json
import numpy as np
import pandas as pd
import programme_thresholds as old
from scipy.optimize import minimize,linprog
base=old.base; ROOT=old.ROOT; PS=old.PS; RS=old.RS; KINDS=old.KINDS
VERSION='joint-fixed20-coherent-thresholds-2026-10-24'
ALPHAS=old.ALPHAS; STRICT=old.STRICT; EPS=1e-9
predict=base.predict;raw_probability=base.raw_probability;datasets=old.datasets

def join(frames):
 return pd.concat([f.assign(test_kind=k) for k,f in frames.items()],ignore_index=True)

def design(f):
 sat=f.test_kind.to_numpy()=='SAT';g=f.GPA.to_numpy(float)-9;s=f.score.to_numpy(float)
 return np.column_stack([np.ones(len(f)),g+np.where(sat,(s-1500)/200,.5),np.where(sat,0,(s-50)/10)]+[(f.Course==p).to_numpy(float) for p in PS]+[(f.Session=='Winter').to_numpy(float)])

def matrices(f,strict=True):
 x=design(f);n=x.shape[1];order=[];lo=[];d=[]
 for easy,hard in base.PROGRAM_ORDER:
  r=np.zeros(n);r[3+PS.index(easy)]=1;r[3+PS.index(hard)]=-1;order.append(r);lo.append(1e-8 if strict and (easy,hard) in STRICT else 0.)
 for easy,hard in STRICT:
  r=np.zeros(n);r[3+PS.index(easy)]=1;r[3+PS.index(hard)]=-1;d.append(r)
 anchors=[]
 for p in PS:
  for session in RS:
   for kind,score,gpa in [('SAT',1600,10),('SAT',1040,6),('Bocconi',17,6)]:
    anchors.append(dict(test_kind=kind,score=score,GPA=gpa,Course=p,Session=session))
 ax=design(pd.DataFrame(anchors));maxx=ax[::3];minx=np.vstack([ax[1::3],ax[2::3]])
 a=np.vstack([order,x[f.outcome.to_numpy()==1],-x[f.outcome.to_numpy()==0],maxx,-minx])
 lb=np.r_[lo,np.full(int(f.outcome.sum()),EPS),np.full(int((f.outcome==0).sum()),-1+EPS),np.ones(len(maxx)),np.zeros(len(minx))]
 return x,np.array(d),a,lb

def fit(frames,alpha,scale_info=None):
 f=join(frames);x,d,a,lb=matrices(f,alpha>0);y=f.outcome.to_numpy(float)
 # Every verified record has equal likelihood weight. No manual test-type boosts.
 scale_info=scale_info or old.matched_scale(frames['SAT']);scale=scale_info['scale']
 def objective(t):
  p=np.clip(x@t,EPS,1-EPS);g=np.maximum(d@t,1e-12)
  return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))+.001*np.sum(t[3:]**2)+(alpha*np.sum(g/scale-np.log(g/scale)) if alpha else 0))
 def jac(t):
  raw=x@t;p=np.clip(raw,EPS,1-EPS);active=np.where(y==1,raw<1,raw>0)
  grad=x.T@((p-y)/(p*(1-p))*active)/len(y);grad[3:]+=.002*t[3:]
  if alpha:
   g=np.maximum(d@t,1e-12);grad+=alpha*d.T@((1/scale-1/g)*((d@t)>1e-12))
  return grad
 bounds=[(-5,5),(1e-6,2),(1e-6,2)]+[(-2,2)]*len(PS)+[(0,2)]
 phase_lb=lb.copy();phase_lb[:len(base.PROGRAM_ORDER)]=[.005 if strictpair in STRICT and alpha else 0. for strictpair in base.PROGRAM_ORDER]
 phase_lb[len(base.PROGRAM_ORDER):len(base.PROGRAM_ORDER)+len(f)]=np.where(lb[len(base.PROGRAM_ORDER):len(base.PROGRAM_ORDER)+len(f)]>0,.02,-.98)
 phase=linprog(np.zeros(x.shape[1]),A_ub=-a,b_ub=-phase_lb,bounds=bounds,method='highs')
 if not phase.success:raise RuntimeError('Joint interior initialisation infeasible')
 initials=[]
 for A0,Q0 in [(.25,.4),(.5,.6)]:
  ref=np.zeros(x.shape[1]);ref[0]=.6;ref[1]=A0;ref[2]=Q0;ref[-1]=.02
  qp=minimize(lambda t:float(np.sum((t-ref)**2)),phase.x,jac=lambda t:2*(t-ref),method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':lambda t:a@t-phase_lb,'jac':lambda t:a}],options={'maxiter':1500,'ftol':1e-12})
  if not qp.success or np.min(a@qp.x-phase_lb)<-1e-7:raise RuntimeError('Joint initial projection failed')
  initials.append(qp.x)
 results=[]
 for init in initials:
  result=minimize(objective,init,jac=jac,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':lambda t:a@t-lb,'jac':lambda t:a}],options={'maxiter':1600,'ftol':1e-10})
  if np.isfinite(result.fun) and np.min(a@result.x-lb)>=-1e-7 and (not alpha or np.min(d@result.x)>0):
   # A valid feasible stationary result remains usable if SLSQP stops on its line search.
   if result.success:results.append(result)
 if not results:raise RuntimeError('Joint optimiser failed')
 r=min(results,key=lambda r:r.fun);t=r.x;A=t[1];Q=t[2]
 models={}
 for kind in KINDS:
  intercept=t[0] if kind=='SAT' else t[0]+.5*A-1.5*Q
  slope=A if kind=='SAT' else Q;k=200. if kind=='SAT' else 10*A/Q
  params=[intercept,slope]+t[3:3+len(PS)].tolist()+[0.,t[-1]]
  models[kind]={'kind':kind,'k':k,'centre':base.CENTRES[kind],'scale':base.SCALES[kind],'programs':PS,'rounds':RS,'parameters':params,'fit_rows':len(frames[kind]),'cycles':sorted(frames[kind].Academic_Year.unique()),'programme_alpha':alpha,'programme_prior_scale':scale,'programme_prior_counts':{k:scale_info[k] for k in ['n','AR','RA']},'objective_kind':'joint_clipped_linear_bernoulli','conditional_rank_endpoints':True,'optimizer_success':True,'training_domain':'outcome_halfspaces','strict_programme_order':STRICT,'gamma':1.,'prevalence':float(frames[kind].outcome.mean())}
 return {'parameters':t.tolist(),'models':models,'alpha':alpha,'scale_info':scale_info,'objective':float(r.fun),'optimizer_objectives':[float(v.fun) for v in results],'optimizer_start_count':len(initials),'minimum_constraint_slack':float(np.min(a@t-lb)),'active_constraints':np.flatnonzero(a@t-lb<1e-6).tolist(),'sat_equivalent_per_bt_point':20*Q/A,'winter_probability_offset':float(t[-1]),'assumptions':['SAT1600_equals_Bocconi50_at_same_GPA','Winter_no_harder_than_Early','fixed_SAT20_per_GPA0.1','positive_user_programme_order']}

def predictions(m,frames):return {k:base.predictions(m['models'][k],f) for k,f in frames.items()}

def fold_assign(frames,n=5,salt='v8-outer'):
 assignment={k:base.folds(f,n,salt) for k,f in frames.items()}
 # Source S07 row81 includes both tests from one applicant. Union the entire
 # same-kind fingerprint groups, preserving all remaining previous assignments.
 sat=frames['SAT'];bt=frames['Bocconi'];sm=sat.id=='SAT:362';bm=bt.id=='Bocconi:110'
 if sm.any() and bm.any():
  target=int(assignment['SAT'][np.flatnonzero(sm)[0]]);btkey=base.keys(bt.loc[bm])[0]
  assignment['Bocconi'][np.array(base.keys(bt))==btkey]=target
 elif sm.any()!=bm.any():
  # A one-sided absence leaves no known counterpart in these data; keep the
  # remaining fingerprint group. Unknown applicant identities remain a limitation.
  pass
 return assignment

def parts_for(frames):
 assignment=fold_assign(frames,3,'programme-inner-20261024');parts=[];proof=[]
 for fold in range(3):
  train={k:f.iloc[np.flatnonzero(assignment[k]!=fold)] for k,f in frames.items()};held={k:f.iloc[np.flatnonzero(assignment[k]==fold)] for k,f in frames.items()}
  proof.append({'training':{k:sorted(set(base.keys(f))) for k,f in train.items()},'heldout':{k:sorted(set(base.keys(f))) for k,f in held.items()},'scale_info':old.matched_scale(train['SAT'])});parts.append((train,held))
 return parts,proof

def select(frames):
 parts,proof=parts_for(frames)
 scores=[]
 for alpha in ALPHAS:
  ys=[];pp=[];by_kind={k:([],[]) for k in frames}
  for train,held in parts:
   m=fit(train,alpha);pred=predictions(m,held)
   for k,f in held.items():ys.extend(f.outcome.tolist());pp.extend(pred[k].tolist());by_kind[k][0].extend(f.outcome.tolist());by_kind[k][1].extend(pred[k].tolist())
  scores.append({'alpha':alpha,'metrics':base.metrics(ys,pp),'by_kind':{k:base.metrics(*values) for k,values in by_kind.items()}})
 best=min(s['metrics']['log_loss'] for s in scores);chosen=max(s['alpha'] for s in scores if s['metrics']['log_loss']<=best+1e-6)
 return chosen,scores,proof

def baseline_select(frames):
 parts,proof=parts_for(frames);scores={k:[] for k in frames}
 for kind in frames:
  for alpha in ALPHAS:
   yy=[];pp=[]
   for train,held in parts:
    m=old.fit(train[kind],kind,old.matched_scale(train['SAT']),alpha)
    yy.extend(held[kind].outcome.tolist());pp.extend(base.predictions(m,held[kind]).tolist())
   scores[kind].append({'alpha':alpha,'metrics':base.metrics(yy,pp)})
 chosen={k:min(scores[k],key=lambda s:(s['metrics']['log_loss'],-s['alpha']))['alpha'] for k in frames}
 return chosen,scores,proof

def export():
 prior_bundle=old.load();prior_report=json.loads((ROOT/'research/validation-programmes.json').read_text());frames={k:f.reset_index(drop=True) for k,(f,_) in datasets().items()};assigned=fold_assign(frames)
 prior_rows={ev['kind']:{r['id']:r for r in ev['oof']} for ev in prior_report['evaluations']};rows={k:[] for k in frames};proof=[]
 for fold in range(5):
  print('Joint outer fold',fold,flush=True);train={k:f.iloc[np.flatnonzero(assigned[k]!=fold)] for k,f in frames.items()};held={k:f.iloc[np.flatnonzero(assigned[k]==fold)] for k,f in frames.items()}
  alpha,scores,inner=select(train);m=fit(train,alpha);pred=predictions(m,held);ba,bs,bp=baseline_select(train);baseline={k:old.fit(f,k,old.matched_scale(train['SAT']),ba[k]) for k,f in train.items()};baseline0={k:old.fit(f,k,old.matched_scale(train['SAT']),0) for k,f in train.items()}
  for kind,f in held.items():
   for r,p,key in zip(f.itertuples(),pred[kind],base.keys(f)):
    previous=prior_rows[kind][r.id]
    if previous['outcome']!=r.outcome or previous['fingerprint']!=key:raise ValueError('Prior OOF identity mismatch')
    rows[kind].append({'id':r.id,'fingerprint':key,'outcome':int(r.outcome),'cycle':r.Academic_Year,'program':r.Course,'round':r.Session,'score':float(r.score),'gpa':float(r.GPA),'fold':fold,'candidate':float(p),'previous_programmes':base.predict(baseline[kind],r.score,r.GPA,r.Course,r.Session),'fixed20_v8':base.predict(baseline0[kind],r.score,r.GPA,r.Course,r.Session),'previous_original_folds':previous['candidate'],'fixed50':.5,'recent_prior':float(train[kind].outcome.mean())})
  proof.append({'fold':fold,'alpha':alpha,'scores':scores,'inner_proof':inner,'scale_info':m['scale_info'],'training':{k:sorted(set(base.keys(f))) for k,f in train.items()},'heldout':{k:sorted(set(base.keys(f))) for k,f in held.items()},'fit':m,'baseline_alpha':ba,'baseline_scores':bs,'baseline_inner_proof':bp})
 alpha,scores,inner=select(frames);model=fit(frames,alpha)
 older={};latest={}
 for kind,f in frames.items():
  latest[kind]=f[f.Academic_Year=='26-27'];tr=f[f.Academic_Year!='26-27'];older[kind]=tr.loc[~np.isin(base.keys(tr),base.keys(latest[kind]))]
 print('Joint chronological fit',flush=True);ca,cs,cp=select(older);cm=fit(older,ca);chronop=predictions(cm,latest)
 report={'version':VERSION,'label':'V9 · wspólny model testów i tur','protocol':'research/coherent-protocol-2026-10-24.md','outer_proof':proof,'final_inner_proof':inner,'chronological_inner_proof':cp,'evaluations':[],'scope_note':'Retrospective grouped validation. Endpoint alignment and Winter ordering are declared domain assumptions, not proven test equivalence or a forecast guarantee.'}
 for kind,f in frames.items():
  ys=[r['outcome'] for r in rows[kind]];pold=next(ev for ev in prior_report['evaluations'] if ev['kind']==kind)
  ev={'kind':kind,'shared_rows':len(f),'groups':len(set(base.keys(f))),'oof':rows[kind],'metrics':{n:base.metrics(ys,[r[n] for r in rows[kind]]) for n in ['candidate','previous_programmes','previous_original_folds','fixed20_v8','fixed50','recent_prior']},'selected_alpha':alpha,'prior_scale':model['scale_info'],'scope_note':report['scope_note'],'transfer':{'candidate':base.metrics(latest[kind].outcome,chronop[kind]),'previous_programmes':pold['transfer']['candidate'],'fixed50':base.metrics(latest[kind].outcome,[.5]*len(latest[kind])),'training_prior':base.metrics(latest[kind].outcome,[float(older[kind].outcome.mean())]*len(latest[kind])),'training_rows':len(older[kind]),'selected_alpha':ca,'note':'Train older cycles; remove repeated same-kind profile keys. No future calibration.'},'diagnosis':{'score_min':float(f.score.min()),'score_max':float(f.score.max()),'gpa_min':float(f.GPA.min()),'by_round':{str(k):{'rows':len(g),'acceptance':float(g.outcome.mean()),'mean_score':float(g.score.mean()),'mean_gpa':float(g.GPA.mean())} for k,g in f.groupby('Session')},'by_cycle':{str(k):{'rows':len(g),'acceptance':float(g.outcome.mean())} for k,g in f.groupby('Academic_Year')},'by_programme_round':{str(k):{'rows':len(g),'acceptance':float(g.outcome.mean())} for k,g in f.groupby(['Course','Session'])}}}
  report['evaluations'].append(ev);print(kind,ev['metrics'],'transfer',ev['transfer'],flush=True)
 bundle={'version':VERSION,**model,'matched_scale':model['scale_info'],'strict_order':STRICT,'source_sha256':prior_bundle['source_sha256'],'base_overlay_sha256':prior_bundle['base_overlay_sha256'],'overlay_sha256':prior_bundle['overlay_sha256'],'previous_model_sha256':hashlib.sha256((ROOT/'artifacts/model-programmes.json').read_bytes()).hexdigest(),'previous_report_sha256':hashlib.sha256((ROOT/'research/validation-programmes.json').read_bytes()).hexdigest(),'selection':scores}
 raw=base.canonical(bundle);rr=base.canonical(report);(ROOT/'artifacts/model-coherent.json').write_bytes(raw);(ROOT/'research/validation-coherent.json').write_bytes(rr);(ROOT/'artifacts/manifest-coherent.json').write_bytes(base.canonical({'version':VERSION,'sha256':hashlib.sha256(raw).hexdigest(),'report_sha256':hashlib.sha256(rr).hexdigest()}));print('Joint frozen',alpha,'betaSAT/BT',model['sat_equivalent_per_bt_point'],'Winter',model['winter_probability_offset'],flush=True)

def load():
 raw=(ROOT/'artifacts/model-coherent.json').read_bytes();man=json.loads((ROOT/'artifacts/manifest-coherent.json').read_text());b=json.loads(raw)
 if b['version']!=VERSION or man['version']!=VERSION or hashlib.sha256(raw).hexdigest()!=man['sha256']:raise ValueError('Coherent artifact identity')
 for path,sha in [('research/validation-coherent.json',man['report_sha256']),('research/data-overlay-v8.json',b['base_overlay_sha256']),('research/data-overlay-programmes.json',b['overlay_sha256']),('artifacts/model-programmes.json',b['previous_model_sha256']),('research/validation-programmes.json',b['previous_report_sha256'])]:
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('Coherent linked artifact identity')
 for name,sha in b['source_sha256'].items():
  if hashlib.sha256((ROOT/'legacy'/name).read_bytes()).hexdigest()!=sha:raise ValueError('Coherent source identity')
 if set(b['models'])!=set(KINDS) or b['models']['SAT']['k']!=200.:raise ValueError('Coherent model kinds or SAT exchange')
 f=join({k:f for k,(f,_) in datasets().items()});x,d,a,lb=matrices(f);t=np.array(b['parameters'])
 if len(t)!=3+len(PS)+1 or not np.isfinite(t).all() or t[1]<=0 or t[2]<=0 or t[-1]<0 or np.min(a@t-lb)<-1e-7 or np.min(d@t)<=0:raise ValueError('Coherent constraints')
 for kind,m in b['models'].items():
  if m['kind']!=kind or m['programs']!=PS or m['rounds']!=RS or len(m['parameters'])!=2+len(PS)+len(RS):raise ValueError('Coherent model dimensions')
  for pr in PS:
   for session in RS:
    for score in ([1040,1450,1600] if kind=='SAT' else [17,35,50]):
     sample=pd.DataFrame([dict(test_kind=kind,score=score,GPA=8.3,Course=pr,Session=session)])
     if abs(float(design(sample)@t)-raw_probability(m,score,8.3,pr,session))>1e-10:raise ValueError('Coherent export algebra')
    for gpa in [6,8.3,10]:
     if abs(raw_probability(m,KINDS[kind][3],gpa,pr,session)-raw_probability(b['models']['SAT'],1600,gpa,pr,session))>1e-9:raise ValueError('Coherent endpoint equality')
 return b

if __name__=='__main__':export()
