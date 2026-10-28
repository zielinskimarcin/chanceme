"""V11 relative corrections conditional on frozen V9. No cohort recalibration.

SAT BIEM is bit-identical to V9; BT BIEM at40/GPA9 stays at V9 in both rounds.
Profile proxies and endpoint normalization are declared model assumptions.
Selection CV is conditional on an already fitted V9, NOT independent validation.
"""
from __future__ import annotations
import json,hashlib,copy
import numpy as np
import coherent_thresholds as old
import evidence_thresholds as ev
from scipy.optimize import minimize
ROOT=old.ROOT;KINDS=old.KINDS;PS=old.PS;RS=old.RS;datasets=old.datasets
VERSION='v9-level-relative-thresholds-v11-2026-10-24'
COLS=[p for p in PS if p!='BIEM'];ALPHAS=[0.,.01,.1,1.]

def arrays(frames,b):
 f=old.join(frames).reset_index(drop=True);sat=f.test_kind.to_numpy()=='SAT';score=f.score.to_numpy(float);gpa=f.GPA.to_numpy(float)
 anchor=np.array([old.raw_probability(b['models'][r.test_kind],r.score if r.test_kind=='SAT' else 40,r.GPA if r.test_kind=='SAT' else 9,'BIEM',r.Session) for r in f.itertuples()])
 return f,sat,score,gpa,anchor

def probabilities(t,f,sat,score,gpa,anchor):
 d=np.array([t[COLS.index(p)] if p!='BIEM' else 0 for p in f.Course]);raw=(anchor+np.where(sat,0,t[-2]*(score-40)+t[-1]*(gpa-9))-d)/(1-d)
 return np.clip(raw,0,1)

def unit_loss(y,p,units):
 q=np.clip(p,1e-9,1-1e-9);loss=0
 for aa,rr,ind in units:
  if ind:
   ids=np.r_[aa,rr];loss-=np.mean(y[ids]*np.log(q[ids])+(1-y[ids])*np.log1p(-q[ids]))
  elif len(aa) and len(rr):loss-=np.log(max(min(q[aa])-max(q[rr]),1e-9))
  elif len(aa):loss-=np.log(min(q[aa]))
  else:loss-=np.log1p(-max(q[rr]))
 return float(loss/len(units))

def fit(frames,b,alpha):
 f,sat,score,gpa,anchor=arrays(frames,b);y=f.outcome.to_numpy(float);units,fallback=ev.units(f)
 a=[]
 for easy,hard in old.base.PROGRAM_ORDER:a.append([float(p==hard)-float(p==easy) for p in COLS]+[0,0])
 a=np.array(a)
 bt_reference=b['models']['Bocconi']
 endpoint_delta=float(old.raw_probability(bt_reference,50,10,'BIEM','Early')-old.raw_probability(bt_reference,40,9,'BIEM','Early'))
 gr=np.array([[float(p=='BIEF') for p in COLS]+[0,0],[float(p=='BEMACS')-float(p=='BIEF') for p in COLS]+[0,0],[float(p=='BAI')-float(p=='BIEF') for p in COLS]+[0,0]])
 center=gr-gr.mean(axis=0)
 def objective(t):
  return unit_loss(y,probabilities(t,f,sat,score,gpa,anchor),units)+.001*np.sum(t[:-2]**2)+alpha*np.sum((center@t)**2)
 results=[]
 for scale in [.6,1.4]:
  initial=np.array([(0 if p=='BEMACC' else -.05 if p=='BIG' else .08 if p=='BIEF' else .17)*scale for p in COLS]+[.014,.1])
  r=minimize(objective,initial,method='SLSQP',bounds=[(-.5,.8)]*len(COLS)+[(1e-5,.1),(0,1)],constraints=[{'type':'ineq','fun':lambda t:a@t-1e-6},{'type':'eq','fun':lambda t:10*t[-2]+t[-1]-endpoint_delta}],options={'ftol':1e-9,'maxiter':700})
  if r.success and np.isfinite(r.fun) and (a@r.x>=1e-6-1e-7).all():results.append(r)
 if not results:raise RuntimeError('V11 optimisation failed')
 result=min(results,key=lambda r:r.fun)
 return dict(parameters=result.x.tolist(),alpha=alpha,objective=float(result.fun),units=len(units),fallback_units=fallback,optimizer_success=True)

def export():
 base=old.load();frames={k:f.reset_index(drop=True) for k,(f,_) in datasets().items()};assign=old.fold_assign(frames,3,'v11-conditional');selection=[];oofs={}
 for alpha in ALPHAS:
  print('V11 alpha',alpha,flush=True);losses=[];preds={k:np.full(len(f),np.nan) for k,f in frames.items()}
  for fold in range(3):
   tr={k:f.iloc[np.flatnonzero(assign[k]!=fold)] for k,f in frames.items()};he={k:f.iloc[np.flatnonzero(assign[k]==fold)] for k,f in frames.items()};m=fit(tr,base,alpha)
   f,sat,s,g,a=arrays(he,base);p=probabilities(np.array(m['parameters']),f,sat,s,g,a);u,_=ev.units(f);losses.append(unit_loss(f.outcome.to_numpy(float),p,u))
   start=0
   for k,ff in he.items():preds[k][assign[k]==fold]=p[start:start+len(ff)];start+=len(ff)
  selection.append(dict(alpha=alpha,conditional_unit_loss=float(np.mean(losses))));oofs[alpha]=preds
 chosen=min(selection,key=lambda r:r['conditional_unit_loss'])['alpha'];fit_result=fit(frames,base,chosen);t=fit_result['parameters'];ds={p:t[COLS.index(p)] if p!='BIEM' else 0 for p in PS}
 models={}
 for kind,prior in base['models'].items():
  m=copy.deepcopy(prior);pars=m['parameters'];pars[0]+=pars[2+PS.index('BIEM')]
  for p in PS:pars[2+PS.index(p)]=0
  if kind=='Bocconi':
   early=float(old.raw_probability(prior,40,9,'BIEM','Early'));pars[0]=early-5*t[-2];pars[1]=10*t[-2];m['k']=t[-1]/t[-2]
  m.update(parameters=pars,thresholds=ds,conditional_rank_endpoints=False);models[kind]=m
 bundle=dict(version=VERSION,models=models,fit=fit_result,selection=selection,scale_info=base['scale_info'],base_sha256=hashlib.sha256((ROOT/'artifacts/model-coherent.json').read_bytes()).hexdigest(),assumptions=['V9 SAT BIEM level and slope fixed by user instruction','V9 BT BIEM pivot40/GPA9 and perfect50/GPA10 fixed; reference anchors are design assumptions, not learned cutoffs','Profile linkage is a proxy; preference order unverified','Positive hierarchy and endpoint-normalized threshold form are assumptions','No cohort offset, prevalence correction or V10 intercept'])
 report=dict(version=VERSION,label='V11 · poziom V9',scope_note='Conditional adaptation of frozen V9. V9 has seen all rows; these are NOT independent future or end-to-end validation metrics.',selection=selection,evaluations=[])
 for kind,f in frames.items():
  oldp=old.base.predictions(base['models'][kind],f);newp=np.array([predict(models[kind],r.score,r.GPA,r.Course,r.Session) for r in f.itertuples()]);report['evaluations'].append(dict(kind=kind,shared_rows=len(f),conditional_metrics={'v9_fixed':old.base.metrics(f.outcome,oldp),'candidate_conditional_cv':old.base.metrics(f.outcome,oofs[chosen][kind]),'candidate_fit':old.base.metrics(f.outcome,newp)},scope_note=report['scope_note']))
 raw=old.base.canonical(bundle);rr=old.base.canonical(report)
 (ROOT/'artifacts/model-targeted.json').write_bytes(raw);(ROOT/'research/validation-targeted.json').write_bytes(rr);(ROOT/'artifacts/manifest-targeted.json').write_bytes(old.base.canonical(dict(version=VERSION,sha256=hashlib.sha256(raw).hexdigest(),report_sha256=hashlib.sha256(rr).hexdigest())))
 print('Selected',chosen,ds,'BT slope GPA',t[-2:],flush=True)

def effective_linear(m,program):
 d=m['thresholds'][program];t=m['parameters'];q=1-d
 return dict(m,parameters=[(t[0]-d)/q,t[1]/q]+[0]*len(PS)+[v/q for v in t[2+len(PS):]])

def raw_probability(m,score,gpa,program,session):
 return old.raw_probability(effective_linear(m,program),score,gpa,program,session)

def predict(m,score,gpa,program,session):
 p=np.clip(raw_probability(m,score,gpa,program,session),0,1)
 return float(p) if np.ndim(p)==0 else p

def load():
 old.load();raw=(ROOT/'artifacts/model-targeted.json').read_bytes();man=json.loads((ROOT/'artifacts/manifest-targeted.json').read_text());b=json.loads(raw)
 if b['version']!=VERSION or man['version']!=VERSION or hashlib.sha256(raw).hexdigest()!=man['sha256']:raise ValueError('V11 identity')
 if hashlib.sha256((ROOT/'research/validation-targeted.json').read_bytes()).hexdigest()!=man['report_sha256']:raise ValueError('V11 report identity')
 if b['base_sha256']!=hashlib.sha256((ROOT/'artifacts/model-coherent.json').read_bytes()).hexdigest():raise ValueError('V11 base identity')
 if b['models']['SAT']['k']!=200:raise ValueError('V11 exchange')
 return b
if __name__=='__main__':export()

def describe(m,score,gpa,program,session,base_diagnostics,bundle):
 out=dict(base_diagnostics);d=m['thresholds'][program];q=1-d;t=m['parameters'];eff=effective_linear(m,program);slope=eff['parameters'][1]/m['scale'];ps=m['programs'];rs=m['rounds'];offset=t[2+len(ps)+rs.index(session)]/q
 out.update(model_label='V11 · poziom V9',model_version=VERSION,experimental_probability=predict(m,score,gpa,program,session),tradeoff_per_gpa=m['k'])
 out['formula']=dict(raw_percent=float(raw_probability(m,score,gpa,program,session))*100,intercept_percent=t[0]/q*100,slope_percent_per_test_point=slope*100,gpa_percent_per_point=slope*m['k']*100,programme_offset_percent=-d/q*100,round_offset_percent=offset*100,academic_centre=m['centre'],gpa_centre=9,fit_note='Poziom i nachylenie SAT BIEM z V9 są zamrożone. Pozostałe kierunki: (indeks BIEM − próg kierunku)/(1 − próg kierunku). Ta normalizacja zachowuje górny koniec skali i stałe20SAT/0,1GPA. BT ma osobno dopasowane wpływy testu i GPA, przy zachowanych punktach odniesienia V9:40/9 oraz50/10. Bez korekty rocznika. Kotwice i kolejność kierunków są założeniami, a nie oficjalnymi progami.')
 out['coherence']=dict(sat_points_per_bocconi_point=(bundle['models']['Bocconi']['parameters'][1]/10)/(bundle['models']['SAT']['parameters'][1]/200),gpa_percent_per_point=slope*m['k']*100,winter_minus_early_pp=100*(t[-1]-t[-2])/q,note='SAT: zachowane20 punktów za0,1GPA. W BT wpływy GPA i testu dopasowano osobno.40BT/GPA9 i maksymalny profil50BT/GPA10 zachowują poziom V9. Nie wymuszamy równości SAT1600 i BT50 przy innych GPA. Wartości w tabeli dotyczą wybranego testu i kierunku.')
 def threshold(p):
  mm=effective_linear(m,p);tt=mm['parameters'];sl=tt[1]/mm['scale'];ro=tt[2+len(ps)+rs.index(session)]
  return mm['centre']+(.5-tt[0]-ro)/sl
 base_threshold=threshold('BIEM');basep=predict(m,score,gpa,'BIEM',session)
 out['programme_difficulty']=[dict(row,offset_percent=100*(predict(m,score,gpa,row['code'],session)-basep),threshold_50=threshold(row['code']),threshold_gap=threshold(row['code'])-base_threshold) for row in out['programme_difficulty']]
 out['programme_method']='Poziom V9 BIEM pozostaje stały. Odstępy wyznacza wiarygodność wspólnych profili i dodatnia hierarchia BIEM→BIEF→CS/AI. Przy małej próbie odstępy pożyczają wspólną skalę, której siłę wybrano w warunkowym podziale na3części. To model przybliżonej zależności rekordów: identyczny profil nie potwierdza tożsamości i nie znamy pełnych preferencji. BIG bez zadanej pozycji. Korekta w tabeli dotyczy bieżącego profilu, nie stałego bonusu.'
 out['programme_evidence']['alpha']=bundle['fit']['alpha']
 out['validation_note']='Korekta warunkowa względem zamrożonego V9. V9 znał te rekordy: podział korekty na3części nie jest niezależną walidacją całego modelu ani prognozą przyszłego naboru. Zachowanie poziomu V9 jest świadomą decyzją produktu; historyczne problemy transferu pozostają.'
 return out
