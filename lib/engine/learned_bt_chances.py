"""V14: normalized sigmoid learned from source-audited BT-only outcomes.
SAT is immutable. No minimum-score equivalence and no fixed 26.4 join.
50BT=1600SAT is the user-required model contract, not official conversion.
"""
from __future__ import annotations
import copy,hashlib,json
import numpy as np
import unified_chances as base
from scipy.optimize import minimize,brentq
ROOT=base.ROOT;KINDS=base.KINDS;PS=base.PS;RS=base.RS;datasets=base.datasets
VERSION='source-audited-bt-curve-v14-2026-10-25'
ALPHAS=(.0001,.001,.01,.1)
AUDIT=ROOT/'research/bt-source-audit-v14.json'

def qualified():
 audit=json.loads(AUDIT.read_text());ids={r['id'] for r in audit['ledger'] if r['status']=='included_source_supported'}
 return datasets()['Bocconi'][0].loc[lambda f:f.id.isin(ids)].reset_index(drop=True)

def factor(parameters,score):
 a,b=parameters;z=a+b*(np.asarray(score,float)-40)/10;top=a+b
 return np.exp(-np.logaddexp(0,-z)+np.logaddexp(0,-top))

def frame_predictions(parameters,frame,sat):
 maximum,d,_=base.components(frame,sat)
 return np.clip((np.clip(maximum,0,1)*factor(parameters,frame.score.to_numpy(float))-d)/(1-d),0,1)

def fit(frame,sat,alpha):
 f=frame.reset_index(drop=True);units,_=base.prior.ev.units(f.assign(test_kind='Bocconi'));y=f.outcome.to_numpy(float)
 def objective(t):return base.prior.unit_loss(y,frame_predictions(t,f,sat),units)+alpha*float(t@t)
 results=[]
 for a,b in [(-1,.5),(1,2),(3,4),(0,.1),(5,7)]:
  r=minimize(objective,[a,b],method='L-BFGS-B',bounds=[(None,None),(0,None)],options={'maxiter':1000,'ftol':1e-11,'gtol':1e-7})
  if r.success and np.isfinite(r.fun):results.append(r)
 if not results:raise RuntimeError('No converged BT fit')
 r=min(results,key=lambda r:r.fun)
 return dict(parameters=r.x.tolist(),alpha=alpha,objective=float(r.fun),rows=len(f),units=len(units),optimizer_success=True)

def select(frame,sat,salt='unified-v12-conditional'):
 f=frame.reset_index(drop=True);assignment=base.prior.old.base.folds(f,3,salt);results=[];oofs={};proof={}
 for alpha in ALPHAS:
  oof=np.full(len(f),np.nan);folds=[]
  for fold in range(3):
   tr=f.loc[assignment!=fold];he=f.loc[assignment==fold];fit_result=fit(tr,sat,alpha);oof[assignment==fold]=frame_predictions(fit_result['parameters'],he,sat)
   folds.append(dict(fold=fold,train=tr.id.tolist(),held=he.id.tolist(),fit=fit_result))
  metrics=base.prior.old.base.metrics(f.outcome,oof);results.append(dict(alpha=alpha,metrics=metrics));oofs[alpha]=oof;proof[alpha]=folds
 # Brier is the selection criterion; log loss is reported separately.
 chosen=min(results,key=lambda r:(r['metrics']['brier'],-r['alpha']))['alpha']
 return chosen,results,oofs[chosen],proof[chosen]

def raw_probability(m,score,gpa,program,session):
 if m['kind']=='SAT':return base.raw_probability(m,score,gpa,program,session)
 maximum=np.clip(base.prior.raw_probability(m['sat_reference'],1600,gpa,'BIEM',session),0,1)
 d=m['thresholds'][program]
 return (maximum*factor(m['bt_curve_parameters'],score)-d)/(1-d)

def predict(m,score,gpa,program,session):
 p=np.clip(raw_probability(m,score,gpa,program,session),0,1)
 if m['kind']=='Bocconi':p=np.where(np.asarray(score,float)<17,0,p)
 return float(p) if np.ndim(p)==0 else p

def export():
 import low_bt_chances as before
 old=before.load();bundle=copy.deepcopy(base.load());sat=bundle['models']['SAT'];f=qualified();full=datasets()['Bocconi'][0].reset_index(drop=True)
 chosen,selection,oof,proof=select(f,sat);final=fit(f,sat,chosen)
 print('Selected',final,flush=True)
 # Outer CV re-selects regularization on outer-training profiles only.
 assign=base.prior.old.base.folds(f,3,'v14-nested-outer');nested=np.full(len(f),np.nan);outer=[]
 for fold in range(3):
  tr=f.loc[assign!=fold].reset_index(drop=True);he=f.loc[assign==fold];alpha,_,_,_=select(tr,sat,'v14-inner-'+str(fold));rr=fit(tr,sat,alpha);nested[assign==fold]=frame_predictions(rr['parameters'],he,sat);outer.append(dict(fold=fold,alpha=alpha,train=tr.id.tolist(),held=he.id.tolist(),parameters=rr['parameters']));print('Outer',fold,rr['parameters'],flush=True)
 # Later-cycle diagnostic: no re-calibration of either test or cohort offset.
 train=f[f.Academic_Year<'26-27'].reset_index(drop=True);held=f[f.Academic_Year=='26-27'];temporal=None
 if len(train)>5 and train.outcome.nunique()==2 and len(held):
  alpha,_,_,_=select(train,sat,'v14-temporal-inner');rr=fit(train,sat,alpha)
  temporal=dict(train_rows=len(train),held_rows=len(held),alpha=alpha,candidate=base.prior.old.base.metrics(held.outcome,frame_predictions(rr['parameters'],held,sat)),previous=base.prior.old.base.metrics(held.outcome,[before.predict(old['models']['Bocconi'],r.score,r.GPA,r.Course,r.Session) for r in held.itertuples()]),note='Chronological BT adaptation only. Frozen SAT/programmes have seen these outcomes; not independent future validation.')
 bt=copy.deepcopy(sat);bt.update(kind='Bocconi',sat_reference=copy.deepcopy(sat),bt_curve_parameters=final['parameters'],bt_fit=final,fit_rows=len(f));bundle.update(version=VERSION,previous_version=before.VERSION,source_audit_sha256=hashlib.sha256(AUDIT.read_bytes()).hexdigest(),source_snapshots=json.loads(AUDIT.read_text())['snapshots'],bt_fit=final,bt_fit_ids=f.id.tolist(),previous_low_bt_sha256=hashlib.sha256((ROOT/'artifacts/model-low-bt.json').read_bytes()).hexdigest());bundle['models']['Bocconi']=bt
 bundle['assumptions']=['SAT V9 level, fixed20 and programme thresholds unchanged','50BT=1600SAT at every valid GPA/programme/round is a required coherence assumption, not an official conversion','BT-only explicit source-reported programme outcomes; not admissions-office verified','Normalized sigmoid is a modelling assumption; two parameters learned with ridge chosen by grouped CV Brier','No minimum-score SAT anchor, no fixed26.4 join, no cohort adjustment','Below observed BT and above observed maximum are extrapolation; self-selection and GPA/preference/category uncertainty remain']
 evaluations=[]
 for kind,ff in [('SAT',datasets()['SAT'][0]),('Bocconi',f)]:
  prev=np.array([before.predict(old['models'][kind],r.score,r.GPA,r.Course,r.Session) for r in ff.itertuples()]);now=np.array([predict(bundle['models'][kind],r.score,r.GPA,r.Course,r.Session) for r in ff.itertuples()]);met=base.prior.old.base.metrics
  evaluations.append(dict(kind=kind,shared_rows=len(ff),scope_note='BT adaptation conditional on frozen SAT/programmes, which already saw these records. Source declarations, not verified university decisions. Nested CV re-selects BT regularization only.',conditional_metrics=dict(v13_fixed=met(ff.outcome,prev),candidate_conditional_cv=met(ff.outcome,nested if kind=='Bocconi' else now),candidate_selection_cv=met(ff.outcome,oof if kind=='Bocconi' else now),candidate_fit=met(ff.outcome,now),fixed50=met(ff.outcome,np.full(len(ff),.5)))))
 loo=[]
 for index,r in enumerate(f.itertuples()):
  rr=fit(f.drop(index).reset_index(drop=True),sat,chosen);temp=copy.deepcopy(bt);temp['bt_curve_parameters']=rr['parameters'];loo.append(dict(omitted=r.id,parameters=rr['parameters'],p26_4_gpa10=predict(temp,26.4,10,'BIEM','Winter'),p30_gpa10=predict(temp,30,10,'BIEM','Winter')))
 report=dict(version=VERSION,label='V14 · krzywa BT ze sprawdzonych źródeł',scope_note=evaluations[-1]['scope_note'],evaluations=evaluations,selection=selection,selected_alpha=chosen,selection_fold_proof=proof,nested_fold_proof=outer,temporal=temporal,leave_one_profile_out_fixed_alpha=loo,training_rows=len(f),reported_accepts=int(f.outcome.sum()),historical_comparison_rows=len(full),source_audit='bt-source-audit-v14.json',tail_observed_rows=0,observed_bt_range=[float(f.score.min()),float(f.score.max())],full_93_secondary=dict(previous=base.prior.old.base.metrics(full.outcome,[before.predict(old['models']['Bocconi'],r.score,r.GPA,r.Course,r.Session) for r in full.itertuples()]),candidate=base.prior.old.base.metrics(full.outcome,frame_predictions(final['parameters'],full,sat))),extrapolation_note='No empirical observations below26.4. A learned smooth sigmoid is extrapolated there, without fixed probability or SAT-minimum alignment. These are conditional estimates, not calibrated future-population chances.')
 raw=base.prior.old.base.canonical(bundle);rr=base.prior.old.base.canonical(report);(ROOT/'artifacts/model-learned-bt.json').write_bytes(raw);(ROOT/'research/validation-learned-bt.json').write_bytes(rr);(ROOT/'artifacts/manifest-learned-bt.json').write_bytes(base.prior.old.base.canonical(dict(version=VERSION,sha256=hashlib.sha256(raw).hexdigest(),report_sha256=hashlib.sha256(rr).hexdigest())))
 print('Report',[(e['kind'],e['conditional_metrics']['candidate_conditional_cv']['brier'],e['conditional_metrics']['candidate_conditional_cv']['log_loss']) for e in evaluations],flush=True)

def load():
 sat=base.load()['models']['SAT'];raw=(ROOT/'artifacts/model-learned-bt.json').read_bytes();man=json.loads((ROOT/'artifacts/manifest-learned-bt.json').read_text());b=json.loads(raw)
 if b['version']!=VERSION or man['version']!=VERSION or hashlib.sha256(raw).hexdigest()!=man['sha256']:raise ValueError('V14 model identity')
 if hashlib.sha256((ROOT/'research/validation-learned-bt.json').read_bytes()).hexdigest()!=man['report_sha256']:raise ValueError('V14 report identity')
 if b['models']['SAT']!=sat or b['models']['Bocconi']['sat_reference']!=sat or b['models']['Bocconi']['thresholds']!=sat['thresholds']:raise ValueError('Frozen SAT contract changed')
 if hashlib.sha256(AUDIT.read_bytes()).hexdigest()!=b['source_audit_sha256']:raise ValueError('Source audit changed')
 for ss in b['source_snapshots'].values():
  if hashlib.sha256((ROOT/ss['path']).read_bytes()).hexdigest()!=ss['sha256']:raise ValueError('Source snapshot changed')
 if b['bt_fit_ids']!=qualified().id.tolist() or b['models']['Bocconi']['bt_curve_parameters'][1]<0:raise ValueError('BT fit identity')
 return b

def describe(m,score,gpa,program,session,diagnostics,bundle):
 sat=bundle['models']['SAT'];out=base.prior.describe(sat,score if m['kind']=='SAT' else 1600,gpa,program,session,diagnostics,bundle)
 out.update(model_label='V14 · krzywa BT ze źródeł',model_version=VERSION,experimental_probability=predict(m,score,gpa,program,session))
 out['coherence']['sat_points_per_bocconi_point']=None
 out['coherence']['note']='SAT pozostaje identyczny z V9 (20SAT/0,1GPA).50BT=1600SAT dla tego samego GPA/kierunku/tury jest założeniem spójności. Nie stosujemy17BT=1040SAT ani stałego przelicznika testów.'
 out['validation_note']='BT dopasowano do49źródłowo sprawdzonych deklaracji wyników, bez nieustalonego wyboru między SAT i BT. Nie są to decyzje zweryfikowane przez uczelnię. Ocena zewnętrzna ponownie wybiera regularyzację tylko BT; baza SAT/progi znały te dane. Nie jest to niezależna walidacja przyszłego naboru.'
 if m['kind']=='SAT':out['formula']['fit_note']='SAT i stałe20SAT/0,1GPA bez zmian. KorektaV14 dotyczy wyłącznieBT.';return out
 a,b=m['bt_curve_parameters'];maximum=float(np.clip(base.prior.raw_probability(sat,1600,gpa,'BIEM',session),0,1));d=m['thresholds'][program];ff=float(factor([a,b],score));out['tradeoff_per_gpa']=None
 out['formula']=dict(kind='bt_learned',raw_percent=100*float(raw_probability(m,score,gpa,program,session)),max_test_numerator_percent=100*maximum,curve_a=a,curve_b=b,bt_factor=ff,programme_threshold=d,training_rows=m['fit_rows'],fit_note='p=(M×sigmoid(a+b×(BT−40)/10)/sigmoid(a+b)−d)/(1−d), ograniczone0–100%.M to ograniczony wynik BIEM dla1600SAT przy tym GPA/turze.40 jest wyłącznie środkiem zapisu współczynników, nie kotwicą wyniku. a i b uczone na jawnych deklaracjach przyjęcia/odmowy. Poniżej26,4 i powyżej46,4BT: ekstrapolacja bez potwierdzonych przypadków.')
 def threshold(p):
  f=lambda s:predict(m,s,9,p,session)-.5
  return float(brentq(f,17,50)) if f(17)*f(50)<=0 else None
 ref=threshold('BIEM');bp=predict(m,score,gpa,'BIEM',session)
 out['programme_difficulty']=[dict(row,offset_percent=100*(predict(m,score,gpa,row['code'],session)-bp),threshold_50=threshold(row['code']),threshold_gap=None if threshold(row['code']) is None or ref is None else threshold(row['code'])-ref) for row in out['programme_difficulty']]
 step=.0001;out['coherence']['gpa_percent_per_point']=100*(float(raw_probability(m,score,min(10,gpa+step),program,session))-float(raw_probability(m,score,max(6,gpa-step),program,session)))/(min(10,gpa+step)-max(6,gpa-step));out['coherence']['winter_minus_early_pp']=100*(float(raw_probability(m,score,gpa,program,'Winter'))-float(raw_probability(m,score,gpa,program,'Early')))
 return out
if __name__=='__main__':export()
