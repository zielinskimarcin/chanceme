"""V12: frozen SAT levels, programme thresholds, shared maximum-test ceiling.

BT raw numerator = SAT1600 BIEM raw probability * exp(-rate*(50-BT)/10).
The same programme normalization applies to both tests. The rate is the only
new learned parameter. No separate BT GPA weight or cohort adjustment.
"""
from __future__ import annotations
import copy,hashlib,json
import numpy as np
import targeted_thresholds as prior
from scipy.optimize import minimize_scalar
ROOT=prior.ROOT;KINDS=prior.KINDS;PS=prior.PS;RS=prior.RS;datasets=prior.datasets
VERSION='unified-chances-fixed-sat-v12-2026-10-24'


def components(frame,sat_model):
 return np.array([prior.raw_probability(sat_model,1600,r.GPA,'BIEM',r.Session) for r in frame.itertuples()]),np.array([sat_model['thresholds'][p] for p in frame.Course]),(50-frame.score.to_numpy(float))/10


def fit_rate(frame,sat_model):
 f=frame.reset_index(drop=True);base,d,x=components(f,sat_model);units,_=prior.ev.units(f.assign(test_kind='Bocconi'));y=f.outcome.to_numpy(float)
 def objective(rate):
  p=np.clip((base*np.exp(-rate*x)-d)/(1-d),0,1)
  return prior.unit_loss(y,p,units)
 r=minimize_scalar(objective,bounds=(0.,2.),method='bounded',options={'xatol':1e-11})
 if not r.success or not np.isfinite(r.fun):raise RuntimeError('BT rate fit failed')
 # Explicitly compare the permitted zero-rate boundary.
 rate=min([0.,float(r.x),2.],key=objective)
 if rate<=0:raise RuntimeError('Data do not identify a positive BT slope')
 return dict(rate=rate,objective=float(objective(rate)),rows=len(f),profile_units=len(units))


def rate_predictions(rate,frame,sat_model):
 base,d,x=components(frame,sat_model)
 return np.clip((base*np.exp(-rate*x)-d)/(1-d),0,1)


def export():
 b11=prior.load();sat=b11['models']['SAT'];f=datasets()['Bocconi'][0].reset_index(drop=True)
 assigned=prior.old.base.folds(f,3,'unified-v12-conditional');oof=np.full(len(f),np.nan);proof=[]
 for fold in range(3):
  tr=f.iloc[np.flatnonzero(assigned!=fold)];he=f.iloc[np.flatnonzero(assigned==fold)];fit=fit_rate(tr,sat);oof[assigned==fold]=rate_predictions(fit['rate'],he,sat)
  proof.append(dict(fold=fold,train=tr.id.tolist(),held=he.id.tolist(),rate=fit['rate']))
 fit=fit_rate(f,sat);bundle=copy.deepcopy(b11);bundle.update(version=VERSION,bt_fit=fit,previous_version=prior.VERSION,previous_model_sha256=hashlib.sha256((ROOT/'artifacts/model-targeted.json').read_bytes()).hexdigest())
 # SAT is copied without any numerical or metadata mutation.
 bt=copy.deepcopy(sat);bt.update(kind='Bocconi',bt_rate=fit['rate'],sat_reference=copy.deepcopy(sat),fit_rows=len(f),k=sat['k'])
 bundle['models']['Bocconi']=bt
 bundle['assumptions']=['SAT V11 unchanged; SAT BIEM remains at frozen V9 levels','SAT1600 equals BT50 at every GPA/programme/round: explicit coherence assumption, not official equivalence','BT is a monotone exponential penalty relative to the same maximum-test numerator','Programme thresholds and proxy-profile assumptions inherited unchanged from V11','No separate BT GPA coefficient; no cohort or prevalence correction']
 note='Conditional adaptation: the frozen SAT and programme thresholds have already seen these records. Conditional CV is NOT independent end-to-end or future validation.'
 previous_report=json.loads((ROOT/'research/validation-targeted.json').read_text());evs=[]
 for kind,(ff,_) in datasets().items():
  oldp=np.array([prior.predict(b11['models'][kind],r.score,r.GPA,r.Course,r.Session) for r in ff.itertuples()]);newp=np.array([predict(bundle['models'][kind],r.score,r.GPA,r.Course,r.Session) for r in ff.itertuples()])
  conditional=oldp if kind=='SAT' else oof
  evs.append(dict(kind=kind,shared_rows=len(ff),scope_note=note,conditional_metrics={'v11_fixed':prior.old.base.metrics(ff.outcome,oldp),'candidate_conditional_cv':prior.old.base.metrics(ff.outcome,conditional),'candidate_fit':prior.old.base.metrics(ff.outcome,newp)}))
 report=dict(version=VERSION,label='V12 · spójny procent szans',scope_note=note,evaluations=evs,bt_fold_proof=proof,bt_fit=fit,previous_report=previous_report['version'])
 raw=prior.old.base.canonical(bundle);rr=prior.old.base.canonical(report)
 (ROOT/'artifacts/model-unified.json').write_bytes(raw);(ROOT/'research/validation-unified.json').write_bytes(rr);(ROOT/'artifacts/manifest-unified.json').write_bytes(prior.old.base.canonical(dict(version=VERSION,sha256=hashlib.sha256(raw).hexdigest(),report_sha256=hashlib.sha256(rr).hexdigest())))
 print('V12 frozen',fit,flush=True)
 for e in evs:print(e['kind'],{k:(v['brier'],v['log_loss']) for k,v in e['conditional_metrics'].items()},flush=True)


def raw_probability(m,score,gpa,program,session):
 if m['kind']=='SAT':return prior.raw_probability(m,score,gpa,program,session)
 d=m['thresholds'][program];base=prior.raw_probability(m['sat_reference'],1600,gpa,'BIEM',session)
 return (base*np.exp(-m['bt_rate']*(50-np.asarray(score,float))/10)-d)/(1-d)


def predict(m,score,gpa,program,session):
 p=np.clip(raw_probability(m,score,gpa,program,session),0,1)
 return float(p) if np.ndim(p)==0 else p


def load():
 before=prior.load();raw=(ROOT/'artifacts/model-unified.json').read_bytes();man=json.loads((ROOT/'artifacts/manifest-unified.json').read_text());b=json.loads(raw)
 if b['version']!=VERSION or man['version']!=VERSION or hashlib.sha256(raw).hexdigest()!=man['sha256']:raise ValueError('V12 identity')
 if hashlib.sha256((ROOT/'research/validation-unified.json').read_bytes()).hexdigest()!=man['report_sha256']:raise ValueError('V12 report identity')
 if b['previous_model_sha256']!=hashlib.sha256((ROOT/'artifacts/model-targeted.json').read_bytes()).hexdigest():raise ValueError('V12 base identity')
 if b['models']['SAT']!=before['models']['SAT']:raise ValueError('SAT changed')
 bt=b['models']['Bocconi']
 if not 0<bt['bt_rate']<=2 or bt['sat_reference']!=b['models']['SAT'] or bt['thresholds']!=b['models']['SAT']['thresholds']:raise ValueError('BT coherence')
 return b


def describe(m,score,gpa,program,session,base_diagnostics,bundle):
 # Reuse the historical evidence contract, then replace all formula/scale fields.
 out=prior.describe(bundle['models']['SAT'] if m['kind']=='SAT' else m['sat_reference'],1600 if m['kind']=='Bocconi' else score,gpa,program,session,base_diagnostics,bundle)
 out.update(model_label='V12 · spójne szanse',model_version=VERSION,experimental_probability=predict(m,score,gpa,program,session))
 common='SAT i progi kierunków pozostają bez zmian. Dla każdego GPA/kierunku/tury:50BT=1600SAT. To założenie spójności, nie oficjalna konwersja testów. Brak korekty rocznika.'
 out['validation_note']='Procent jest estymacją szans modelu. Dopasowanie BT oceniono warunkowo na3częściach przy zamrożonej bazie; baza znała te rekordy. To nie jest niezależna walidacja przyszłego naboru.'
 if m['kind']=='SAT':
  out['formula']['fit_note']=common+' W SAT pozostaje dokładnie20punktów za0,1GPA.'
 else:
  sat=m['sat_reference'];d=m['thresholds'][program];q=1-d;rate=m['bt_rate'];t=sat['parameters'];sl=t[1]/sat['scale'];gp=sl*sat['k'];ro=t[2+len(PS)+RS.index(session)];start=t[0]+sl*(1600-sat['centre']);factor=float(np.exp(-rate*(50-score)/10))
  out['tradeoff_per_gpa']=None
  out['formula']=dict(kind='bt_factor',raw_percent=float(raw_probability(m,score,gpa,program,session))*100,max_test_numerator_percent=100*(start+gp*(gpa-9)+ro),bt_rate=rate,bt_factor=factor,programme_threshold=d,fit_note=common+' BT: wspólna podstawa GPA pomnożona przez exp(−r×(50−BT)/10), następnie wspólny próg kierunku. Parametr r dopasowany do93rekordów. Nie ma osobnej swobodnej wagi GPA dlaBT. Wartości poza zakresemBT26,4–46,4 są ekstrapolacją.')
  def threshold(pr):
   target=(1+m['thresholds'][pr])/2;maximum=start+ro
   return 50+10/rate*np.log(target/maximum)
  reference=threshold('BIEM');basep=predict(m,score,gpa,'BIEM',session)
  out['programme_difficulty']=[dict(row,offset_percent=100*(predict(m,score,gpa,row['code'],session)-basep),threshold_50=threshold(row['code']),threshold_gap=threshold(row['code'])-reference) for row in out['programme_difficulty']]
  out['coherence']['gpa_percent_per_point']=100*gp*factor/q
  out['coherence']['winter_minus_early_pp']=100*(t[-1]-t[-2])*factor/q
  out['coherence']['sat_points_per_bocconi_point']=None
 out['coherence']['sat_points_per_bocconi_point']=None
 out['coherence']['note']=common+' Wpływ GPA w BT wynika ze wspólnej podstawy i czynnika wyniku testu; nie jest osobno dopasowaną wagą. Poniższe przyrosty dotyczą bieżącego profilu.'
 return out

if __name__=='__main__':export()
