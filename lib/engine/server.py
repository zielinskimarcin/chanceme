"""Local-only algorithm review. Frozen model loading, no runtime training."""
from __future__ import annotations
import argparse
import json
import math
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from learned_bt_chances import ROOT, VERSION, KINDS, datasets, load, predict, raw_probability, describe
from programme_thresholds import load as load_programmes, predict as predict_programmes
from model_v8 import load as load_v8, predict as predict_v8
from model_v7 import load as load_v7, predict as predict_v7
from model_v6 import load as load_v6, predict as predict_v6
from model_v5 import load as load_v5, predict as predict_v5
from model import load as load_v4, predict as predict_v4
from legacy.programs import PROGRAM_CONTRACTS, AI_PROGRAM

app=Flask(__name__,static_folder=None)
app.config['MAX_CONTENT_LENGTH']=16*1024
from release_loader import load_release_snapshot
ENGINE_ERROR = None
try:
    _release = load_release_snapshot(ROOT)
    BUNDLE = _release['bundles']['BUNDLE']
    V5, V6, V7, V8, PREVIOUS_PROGRAMMES = [_release['bundles'][name] for name in
        ('V5', 'V6', 'V7', 'V8', 'PREVIOUS_PROGRAMMES')]
    FRAMES = _release['frames']
    BT_STATUS = _release['bt_status']
    DATA_OVERLAY = {key: [None] * count for key, count in _release['overlay_counts'].items()}
except (OSError, ValueError, KeyError, RuntimeError):
    ENGINE_ERROR = 'The frozen model failed its integrity check.'
    BUNDLE = FRAMES = V5 = V6 = V7 = V8 = PREVIOUS_PROGRAMMES = None

SOURCES=[
 {'title':'Bocconi: zasady, wagi, programy i wyjątki','url':'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/admissions'},
 {'title':'Bocconi: SAT, ACT i wewnętrzne przeliczenia','url':'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/sat-and-act'},
 {'title':'Bocconi: test online','url':'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/online-bocconi-test'},
 {'title':'Bocconi: kategorie kandydatów','url':'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/categories-applicants'},
 {'title':'Bocconi: ranking i preferencje','url':'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/results-and-enrollment'},
 {'title':'Bocconi: właściwe lata i GPA','url':'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/gpa-and-high-school-curriculum'}]
DEFAULT={'test_type':'SAT','score':1450,'gpa':9.,'course':'Management','session':'Winter',
         'cycle':'27-28','applicant_category':'international','gpa_confirmed':True}
SCENARIOS=[
 ('balanced','1450 / GPA 9',{}),
 ('gpa_regression_low','1040 / GPA 9,045',{'score':1040,'gpa':9.045}),
 ('gpa_regression_high','1600 / GPA 8,3',{'score':1600,'gpa':8.3}),
 ('early_management','1510 / GPA 8 · Management Early',{'score':1510,'gpa':8.,'session':'Early'}),
 ('early_finance','1510 / GPA 8 · Finance Early',{'score':1510,'gpa':8.,'session':'Early','course':'Finance'}),
 ('winter_mid','1510 / GPA 8,5 · Winter',{'score':1510,'gpa':8.5}),
 ('high_test_average_gpa','1600 / GPA 9',{'score':1600,'gpa':9}),
 ('high_test_low_gpa','1600 / GPA 7',{'score':1600,'gpa':7}),
 ('maximum','1600 / GPA 10',{'score':1600,'gpa':10}),
 ('near_maximum','1590 / GPA 9,045',{'score':1590,'gpa':9.045}),
 ('high_test_low_gpa','1560 / GPA 8,1',{'score':1560,'gpa':8.1}),
 ('low_test_perfect_gpa','1360 / GPA 10',{'score':1360,'gpa':10}),
 ('weak','1200 / GPA 7',{'score':1200,'gpa':7}),
 ('test_maximum','Test 50 / GPA 10',{'test_type':'Bocconi','score':50,'gpa':10}),
 ('below_minimum','Poniżej minimum',{'score':1030,'gpa':10}),
 ('no_context','AI / test / brak danych',{'test_type':'Bocconi','score':40,'gpa':9.,'course':AI_PROGRAM}),
 ('invalid','Błędna skala GPA',{'gpa':90}),
 ('extreme','1600 / GPA 1',{'score':1600,'gpa':1})]


def empty(status,headline,explanation,**extra):
    return dict(status=status,probability=None,display_percent=None,headline=headline,
                explanation=explanation,next_step='Sprawdź dane wejściowe i spróbuj ponownie.',
                endpoint_note=None,evidence=None,diagnostics=None,eligibility={'status':'unknown','notes':[]},
                warnings=[],sources=SOURCES,**extra)


def number(value,label):
    if isinstance(value,bool):raise ValueError(label+': wpisz liczbę.')
    try:n=float(value)
    except (TypeError,ValueError):raise ValueError(label+': wpisz liczbę.')
    if not math.isfinite(n):raise ValueError(label+': wpisz skończoną liczbę.')
    return n


def evaluate(payload):
    if not isinstance(payload,dict):raise ValueError('Wymagany jest obiekt danych.')
    if {'cycle','applicant_category','test_type','course','session','score','gpa'}-payload.keys():
        raise ValueError('Brakuje wymaganych danych profilu.')
    if payload.get('cycle')!='27-28':
        return empty('unsupported','Nieobsługiwany rocznik','Ten eksperyment opisuje tylko kontekst 2027/28.')
    if payload.get('applicant_category')!='international':
        return empty('unsupported','Wybierz właściwą kategorię','Model nie jest sprawdzony dla innej ścieżki niż international. Kategoria zależy od szkoły, nie samego paszportu.')
    kind=payload.get('test_type');course=payload.get('course');session=payload.get('session')
    if not isinstance(kind,str) or kind not in KINDS:raise ValueError('Wybierz SAT lub standardowy Bocconi Test.')
    if not isinstance(course,str) or course not in PROGRAM_CONTRACTS:raise ValueError('Ten kierunek nie ma sprawdzonego historycznego odpowiednika w eksperymencie.')
    if not isinstance(session,str) or session not in ('Early','Winter'):raise ValueError('International 2027/28: wybierz Early albo Winter.')
    if 'eligible_application' in payload and not isinstance(payload['eligible_application'],bool):raise ValueError('Kwalifikowalność musi być potwierdzona jako tak/nie.')
    score=number(payload.get('score'),'Wynik testu');gpa=number(payload.get('gpa'),'GPA')
    if not 6<=gpa<=10:raise ValueError('GPA z oficjalnego kalkulatora Bocconi jest w skali 6–10. Sprawdź przeliczenie.')
    if kind=='SAT' and (not 400<=score<=1600 or score%10!=0):raise ValueError('SAT musi być od 400 do 1600, co 10 punktów.')
    if kind=='Bocconi' and not -11.17-1e-6<=score<=50:raise ValueError('Bocconi Test: niemożliwy wynik całkowity.')
    if payload.get('gpa_confirmed') is not True:
        return empty('unsupported','Najpierw potwierdź GPA','Użyj przeliczenia Bocconi dla trzeciego i drugiego od końca roku szkoły. Lokalne 90/100 nie oznacza automatycznie 9/10.')
    excluded=[];unknown=[];notes=[]
    if kind=='SAT':
        if score<1040:excluded.append('SAT poniżej oficjalnego minimum 1040.')
        if course==AI_PROGRAM and score<1120:
            excluded.append('AI wymaga 600 SAT Math i 520 Reading and Writing, czyli co najmniej 1120 łącznie.')
        reading=payload.get('sat_reading');math_score=payload.get('sat_math')
        supplied=[v not in (None,'') for v in (reading,math_score)]
        if any(supplied) and not all(supplied):raise ValueError('Podaj obie sekcje SAT albo pozostaw obie puste.')
        if all(supplied):
            reading=number(reading,'SAT Reading and Writing');math_score=number(math_score,'SAT Math')
            if any(not 200<=v<=800 or v%10!=0 for v in (reading,math_score)):raise ValueError('Sekcje SAT: 200–800, co 10 punktów.')
            if reading+math_score!=score:raise ValueError('Suma sekcji SAT musi być równa wynikowi całkowitemu.')
            if min(reading,math_score)<520:excluded.append('Każda sekcja SAT musi mieć co najmniej 520.')
            if course==AI_PROGRAM and math_score<600:excluded.append('AI wymaga co najmniej 600 SAT Math.')
        else:
            if score>=1320:notes.append('Przy tym totalu obie sekcje muszą mieć co najmniej 520 (każda ma maksymalnie 800).')
            else:unknown.append('Bez obu sekcji nie potwierdzimy minimum 520 w każdej.')
            if course==AI_PROGRAM:
                if score>=1400:notes.append('Ten total implikuje co najmniej 600 SAT Math.')
                else:unknown.append('Bez SAT Math nie potwierdzimy minimum 600 dla AI.')
    else:
        if score<17:excluded.append('Bocconi Test poniżej oficjalnego minimum 17/50.')
        value=payload.get('bocconi_math')
        if value not in (None,''):
            value=number(value,'Bocconi Math')
            if not -4.8-1e-6<=value<=24:raise ValueError('Bocconi Math musi być od −4,8 do 24.')
            # Remaining 26 questions can contribute no more than 26 points.
            if score-value>26+.001 or score-value<(-6.37)-.001:raise ValueError('Bocconi Math nie jest zgodne z wynikiem całkowitym.')
            if course==AI_PROGRAM and value<11:excluded.append('AI wymaga co najmniej 11/24 z matematyki.')
        elif course==AI_PROGRAM:
            if score>=37:notes.append('Pozostałe 26 pytań daje najwyżej 26 punktów; ten total implikuje Math co najmniej 11.')
            else:unknown.append('Bez Bocconi Math nie potwierdzimy minimum 11/24 dla AI.')
    if payload.get('eligible_application') is False:excluded.append('Niepotwierdzona kwalifikowalność aplikacji — sam wynik nie wystarczy.')
    eligibility={'status':'failed' if excluded else 'unverified' if unknown else 'score_minimums_met',
                 'notes':excluded+unknown+notes+['Ocena zakłada poprawną, kompletną aplikację spełniającą pozostałe warunki.']}
    if excluded:
        result=empty('ineligible','Ten wynik nie spełnia minimum',' '.join(excluded))
        result.update(probability=0.,display_percent='0%',eligibility=eligibility,
                      endpoint_note='0% z tym wynikiem w tej procedurze — nie ocena Twojej przyszłej aplikacji.',
                      next_step='Sprawdź możliwość kolejnego testu i aktualne zasady kwalifikacji.')
        return result
    if ENGINE_ERROR:raise RuntimeError(ENGINE_ERROR)
    proxy=PROGRAM_CONTRACTS[course]['historical_proxy']
    frame,screening=FRAMES[kind]
    fit_total=sum(len(f) for f,_ in FRAMES.values())
    quarantined_total=sum(meta['quality_quarantined'] for _,meta in FRAMES.values())
    context=frame[(frame.Course==proxy)&(frame.Session==session)].copy()
    fitted=BUNDLE['models'][kind]
    p=predict(fitted,score,gpa,proxy,session)
    raw=float(raw_probability(fitted,score,gpa,proxy,session))
    window=50 if kind=='SAT' else 4
    near=context[(context.score-score).abs()<=window]
    near=near[(near.GPA-gpa).abs()<=.5]
    def counts(part):return {'total':len(part),'admitted':int(part.outcome.sum()),'rejected':len(part)-int(part.outcome.sum())}
    recent_context=context[context.Academic_Year=='26-27']
    recent_near=near[near.Academic_Year=='26-27']
    recent_ranges=None if recent_context.empty else {'score_min':float(recent_context.score.min()),'score_max':float(recent_context.score.max()),'gpa_min':float(recent_context.GPA.min()),'gpa_max':float(recent_context.GPA.max())}
    context['distance']=((context.score-score)/window)**2+((context.GPA-gpa)/.5)**2
    records=[];by_year=[]
    for year in ('24-25','25-26','26-27'):
        part=context[context.Academic_Year==year];nearpart=near[near.Academic_Year==year]
        by_year.append(dict(counts(part),cycle=year,nearby_total=len(nearpart),nearby_admitted=int(nearpart.outcome.sum()),nearby_rejected=len(nearpart)-int(nearpart.outcome.sum())))
        for r in part.sort_values(['distance','id']).head(4).itertuples():
            records.append({'id':r.id,'score':float(r.score),'gpa':float(r.GPA),'course':proxy,
                            'outcome':'admitted' if r.outcome else 'rejected','cycle':r.Academic_Year,
                            'nearby':abs(r.score-score)<=window and abs(r.GPA-gpa)<=.5})
    extrapolated=context.empty or score<context.score.min() or score>context.score.max() or gpa<context.GPA.min() or gpa>context.GPA.max()
    support='ekstrapolacja poza lokalny zakres' if extrapolated else 'mała próba w pobliżu' if len(near)<10 else 'historyczne porównania w pobliżu'
    if p>=.85:
        headline='Bardzo mocny profil w tym modelu'
        explanation='Profil wypada wysoko względem modelowanych historycznych wyników. Sprawdź niżej, z jakich raportów pochodzi porównanie.'
    elif p>=.65:
        headline='Masz mocne podstawy, żeby aplikować'
        explanation='Model ocenia profil korzystnie. To estymata z historycznych zgłoszeń, a wynik zależy też od konkurencji w Twoim naborze.'
    elif p>=.35:
        headline='Masz szansę, ale wynik jest otwarty'
        explanation='To konkurencyjny zakres. Przygotuj aplikację i rozsądny wybór kierunków; wynik nie przesądza o przyjęciu.'
    elif p>=.1:
        headline='Małe szanse w tym modelu — aplikacja jest możliwa'
        explanation='Profil wypada niżej względem historycznych zgłoszeń. Sprawdź porównania i rozważ poprawę testu oraz alternatywne kierunki.'
    else:
        headline='Bardzo trudny profil w tym modelu'
        explanation='Model ocenia profil nisko. Ta liczba nie stanowi decyzji uczelni ani dowodu, że przyjęcie jest niemożliwe.'
    if score>=KINDS[kind][3]-(10 if kind=='SAT' else .5):
        next_step='Test jest już blisko maksimum. Sprawdź GPA, preferencje i formalności; kolejny test może dać bardzo niewiele.'
    else:next_step='Porównaj wyższy wynik testu oraz kierunki zgodne z Twoimi preferencjami. Zachowaj plan B.'
    endpoint_note=None
    if p<=0 or p>=1:
        endpoint_note='Koniec skali modelu; nie jest to pewna odmowa ani gwarancja przyjęcia.'
    maximum=score==KINDS[kind][3] and gpa==10
    if maximum and p>=1-1e-7:
        endpoint_note='100% w warunkowym modelu rankingu: kompletna kwalifikująca aplikacja, próg poniżej maksimum i brak nadmiaru remisów na maksimum. To założenie modelu, nie gwarancja Bocconi.'
    t=fitted['parameters'];ps=fitted['programs'];rs=fitted['rounds']
    slope=t[1]/fitted['scale'];programme_offset=t[2+ps.index(proxy)];round_offset=t[2+len(ps)+rs.index(session)]
    capacities={'BIEM':(6,690),'BIEF':(4,460),'BEMACC':(2,230),'BEMACS':(2,230),'BAI':(1,115),'BIG':(1,115)}
    names={c['historical_proxy']:name for name,c in PROGRAM_CONTRACTS.items()}
    programme_difficulty=[{'code':pr,'name':names[pr],'offset_percent':100*(t[2+ps.index(pr)]-t[2+ps.index('BIEM')]),
                           'classes':capacities[pr][0],'seats':capacities[pr][1],'capacity_year':'2027/28',
                           'threshold_50':fitted['centre']+(.5-t[0]-t[2+ps.index(pr)]-round_offset)/slope,
                           'threshold_gap':(t[2+ps.index('BIEM')]-t[2+ps.index(pr)])/slope} for pr in ps]
    warnings=['To historyczny model niezweryfikowanych deklaracji, nie potwierdzone szanse populacyjne 2027/28.',
              'Stałe20SAT za0,1GPA jest świadomym założeniem uproszczonego modelu. Porównanie innych stałych jest w raporcie; nie jest to oficjalny przelicznik Bocconi.',
              'GPA w źródłach bywało przeliczane proporcjonalnie z innych skal, a kategorie kandydatów są wymieszane. Nie znamy poprawnego GPA i preferencji wszystkich raportów.',
              'accept/reject oznacza zachowaną deklarację. Bez kolejności preferencji nie każda odmowa dowodzi niespełnienia progu kierunku.',
              'Wszystkie trzy roczniki uczą jeden historyczny poziom; nie stosujemy globalnej korekty najnowszym rocznikiem. Zmiana konkurencji pozostaje ryzykiem.',
              'Model ma niższy próg BIEM niż BIEF oraz wyższe progi CS/AI. To zadana kolejność z dopasowanymi dodatnimi odstępami, nie opublikowana hierarchia uczelni. CS i AI nie mają narzuconej kolejności między sobą. BIG pozostaje bez narzuconej pozycji.',
              'Liczba miejsc nie wyznacza szans bez liczby i wyników kandydatów. Historyczne programy są przybliżeniem obecnych nazw.']
    if kind=='Bocconi':warnings.append('Bocconi Test: mała próba. Wspólna skala usuwa sprzeczne założenia poprzednich osobnych modeli; nie dowodzi dokładności szans ani oficjalnej równoważności testów.')
    if len(near)<4 or len(recent_near)==0:warnings.append('Mało bliskich raportów, zwłaszcza nowych. Procent nadal pokazujemy, ale ma słabe lokalne potwierdzenie.')
    if extrapolated:warnings.append('Profil poza lokalnym zakresem danych. Procent jest ekstrapolacją stałego wzoru.')
    warnings.extend(unknown)
    audit={'confirmed_removed':3,'conflicting_pairs':2,'category_verified':False,'label_corrections_total':len(DATA_OVERLAY['label_corrections']),'label_corrections_in_fit':sum(meta['label_corrected'] for _,meta in FRAMES.values()),'rescaling_quarantined_total':len(DATA_OVERLAY['quarantine_unverified_rescaling']),'rescaling_quarantined_in_scope':quarantined_total,'nationality_flags':47,
           'provenance':'Brak źródła/aplikantID/preferencji/skaliGPA w CSV. Audyt odzyskanych arkuszy wskazał wartości zgodne z proporcjonalnym przeskalowaniem GPA oraz odpowiedzi Italian. Nie jest to potwierdzona kategoria zależna od szkoły ani weryfikacja całej bazy.'}
    comparison={'previous_programmes_probability':predict_programmes(PREVIOUS_PROGRAMMES['models'][kind],score,gpa,proxy,session)}
    for label,bundle,fn in [('v5',V5,predict_v5),('v6',V6,predict_v6),('v7',V7,predict_v7),('v8',V8,predict_v8)]:
        comparison[label+'_probability']=fn(bundle['models'][kind],score,gpa,proxy,session)
    result={'status':'estimated','probability':p,'display_percent':'100%' if maximum and p>=1-1e-7 else '>99%' if p>=.995 else '<1%' if p<.005 else f'{round(p*100)}%',
            'headline':headline,'explanation':explanation,'next_step':next_step,'endpoint_note':endpoint_note,
            'eligibility':eligibility,'sources':SOURCES,'warnings':warnings,
            'evidence':{'context':counts(context),'nearby':dict(counts(near),test_window=window,gpa_window=.5),
                        'data_funnel':{'raw':663,'screened':660,'scope':570,'quality_fit_total':fit_total,'quality_quarantined':quarantined_total,'fit_rows':fit_total,'test_rows':len(frame),'context_rows':len(context),'nearby_rows':len(near)},
                        'scope_exclusions':{'BESS':50,'BGL':22,'CLEACC':5,'CLEAM':10,'CLMG':3},'data_audit':audit,
                        'by_year':by_year,'recent_context':dict(counts(recent_context),cycle='26-27'),'recent_nearby':dict(counts(recent_near),cycle='26-27'),
                        'recent_bounds':recent_ranges,'records':records,'support':support,'proxy_notice':PROGRAM_CONTRACTS[course]['mapping_notice'],
                        'cycles':sorted(context.Academic_Year.unique())},
            'diagnostics':{'model_label':'V9 · wspólna skala','model_version':VERSION,'experimental_probability':p,'tradeoff_per_gpa':fitted['k'],
                           'formula':{'raw_percent':raw*100,'intercept_percent':t[0]*100,'slope_percent_per_test_point':slope*100,
                                      'gpa_percent_per_point':slope*fitted['k']*100,'programme_offset_percent':programme_offset*100,
                                      'round_offset_percent':round_offset*100,'academic_centre':fitted['centre'],'gpa_centre':9,
                                      'fit_note':'Jedna wspólna liniowa skala dla SAT i Bocconi Test: taki sam wpływ GPA, kierunków i tury. Stałe20 SAT za0,1GPA pozostaje. Nachylenie testu Bocconi dopasowano wspólnie do danych. Wyrównanie maksimów obu testów jest jawnym założeniem spójności, nie oficjalną tabelą konwersji. Winter nie trudniejsze od Early jest ograniczeniem kierunku; wielkość efektu jest dopasowana, może wynosić zero. Warunkowe końce całej prostej i ograniczenie0–100% nie gwarantują przyjęcia.'},
                           'coherence':{'sat_points_per_bocconi_point':(BUNDLE['models']['Bocconi']['parameters'][1]/BUNDLE['models']['Bocconi']['scale'])/(BUNDLE['models']['SAT']['parameters'][1]/BUNDLE['models']['SAT']['scale']),
                                        'gpa_percent_per_point':100*slope*fitted['k'],
                                        'winter_minus_early_pp':100*(t[2+len(ps)+rs.index('Winter')]-t[2+len(ps)+rs.index('Early')]),
                                        'note':'Wspólna skala obu testów, GPA, kierunków i tur. Maksima zrównane jako założenie; przelicznik nie jest oficjalną konwersją Bocconi. Spójność nie zastępuje walidacji procentów.'},
                           'programme_difficulty':programme_difficulty,
                           'programme_threshold_context':{'gpa':9,'session':session,'units':'SAT' if kind=='SAT' else 'Bocconi Test'},
                           'programme_method':'Kolejność BIEM → BIEF → CS/AI jest jawnym założeniem. Dodatnie odstępy dopasowano z regularyzacją; jej siłę wybrano na wewnętrznych częściach treningu. Skala odniesienia pochodzi z porównań BIEM/BIEF. CS/AI oraz Bocconi Test korzystają z tego wspólnego założenia; ich odstępy nie są niezależnie potwierdzonymi progami uczelni.',
                           'programme_evidence':{'matched_BIEM_BIEF':BUNDLE['scale_info']['n'],'BIEM_accept_BIEF_reject':BUNDLE['scale_info']['AR'],'reverse':BUNDLE['scale_info']['RA'],'alpha':fitted['programme_alpha']},
                           'limitations':warnings,
                           'validation_note':'Grupowa ocena mieszanych historycznych roczników. Osobno pokazujemy trudniejszy test przeniesienia na najnowszy rocznik. Nie są to niezależnie potwierdzone szanse przyjęcia.',
                           **comparison}}

    if kind=='Bocconi':
        audit_rows=BUNDLE['bt_fit_ids']
        result['evidence']['bt_training']={'rows':len(audit_rows),'historical_comparison_rows':len(frame),'definition':'Do dopasowania BT: tylko źródłowo sprawdzone deklaracje przyjęcia lub odmowy dla kierunku, bez nieustalonego wyboru SAT/BT. Nie są to decyzje zweryfikowane przez uczelnię.'}
        result['evidence']['data_funnel']['fit_rows']=len(audit_rows)+len(FRAMES['SAT'][0])
        statuses=BT_STATUS
        for record in result['evidence']['records']:
            entry=statuses[record['id']];record['used_for_bt_fit']=record['id'] in audit_rows;record['source_status']=entry['status'];record['source_urls']=[v['url'] for v in entry['sources']]
    result['diagnostics']=describe(fitted,score,gpa,proxy,session,result['diagnostics'],BUNDLE)
    result['warnings']=[v for v in result['warnings'] if not v.startswith('Bocconi Test: mała próba. Wspólna skala')]
    result['warnings'].append('Poziom SAT V9 jest zachowany na polecenie użytkownika; nie jest potwierdzoną kalibracją przyszłego naboru. Progi i skrajności pozostają zależne od założeń.')
    result['diagnostics']['limitations']=result['warnings']
    return result


@app.after_request
def headers(response):
    response.headers['Cache-Control']='no-store'
    response.headers['X-Robots-Tag']='noindex, nofollow'
    response.headers['X-Content-Type-Options']='nosniff'
    return response


@app.get('/api/options')
def options():
    return jsonify({'programs':[{'id':n,'name':n} for n in PROGRAM_CONTRACTS],
                    'defaults':DEFAULT,'scenarios':[{'id':i,'label':label,'input':dict(DEFAULT,**changes)} for i,label,changes in SCENARIOS]})


@app.post('/api/estimate')
def estimate():
    try:return jsonify(evaluate(request.get_json(silent=True)))
    except ValueError as exc:return jsonify(empty('invalid','Sprawdź dane',str(exc))),400
    except RuntimeError:return jsonify(empty('unsupported','Model jest niedostępny','Artefakt nie przeszedł kontroli. Nie pokazujemy zastępczego procentu.')),503


@app.get('/api/validation')
def validation():
    return send_from_directory(ROOT/'research','validation-learned-bt.json')


@app.get('/')
def home():return send_from_directory(ROOT/'dist','index.html')


@app.get('/bocconi-gpa-calculator/')
def gpa_calculator():return send_from_directory(ROOT/'dist','gpa.html')


@app.get('/<path:name>')
def assets(name):
    if name not in ('app.js','style.css','gpa.js','gpa-core.js','gpa.css'):return jsonify({'error':'Not found'}),404
    return send_from_directory(ROOT/'dist',name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18780)
    args=parser.parse_args()
    print(f'Chance Me coherent local experiment: http://127.0.0.1:{args.port}/',flush=True)
    app.run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False)
