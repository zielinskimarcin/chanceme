"""Portable production API for the approved Chance Me interface and frozen V14."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
from flask import Flask, jsonify, request, Response, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT / 'lib/engine'
sys.path.insert(0, str(ENGINE_ROOT))
spec = importlib.util.spec_from_file_location('chanceme_frozen_service', ENGINE_ROOT / 'server.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
BT_STATUS = engine.BT_STATUS if not engine.ENGINE_ERROR else {}
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024

@app.after_request
def response_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Robots-Tag'] = 'noindex, nofollow'
    elif os.environ.get('VERCEL_ENV') != 'production':
        response.headers['X-Robots-Tag'] = 'noindex, nofollow'
    return response

@app.get('/api/health')
def health():
    return jsonify(status='unavailable' if engine.ENGINE_ERROR else 'ok', version=engine.VERSION,
        reports={} if engine.ENGINE_ERROR else {kind:len(frame) for kind,(frame,_) in engine.FRAMES.items()}), 503 if engine.ENGINE_ERROR else 200

@app.get('/api/options')
def options():
    return engine.options()

@app.post('/api/estimate')
def estimate():
    try:
        return jsonify(engine.evaluate(request.get_json(silent=True)))
    except (ValueError, TypeError, KeyError):
        return jsonify(status='invalid', probability=None, message='Check your score, GPA, programme and round.'),400
    except RuntimeError:
        return jsonify(status='unavailable', probability=None, message='The model is temporarily unavailable.'),503

def comparison_data(payload):
    if not isinstance(payload, dict):
        raise ValueError("A profile is required.")
    kind = payload.get("test_type")
    course = payload.get("course")
    session = payload.get("session")
    if session not in ("Early", "Winter", "All"):
        raise ValueError("Choose Early, Winter or both rounds.")
    checked = dict(payload, session="Winter" if session == "All" else session)
    result = engine.evaluate(checked)
    if result["status"] not in ("estimated", "ineligible"):
        raise ValueError("This profile is outside the supported scope.")
    checked["score"] = engine.number(checked["score"], "Score")
    checked["gpa"] = engine.number(checked["gpa"], "GPA")
    proxy = engine.PROGRAM_CONTRACTS[course]["historical_proxy"]
    frame = engine.FRAMES[kind][0]
    part = frame[frame.Course == proxy]
    if session != "All":
        part = part[part.Session == session]
    window = 50 if kind == "SAT" else 4
    records = []
    for r in part.sort_values("id").itertuples():
        record = dict(id=r.id, score=float(r.score), gpa=float(r.GPA), course=proxy,
                      session=r.Session, cycle=r.Academic_Year,
                      outcome="admitted" if r.outcome else "rejected",
                      nearby=abs(r.score-checked["score"]) <= window and abs(r.GPA-checked["gpa"]) <= .5)
        if kind == "Bocconi":
            record["used_for_bt_fit"] = r.id in engine.BUNDLE["bt_fit_ids"]
            record["source_status"] = BT_STATUS[r.id]["status"]
        records.append(record)
    def counts(rows):
        return dict(total=len(rows), admitted=sum(r["outcome"] == "admitted" for r in rows),
                    rejected=sum(r["outcome"] == "rejected" for r in rows))
    near = [r for r in records if r["nearby"]]
    years = []
    for cycle in sorted(set(r["cycle"] for r in records)):
        rows = [r for r in records if r["cycle"] == cycle]
        nearby = counts([r for r in rows if r["nearby"]])
        years.append(dict(counts(rows), cycle=cycle, nearby_total=nearby["total"],
                          nearby_admitted=nearby["admitted"], nearby_rejected=nearby["rejected"]))
    return dict(status="ok", records=records, nearby=dict(counts(near), test_window=window, gpa_window=.5),
                by_year=years, context=counts(records), complete=True,
                proxy_notice=engine.PROGRAM_CONTRACTS[course]["mapping_notice"],
                filters=dict(course=course, session=session, test_type=kind))

@app.post('/api/reports')
def reports():
    try:
        return jsonify(comparison_data(request.get_json(silent=True)))
    except (ValueError, TypeError, KeyError):
        return jsonify(status='invalid', message='Check the comparison filters.'),400
    except RuntimeError:
        return jsonify(status='unavailable', message='Reports are unavailable.'),503

@app.post('/api/report.pdf')
def download_pdf():
    try:
        body=request.get_json(silent=True)
        if not isinstance(body,dict) or not isinstance(body.get('profile'),dict):
            raise ValueError('Missing profile')
        profile=dict(body['profile'])
        filters=body.get('filters',{})
        if not isinstance(filters,dict):
            raise ValueError('Invalid filters')
        profile['course']=filters.get('course',profile['course'])
        session=filters.get('session',profile['session'])
        if session in ('Early','Winter'):
            profile['session']=session
        profile['score']=engine.number(profile['score'],'Score')
        profile['gpa']=engine.number(profile['gpa'],'GPA')
        result=engine.evaluate(profile)
        if result['status']!='estimated':
            raise ValueError('No estimate')
        comparison=comparison_data(dict(profile,session=session))
        cycle=filters.get('cycle','')
        if cycle and cycle not in [row['cycle'] for row in comparison['by_year']]:
            raise ValueError('Invalid cycle')
        rows=[row for row in comparison['records'] if not cycle or row['cycle']==cycle]
        comparison['records']=rows
        comparison['nearby'].update(total=sum(r['nearby'] for r in rows),
            admitted=sum(r['nearby'] and r['outcome']=='admitted' for r in rows),
            rejected=sum(r['nearby'] and r['outcome']=='rejected' for r in rows))
        comparison['filters']['cycle']=cycle
        from lib.report_pdf import render
        data=render(dict(profile=profile,estimate=dict(probability=result['probability'],display_percent=result['display_percent']),comparison=comparison))
        if not data.startswith(b'%PDF-'):
            raise RuntimeError('Invalid document')
        return Response(data,mimetype='application/pdf',headers={'Content-Disposition':'attachment; filename="chance-me-report.pdf"'})
    except (ValueError,TypeError,KeyError):
        return jsonify(status='invalid',message='Calculate a result before downloading.'),400
    except (RuntimeError,OSError):
        return jsonify(status='unavailable',message='The PDF could not be created. Please try again.'),503

@app.errorhandler(413)
def too_large(error):
    return jsonify(status='invalid',message='The request is too large.'),413

# Local development mirrors public/ only; Vercel serves these files from its CDN.
@app.get('/')
@app.get('/<path:name>')
def public_files(name=''):
    public=(ROOT/'public').resolve()
    target=(public/name).resolve()
    if target!=public and public not in target.parents:
        return 'Not found',404
    if target.is_dir():
        target=target/'index.html'
    if not target.is_file():
        return send_from_directory(public,'404.html'),404
    return send_from_directory(public,str(target.relative_to(public)))

if __name__=='__main__':
    app.run(host='127.0.0.1',port=int(os.environ.get('PORT','18782')),threaded=True,debug=False)
