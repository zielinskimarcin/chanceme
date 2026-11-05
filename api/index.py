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
        payload = request.get_json(silent=True)
        result = engine.evaluate(payload)
        # The frozen model is unchanged; restrict individual data at the API boundary.
        if isinstance(result.get('evidence'), dict):
            result['evidence']['records'] = closest_records(result['evidence'].get('records', []), payload)
            result['evidence']['record_limit'] = 10
        return jsonify(result)
    except (ValueError, TypeError, KeyError):
        return jsonify(status='invalid', probability=None, message='Check your score, GPA, programme and round.'),400
    except RuntimeError:
        return jsonify(status='unavailable', probability=None, message='The model is temporarily unavailable.'),503

def closest_records(records, profile):
    window = 50 if profile['test_type'] == 'SAT' else 4
    score = engine.number(profile['score'], 'Score')
    gpa = engine.number(profile['gpa'], 'GPA')
    nearby = [r for r in records if abs(r['score'] - score) <= window and abs(r['gpa'] - gpa) <= .5]
    return sorted(nearby, key=lambda r: (
        abs(r['score'] - score) / window + abs(r['gpa'] - gpa) / .5,
        str(r['id'])))[:10]

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
    return dict(status="ok", records=closest_records(records, checked), nearby=dict(counts(near), test_window=window, gpa_window=.5),
                by_year=years, context=counts(records), complete=False, record_limit=10,
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
    # Do not expose a direct download while email delivery is unavailable.
    request.get_json(silent=True)  # Preserve the request-size boundary.
    return jsonify(status='unavailable', delivery='email',
                   message='PDF reports are currently unavailable. Future reports will require an email address and be delivered by email.'), 503

def report_pdf(profile, filters):
    if not isinstance(profile, dict) or not isinstance(filters, dict):
        raise ValueError('Missing profile')
    selected = dict(profile, course=filters.get('course', profile.get('course')),
                    session=filters.get('session', profile.get('session')))
    result = engine.evaluate(selected)
    if result['status'] != 'estimated':
        raise ValueError('No estimate')
    # Retain only validated, needed fields. The request cannot set a PDF recipient or template.
    clean = {key: selected[key] for key in ('test_type', 'course', 'session', 'cycle', 'applicant_category', 'gpa_confirmed')}
    clean.update(score=engine.number(selected['score'], 'Score'),
                 gpa=engine.number(selected['gpa'], 'GPA'),
                 gpa_estimated=selected.get('gpa_estimated') is True)
    comparison = comparison_data(clean)
    cycle = filters.get('cycle', '')
    if not isinstance(cycle, str) or (cycle and cycle not in [r['cycle'] for r in comparison['by_year']]):
        raise ValueError('Invalid cycle')
    comparison['records'] = [r for r in comparison['records'] if not cycle or r['cycle'] == cycle]
    comparison['filters']['cycle'] = cycle
    if cycle:
        counts = next(r for r in comparison['by_year'] if r['cycle'] == cycle)
        comparison['nearby'].update(total=counts['nearby_total'], admitted=counts['nearby_admitted'], rejected=counts['nearby_rejected'])
    from lib.report_pdf import render
    return render(dict(profile=clean, estimate=result, comparison=comparison))

@app.get('/api/report-email/status')
def email_status():
    from lib.report_email import configured
    return jsonify(enabled=configured())

def email_origin_allowed():
    origin = request.headers.get('Origin', '')
    if origin in ('https://chance-me.com', 'https://www.chance-me.com'):
        return True
    # Only the exact Vercel-assigned preview hostname or loopback development origin.
    if os.environ.get('VERCEL_ENV') != 'production':
        preview = os.environ.get('VERCEL_URL')
        return origin in {'http://127.0.0.1:18786', 'http://127.0.0.1:18782'} or bool(preview and origin == 'https://' + preview)
    return False

@app.post('/api/report-email')
@app.post('/api/report-email/verify')
def email_report():
    from lib import report_email as delivery
    if not email_origin_allowed():
        return jsonify(status='invalid', message='Open this form on Chance Me.'), 403
    if not delivery.configured():
        return jsonify(status='unavailable', message='Email reports are temporarily unavailable.'), 503
    try:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError('Invalid request')
        if request.path.endswith('/verify'):
            delivery.verify_report(body.get('request_id'), body.get('code'))
            return jsonify(status='accepted', message='Your report has been submitted for email delivery. Check your inbox and spam folder.')
        profile, filters = body.get('profile'), body.get('filters', {})
        # Validate and build from server data only. The browser never supplies PDF bytes.
        pdf = report_pdf(profile, filters)
        ip = request.headers.get('X-Vercel-Forwarded-For', '') if os.environ.get('VERCEL') else request.remote_addr
        ident = delivery.request_report(body.get('email'), ip, lambda: pdf)
        return jsonify(status='confirmation_required', request_id=ident, expires_in=delivery.TTL,
                       message='We emailed you a confirmation code. Enter it below within 15 minutes.')
    except delivery.Limited:
        return jsonify(status='limited', message='Please wait before trying again. Report requests are limited.'), 429
    except (delivery.Invalid, ValueError, TypeError, KeyError):
        return jsonify(status='invalid', message='Check the email, profile or confirmation code. Expired codes require a new request.'), 400
    except (delivery.Unavailable, RuntimeError, OSError):
        return jsonify(status='unavailable', message='Email could not be submitted. Please wait one minute and try again.'), 503

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
