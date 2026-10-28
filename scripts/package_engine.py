"""Package the approved frozen engine without private research or runtime fitting.

Run once with --source pointing to the reviewed V14 lab. Never changes that lab.
The public package contains numeric model state and pseudonymous report columns.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
root = Path(__file__).resolve().parents[1]
dest = root / 'lib/engine'
sys.path[:0] = [str(source), str(source / '.deps')]
spec = importlib.util.spec_from_file_location('approved_v14_export', source / 'server.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
if engine.ENGINE_ERROR:
    raise RuntimeError(engine.ENGINE_ERROR)
columns = ['id', 'score', 'GPA', 'Course', 'Session', 'Academic_Year', 'outcome']
audit = json.loads((source / 'research/bt-source-audit-v14.json').read_text())
snapshot = {
    'schema': 1, 'version': engine.VERSION,
    'bundles': {name: getattr(engine, name) for name in
                ['BUNDLE', 'V5', 'V6', 'V7', 'V8', 'PREVIOUS_PROGRAMMES']},
    'frames': {kind: {'records': frame[columns].to_dict('records'), 'screening': screening}
               for kind, (frame, screening) in engine.FRAMES.items()},
    'bt_status': {row['id']: {'status': row['status'],
                   'sources': [{'url': item['url']} for item in row['sources']]}
                  for row in audit['ledger']},
    'overlay_counts': {key: len(engine.DATA_OVERLAY[key]) for key in
                       ['label_corrections', 'quarantine_unverified_rescaling']},
}
raw = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
(dest / 'release').mkdir(parents=True, exist_ok=True)
(dest / 'release/snapshot.json').write_bytes(raw)
modules = ['learned_bt_chances.py', 'unified_chances.py', 'targeted_thresholds.py',
           'coherent_thresholds.py', 'evidence_thresholds.py', 'programme_thresholds.py',
           'model.py', 'model_v5.py', 'model_v6.py', 'model_v7.py', 'model_v8.py',
           'pooled_v7.py', 'calibration_v7.py', 'legacy/data.py', 'legacy/programs.py',
           'legacy/experimental.py']
for name in modules:
    (dest / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / name, dest / name)
text = (source / 'server.py').read_text()
text = text.replace('from legacy.frozen import load_bundle\n', '')
start = text.index('ENGINE_ERROR=None')
end = text.index('\nSOURCES=', start)
text = text[:start] + '''from release_loader import load_release_snapshot
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
''' + text[end:]
text = text.replace("statuses={r['id']:r for r in json.loads((ROOT/'research/bt-source-audit-v14.json').read_text())['ledger']}", 'statuses=BT_STATUS')
(dest / 'server.py').write_text(text)
for package in [dest, dest / 'legacy']:
    (package / '__init__.py').touch()
paths = [*modules, 'server.py', 'release/snapshot.json', 'release_loader.py']
manifest = {
    'schema': 1, 'version': engine.VERSION,
    'exported_at': datetime.now(timezone.utc).isoformat(),
    'upstream_server_sha256': hashlib.sha256((source / 'server.py').read_bytes()).hexdigest(),
    'upstream_model_sha256': hashlib.sha256((source / 'artifacts/model-learned-bt.json').read_bytes()).hexdigest(),
    'rows': {kind: len(frame) for kind, (frame, _) in engine.FRAMES.items()},
    'bt_fit_rows': len(engine.BUNDLE['bt_fit_ids']),
    'files': {p: hashlib.sha256((dest / p).read_bytes()).hexdigest() for p in paths},
    'privacy': 'No raw source snapshots, respondent timestamps, free text or contact identifiers.',
    'contract': 'Frozen coefficients and full estimate response preserved; no fitting at runtime.'
}
(dest / 'release/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Packaged', manifest['version'], manifest['rows'], 'BT fit', manifest['bt_fit_rows'])
