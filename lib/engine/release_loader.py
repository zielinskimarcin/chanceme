"""Fail closed if the reviewed numeric serving package has changed."""
import hashlib
import json
from pathlib import PurePosixPath
import pandas as pd

VERSION = 'source-audited-bt-curve-v14-2026-10-25'

def load_release_snapshot(root):
    manifest = json.loads((root / 'release/manifest.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('version') != VERSION:
        raise ValueError('Unsupported model manifest')
    for name, expected in manifest['files'].items():
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Invalid model manifest path')
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen model identity mismatch')
    data = json.loads((root / 'release/snapshot.json').read_text())
    if data['schema'] != 1 or data['version'] != VERSION:
        raise ValueError('Unsupported model snapshot')
    if data['bundles']['BUNDLE']['version'] != VERSION:
        raise ValueError('Inconsistent model version')
    frames = {}
    for kind, frame in data['frames'].items():
        if len(frame['records']) != manifest['rows'][kind]:
            raise ValueError('Unexpected report count')
        frames[kind] = (pd.DataFrame(frame['records']), frame['screening'])
    if len(data['bundles']['BUNDLE']['bt_fit_ids']) != manifest['bt_fit_rows']:
        raise ValueError('Unexpected fit count')
    data['frames'] = frames
    return data
