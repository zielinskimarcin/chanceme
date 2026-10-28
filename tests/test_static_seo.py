"""Regression checks for the generated deployment, not legacy root files."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_check', ROOT / 'scripts/check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class StaticReleaseTest(unittest.TestCase):
    def test_generated_output_passes_release_gate(self):
        self.assertEqual(check.validate(), [])

    def test_checker_rejects_wrong_build_mode(self):
        production = json.loads((ROOT / 'public/build-manifest.json').read_text())['indexable']
        self.assertTrue(any('indexability' in error for error in check.validate(production=not production)))

    def test_gate_catches_broken_assets_and_private_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'public'
            shutil.copytree(ROOT / 'public', output)
            (output / 'campus.css').unlink()
            (output / 'snapshot.json').write_text('{}')
            errors = check.validate(output)
            self.assertTrue(any('campus.css' in error for error in errors), errors)
            self.assertTrue(any('unexpected JSON' in error for error in errors), errors)

    def test_gate_rejects_unsafe_metadata_and_invalid_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'public'
            shutil.copytree(ROOT / 'public', output)
            home = output / 'index.html'
            html = home.read_text().replace('https://chance-me.com/"', 'https://invalid.example/"')
            html = html.replace('"@context":"https://schema.org"', '"@context":INVALID')
            home.write_text(html)
            errors = check.validate(output)
            self.assertTrue(any('canonical' in error for error in errors), errors)
            self.assertTrue(any('JSON-LD' in error for error in errors), errors)

    def test_analytics_helpers_remove_inputs_and_url_fragments(self):
        node = shutil.which('node')
        if not node:
            candidate = Path('/Users/mz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
            node = str(candidate) if candidate.is_file() else None
        self.assertIsNotNone(node, 'Node is required to verify the shipped analytics helper')
        code = """
const assert=require('node:assert/strict');
const a=require('./public/analytics.js');
assert.deepEqual(a.cleanDimensions({score:1450,gpa:9,email:'private@example.com',test_type:'SAT',program:'Management',round:'Winter',result_band:'stronger'}),{test_type:'SAT',program:'Management',round:'Winter',result_band:'stronger'});
assert.deepEqual(a.cleanDimensions({test_type:'SAT:1450',program:'private free text',round:'invalid',result_band:'9.5'}),{});
assert.equal(a.safeLocation({origin:'https://chance-me.com',pathname:'/report/',search:'?email=private@example.com',hash:'#score=1450&gpa=9'}),'https://chance-me.com/report/');
assert.equal(a.cleanPath('/private-profile/1450'),'/404/');
"""
        subprocess.run([node, '-e', code], cwd=ROOT, check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
