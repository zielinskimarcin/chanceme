"""Check generated deployment output, never the legacy/source HTML.

Usage: python scripts/check.py --production | --preview [--directory public]
Without a mode, use the generated build manifest's indexability flag.
"""
from __future__ import annotations
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
HOST = 'https://chance-me.com'
INDEXABLE = {'/', '/bocconi-gpa-calculator/', '/sat-tutoring/', '/bocconi-sat-score/', '/bocconi-test-score/', '/privacy/'}
PRIVATE_PAGES = {'/report/', '/gpa-school-sources/'}
SCHEMA_TYPES = {'/': 'WebApplication', '/bocconi-gpa-calculator/': 'WebApplication', '/sat-tutoring/': 'Service', '/bocconi-sat-score/': 'Article', '/bocconi-test-score/': 'Article'}


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.ids = set()
        self.title = ''
        self.schemas = []
        self._title = False
        self._schema = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag == 'title':
            self._title = True
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self._schema = ''

    def handle_endtag(self, tag):
        if tag == 'title':
            self._title = False
        if tag == 'script' and self._schema is not None:
            self.schemas.append(self._schema)
            self._schema = None

    def handle_data(self, text):
        if self._title:
            self.title += text
        if self._schema is not None:
            self._schema += text

    def values(self, tag, key, value, result):
        return [attrs.get(result, '') for name, attrs in self.tags if name == tag and attrs.get(key) == value]


def public_file(directory, path):
    target = directory / unquote(path).lstrip('/')
    if target.is_dir() or path.endswith('/'):
        target /= 'index.html'
    return target


def validate(directory=ROOT / 'public', production=None):
    directory = Path(directory).resolve()
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    try:
        manifest = json.loads((directory / 'build-manifest.json').read_text())
        require(type(manifest.get('indexable')) is bool, 'build manifest: missing boolean indexable')
        if production is None:
            production = manifest.get('indexable') is True
        require(manifest.get('indexable') is production, 'build manifest: indexability differs from requested mode')
        require(set(manifest['pages']) == INDEXABLE | PRIVATE_PAGES, 'build manifest: unexpected page inventory')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f'build manifest: {exc}']
    forbidden_parts = {'.git', '.venv', 'node_modules', 'lib', 'api', 'research', 'qa', 'tests', 'docs', 'scripts', 'src', '__pycache__'}
    for file in directory.rglob('*'):
        relative = file.relative_to(directory)
        require(not file.is_symlink(), f'{relative}: symlink in deployment output')
        require(not (set(relative.parts) & forbidden_parts), f'{relative}: private directory in deployment output')
        if file.is_file():
            require(not file.name.startswith('.') and file.suffix.lower() not in {'.py', '.pyc', '.csv', '.md', '.map', '.toml', '.yml', '.yaml', '.lock', '.db', '.sqlite', '.pkl', '.pickle', '.npz', '.zip', '.tar'}, f'{relative}: private/source file in deployment output')
            require(file.suffix.lower() != '.json' or relative.as_posix() == 'build-manifest.json', f'{relative}: unexpected JSON data in public output')
    pages = {}
    for route in sorted(INDEXABLE | PRIVATE_PAGES | {'/404.html'}):
        file = public_file(directory, route)
        if not file.is_file():
            errors.append(f'{route}: missing generated page')
            continue
        text = file.read_text(encoding='utf-8')
        page = Page(text)
        pages[route] = page
        require(bool(page.title.strip()), f'{route}: missing title')
        require(sum(tag == 'h1' for tag, _ in page.tags) == 1, f'{route}: expected one H1')
        require(bool(page.values('html', 'lang', 'en', 'lang')), f'{route}: missing English document language')
        robots = page.values('meta', 'name', 'robots', 'content')
        require(len(robots) == 1, f'{route}: expected one robots meta')
        tokens = {token.strip().lower() for token in ','.join(robots).split(',')}
        expected_index = production and route in INDEXABLE
        require(('index' in tokens and 'noindex' not in tokens) if expected_index else 'noindex' in tokens, f'{route}: wrong indexability for this build')
        canonicals = page.values('link', 'rel', 'canonical', 'href')
        require(canonicals == ([] if route == '/404.html' else [HOST + route]), f'{route}: incorrect/duplicate canonical')
        if route in INDEXABLE:
            descriptions = page.values('meta', 'name', 'description', 'content')
            require(len(descriptions) == 1 and bool(descriptions[0].strip()), f'{route}: missing/duplicate description')
            for prop in ['og:title', 'og:description', 'og:url', 'og:image']:
                values = page.values('meta', 'property', prop, 'content')
                require(len(values) == 1 and bool(values[0]), f'{route}: missing/duplicate {prop}')
            require(page.values('meta', 'property', 'og:url', 'content') == [HOST + route], f'{route}: Open Graph URL disagrees with canonical')
            require(page.values('meta', 'name', 'twitter:card', 'content') == ['summary_large_image'], f'{route}: missing social card')
        types = set()
        for raw in page.schemas:
            try:
                data = json.loads(raw)
                require(data.get('@context') == 'https://schema.org', f'{route}: incorrect schema context')
                for item in data.get('@graph', [data]):
                    kind = item.get('@type')
                    types.update(kind if isinstance(kind, list) else [kind])
            except (ValueError, TypeError, AttributeError):
                errors.append(f'{route}: invalid JSON-LD')
        if route in SCHEMA_TYPES:
            require(SCHEMA_TYPES[route] in types, f'{route}: missing {SCHEMA_TYPES[route]} schema')
        require('/Users/' not in text and '127.0.0.1' not in text and 'localhost:' not in text, f'{route}: local/private runtime path leaked')
        for tag, attrs in page.tags:
            if tag == 'script' and attrs.get('src'):
                require(not urlsplit(attrs['src']).netloc, f'{route}: direct third-party measurement script bypasses the privacy-controlled loader')
            if tag == 'iframe':
                require(not attrs.get('src'), f'{route}: eager third-party iframe')
        privacy_links = [attrs.get('href') for tag, attrs in page.tags if tag == 'a']
        require('/privacy/' in privacy_links, f'{route}: no privacy link')
        require('data-cookie-settings' not in text and 'cookie-notice' not in text, f'{route}: obsolete cookie popup control')
    # Resolve same-origin navigation, fragment targets, all HTML asset references,
    # and social images. External destinations are not fetched by this offline gate.
    for route, page in pages.items():
        for tag, attrs in page.tags:
            refs = [(attr, attrs[attr]) for attr in ['href', 'src', 'poster', 'data-screen'] if attrs.get(attr)]
            if attrs.get('srcset'):
                refs.extend(('srcset', item.strip().split()[0]) for item in attrs['srcset'].split(',') if item.strip())
            if tag == 'meta' and attrs.get('property') == 'og:image':
                refs.append(('content', attrs.get('content', '')))
            for attr, reference in refs:
                parsed = urlsplit(urljoin(HOST + route, reference))
                if parsed.scheme not in {'http', 'https'} or parsed.netloc != 'chance-me.com':
                    continue
                target = public_file(directory, parsed.path)
                require(target.is_file(), f'{route}: missing local target {reference}')
                if not target.is_file():
                    continue
                if parsed.fragment and target.suffix == '.html':
                    target_route = parsed.path if parsed.path.endswith('/') else parsed.path
                    other = pages.get(target_route) or Page(target.read_text())
                    require(unquote(parsed.fragment) in other.ids, f'{route}: missing fragment {reference}')
                version = re.search(r'(?:^|&)v=([a-f0-9]{12})(?:&|$)', parsed.query)
                if version:
                    require(hashlib.sha256(target.read_bytes()).hexdigest().startswith(version[1]), f'{route}: stale asset version {reference}')
    for file in directory.glob('*.css'):
        for reference in re.findall(r'url\(\s*[\"\']?([^\)\"\']+)', file.read_text()):
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc or reference.startswith('#'):
                continue
            target = public_file(directory, urljoin('/' + file.name, parsed.path))
            require(target.is_file(), f'{file.name}: missing CSS asset {reference}')
    try:
        tree = ET.fromstring((directory / 'sitemap.xml').read_text())
        urls = [node.text for node in tree.findall('{*}url/{*}loc')]
        require(len(urls) == len(set(urls)), 'sitemap: duplicate URL')
        require(set(urls) == {HOST + route for route in INDEXABLE}, 'sitemap: wrong canonical/public page inventory')
    except (OSError, ET.ParseError):
        errors.append('sitemap: missing or invalid XML')
    try:
        robots = (directory / 'robots.txt').read_text()
        require(f'Sitemap: {HOST}/sitemap.xml' in robots, 'robots: missing canonical sitemap')
        require('Disallow: /api/' in robots, 'robots: API is not excluded')
        require(not production or not re.search(r'^Disallow:\s*/\s*$', robots, re.M), 'robots: production blocks all crawling')
    except OSError:
        errors.append('robots: missing file')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--production', action='store_true')
    mode.add_argument('--preview', action='store_true')
    parser.add_argument('--directory', type=Path, default=ROOT / 'public')
    args = parser.parse_args()
    errors = validate(args.directory, True if args.production else False if args.preview else None)
    if errors:
        for error in errors:
            print('ERROR:', error, file=sys.stderr)
        return 1
    print('Release output checks passed (' + ('production' if args.production else 'preview' if args.preview else 'manifest mode') + ').')
    return 0


if __name__ == '__main__':
    sys.exit(main())
