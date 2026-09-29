import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / 'bocconi-sat-score' / 'index.html'
BOCCONI_TEST_GUIDE = ROOT / 'bocconi-test-score' / 'index.html'


class MetadataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.canonical = None
        self.description = None
        self.title = ''
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'title':
            self._in_title = True
        if tag == 'link' and attributes.get('rel') == 'canonical':
            self.canonical = attributes.get('href')
        if tag == 'meta' and attributes.get('name') == 'description':
            self.description = attributes.get('content')

    def handle_endtag(self, tag):
        if tag == 'title':
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


class StaticSeoTestCase(unittest.TestCase):
    def setUp(self):
        self.html = GUIDE.read_text(encoding='utf-8')
        self.parser = MetadataParser()
        self.parser.feed(self.html)

    def test_sat_guide_has_unique_metadata(self):
        self.assertIn('Bocconi SAT Score', self.parser.title)
        self.assertEqual(
            self.parser.canonical,
            'https://chance-me.com/bocconi-sat-score/',
        )
        self.assertGreater(len(self.parser.description), 100)
        self.assertLessEqual(len(self.parser.description), 165)

    def test_sat_guide_distinguishes_minimum_from_cutoff(self):
        self.assertIn('1040 overall', self.html)
        self.assertIn('520 in each SAT section', self.html)
        self.assertIn('does not publish a guaranteed competitive SAT cutoff', self.html)
        self.assertIn('not affiliated with Bocconi University', self.html)

    def test_sat_guide_links_official_sources(self):
        self.assertIn('unibocconi.it/en/applying-bocconi', self.html)
        self.assertIn('/sat-and-act', self.html)
        self.assertIn('/gpa-and-high-school-curriculum', self.html)

    def test_structured_data_is_valid_json(self):
        match = re.search(
            r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        data = json.loads(match.group(1))
        self.assertEqual(data['@context'], 'https://schema.org')
        self.assertEqual({item['@type'] for item in data['@graph']}, {'Article', 'FAQPage'})

    def test_sitemap_and_homepage_link_to_guide(self):
        sitemap = (ROOT / 'sitemap.xml').read_text(encoding='utf-8')
        homepage = (ROOT / 'index.html').read_text(encoding='utf-8')
        self.assertIn('https://chance-me.com/bocconi-sat-score/', sitemap)
        self.assertIn('href="/bocconi-sat-score/"', homepage)
        self.assertIn("trackEvent('guide_opened'", homepage)

    def test_bocconi_test_guide_metadata_and_claims(self):
        html = BOCCONI_TEST_GUIDE.read_text(encoding='utf-8')
        parser = MetadataParser()
        parser.feed(html)
        self.assertIn('Bocconi Test Score', parser.title)
        self.assertEqual(parser.canonical, 'https://chance-me.com/bocconi-test-score/')
        self.assertGreater(len(parser.description), 100)
        self.assertLessEqual(len(parser.description), 165)
        self.assertIn('17 out of 50', html)
        self.assertIn('50 questions', html)
        self.assertIn('75 minutes', html)
        self.assertIn('does not publish a guaranteed competitive Bocconi Test score', html)

    def test_bocconi_test_guide_sources_schema_and_links(self):
        html = BOCCONI_TEST_GUIDE.read_text(encoding='utf-8')
        match = re.search(
            r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        data = json.loads(match.group(1))
        self.assertEqual({item['@type'] for item in data['@graph']}, {'Article', 'FAQPage'})
        self.assertIn('/online-bocconi-test', html)
        self.assertIn("guide:'bocconi_test_score'", html)
        sitemap = (ROOT / 'sitemap.xml').read_text(encoding='utf-8')
        homepage = (ROOT / 'index.html').read_text(encoding='utf-8')
        self.assertIn('https://chance-me.com/bocconi-test-score/', sitemap)
        self.assertIn('href="/bocconi-test-score/"', homepage)


if __name__ == '__main__':
    unittest.main()
