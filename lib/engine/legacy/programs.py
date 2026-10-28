"""2027–28 international English-program contract; scoped historical proxies."""
ADMISSIONS_SOURCE = 'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/admissions'
SAT_SOURCE = 'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/sat-and-act'
TEST_SOURCE = 'https://www.unibocconi.it/en/applying-bocconi/bachelor-and-law-programs/application-and-admissions/online-bocconi-test'
BASE = 'https://www.unibocconi.it/en/programs/bachelor-science/'
AI_PROGRAM = 'Mathematical and Computing Sciences for Artificial Intelligence'
ARTS_PROGRAM = 'Management for the Arts and Cultures'
REVIEWED_ON = '2026-10-13'

_SPECS = [
    ('Management', 'BIEM', 'management', 'international-economics-and-management',
     'renamed_historical_proxy',
     'Historical BIEM is a proxy for Management; a name/curriculum connection does not establish admissions equivalence.'),
    ('Finance', 'BIEF', 'finance', 'international-economics-and-finance',
     'renamed_historical_proxy',
     'Historical BIEF combined Economics and Finance tracks. It is an imperfect proxy for current Finance; historic selectivity cannot be transferred.'),
    ('Management and Computer Science', 'BEMACS', 'management-and-computer-science',
     'economics-management-and-computer-science', 'renamed_historical_proxy',
     'Historical BEMACS is a related predecessor. Current places and applicant competition differ; unchanged selectivity is not established.'),
    (ARTS_PROGRAM, 'BEMACC', 'management-arts-and-cultures',
     'economics-and-management-arts-culture-and-communication', 'renamed_historical_proxy',
     'BEMACC is the reviewed English historical proxy. Bare CLEACC reports are excluded because the old code covered Italian and English tracks.'),
    ('International Politics and Government', 'BIG', 'international-politics-and-government',
     None, 'same_name_historical_comparison',
     'BIG uses the same program name. Reported outcomes are historical and do not establish current-cycle selectivity.'),
    (AI_PROGRAM, 'BAI', 'mathematical-and-computing-sciences-artificial-intelligence',
     None, 'same_name_historical_comparison',
     'BAI uses the same program name. Current places and applicant competition differ; historical outcomes are not a validated 2027–28 chance.'),
]

PROGRAM_CONTRACTS = {}
for name, proxy, slug, predecessor, policy, warning in _SPECS:
    sources = [ADMISSIONS_SOURCE, BASE + slug]
    if predecessor:
        sources.append(BASE + predecessor)
    if name == ARTS_PROGRAM:
        sources.append('https://didattica.unibocconi.it/tsg/testo.php?comando=Base&edizione=2027&idAnt=28209&idr=28209&strperc=&volume=N3')
    PROGRAM_CONTRACTS[name] = {
        'id': name, 'name': name, 'historical_proxy': proxy, 'mapping_policy': policy,
        'mapping_notice': warning, 'historical_admissions_equivalence_verified': False,
        'current_cycle_outcomes': 0, 'official_program_url': BASE + slug,
        'mapping_sources': sources, 'reviewed_on': REVIEWED_ON,
        'test_minimums': {'sat_total': 1040, 'sat_each_section': 520,
                          'sat_math_additional': 600 if name == AI_PROGRAM else None,
                          'bocconi_total': 17,
                          'bocconi_math_additional': 11 if name == AI_PROGRAM else None},
        'eligibility_sources': [SAT_SOURCE, TEST_SOURCE],
        'historical_report_category': 'unverified; not a confirmed international-only sample',
        'excluded_historical_codes': ['CLEACC'] if name == ARTS_PROGRAM else [],
    }

PROXIES = {name: (c['historical_proxy'], c['mapping_notice'])
           for name, c in PROGRAM_CONTRACTS.items()}
