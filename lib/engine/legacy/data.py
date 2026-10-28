"""Read and screen the preserved historical CSVs; never fit a model."""
from pathlib import Path
import pandas as pd

YEARS = ('26-27', '25-26', '24-25')
DATASETS = {'SAT': ('bocconi_master_dataset.csv', 'SAT'),
            'Bocconi': ('bocconitest.csv', 'Bocconi_Test')}


def load_profiles(filename, score_column, directory=None):
    directory = Path(directory) if directory else Path(__file__).parent
    data = pd.read_csv(directory / filename, sep=';')
    data['source_row'] = data.index + 2
    required = {score_column, 'GPA', 'Course', 'Decision', 'Academic_Year', 'Session'}
    if required - set(data.columns):
        raise RuntimeError(f'{filename} has missing columns')
    original = len(data)
    data = data.drop_duplicates(subset=[c for c in data.columns if c != 'source_row']).copy()
    duplicates = original - len(data)
    data[score_column] = pd.to_numeric(data[score_column], errors='coerce')
    data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
    valid = (data[score_column].notna() & data.GPA.between(0, 10)
             & data.Decision.isin({'accept', 'reject'})
             & data.Academic_Year.isin(YEARS)
             & data.Session.isin({'Early', 'Winter', 'Spring'}))
    invalid = int((~valid).sum())
    return data.loc[valid].copy(), {'exact_duplicates': duplicates, 'invalid_rows': invalid}


def single_program_fingerprint_rows(data, score_field):
    key = [score_field, 'GPA', 'Academic_Year', 'Session']
    multi_program = data.groupby(key).Course.transform('nunique') > 1
    return data.loc[~multi_program].copy(), int(multi_program.sum())
