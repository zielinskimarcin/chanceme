"""Frozen historical reported-outcome estimator for local review only.

Rows are self-reported program outcomes. This estimate is not a current-cycle
population admission probability. No public research-stage rows enter the fit.
"""
from __future__ import annotations

import math

import numpy as np


RIDGE = 4.0
FIT_CYCLES = ('24-25', '25-26')
CALIBRATION_CYCLE = '26-27'
MODEL_VERSION = 'historical-report-v1-2026-10-12'
# Frozen after the previously inspected 26-27 development comparison.
SELECTED_METHOD = {'SAT': 'latest_cycle_prior', 'Bocconi': 'calibrated_academic'}


def _academic(score, gpa, kind):
    scaled = (score - (1000 if kind == 'SAT' else 17)) / (600 if kind == 'SAT' else 33)
    return scaled, gpa / 10


def _design(frame, score_field, kind, programs, rounds):
    rows = []
    for _, row in frame.iterrows():
        s, g = _academic(float(row[score_field]), float(row.GPA), kind)
        rows.append([1.0, s, g] + [float(row.Course == p) for p in programs]
                    + [float(row.Session == r) for r in rounds])
    return np.asarray(rows, dtype=float)


def _sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))


def _fit_coefficients(frame, score_field, kind, programs, rounds):
    x = _design(frame, score_field, kind, programs, rounds)
    y = (frame.Decision == 'accept').to_numpy(dtype=float)
    theta = np.zeros(x.shape[1])
    theta[0] = math.log((y.sum() + 1) / (len(y) - y.sum() + 1))
    penalty = np.eye(len(theta)) * RIDGE
    penalty[0, 0] = 0
    for _ in range(100):
        p = _sigmoid(x @ theta)
        gradient = x.T @ (p - y) + penalty @ theta
        hessian = x.T @ ((p * (1-p))[:, None] * x) + penalty + np.eye(len(theta)) * 1e-8
        candidate = theta - np.linalg.solve(hessian, gradient)
        candidate[1:3] = np.maximum(candidate[1:3], 0)
        if np.max(np.abs(candidate - theta)) < 1e-9:
            theta = candidate
            break
        theta = candidate
    return theta


def _calibrate_intercept(base, outcomes, weights=None):
    logits = np.log(np.clip(base, 1e-6, 1-1e-6) / np.clip(1-base, 1e-6, 1-1e-6))
    weights = np.ones(len(outcomes)) if weights is None else np.asarray(weights,dtype=float)
    weights = weights/weights.sum()
    target = float(np.sum(weights*outcomes))
    if not 0 < target < 1:
        raise ValueError('Intercept calibration requires both reported outcome classes.')
    # The calibration equation is monotone. Bisection avoids Newton's huge
    # steps when a separated diagnostic subset has probabilities near 0/1.
    lower, upper = -50.0, 50.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        if float(np.sum(weights*_sigmoid(logits + midpoint))) < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


class HistoricalReportedOutcomeModel:
    def __init__(self, data, kind, fit_cycles=FIT_CYCLES,
                 calibration_cycle=CALIBRATION_CYCLE):
        self.kind = kind
        self.score_field = 'SAT' if kind == 'SAT' else 'Bocconi_Test'
        train = data[data.Academic_Year.isin(fit_cycles)].copy()
        calibration = data[data.Academic_Year == calibration_cycle].copy()
        if train.empty or calibration.empty:
            raise ValueError('Both fit and calibration cycles need rows.')
        self.programs = sorted(train.Course.unique())
        self.rounds = sorted(train.Session.unique())
        self.coefficients = _fit_coefficients(train, self.score_field, kind,
                                              self.programs, self.rounds)
        base = _sigmoid(_design(calibration, self.score_field, kind,
                                self.programs, self.rounds) @ self.coefficients)
        outcomes = (calibration.Decision == 'accept').to_numpy(dtype=float)
        self.offset = _calibrate_intercept(base, outcomes)
        self.fit_cycles = tuple(fit_cycles)
        self.calibration_cycle = calibration_cycle
        self.fit_rows = len(train)
        self.calibration_rows = len(calibration)
        self.calibration_admitted = int(outcomes.sum())
        self.latest_prior = (self.calibration_admitted + 1) / (self.calibration_rows + 2)
        self.fit_context = train.groupby(['Course', 'Session']).Decision.agg(
            total='size', admitted=lambda x: int((x == 'accept').sum()))
        self.calibration_context = calibration.groupby(['Course', 'Session']).Decision.agg(
            total='size', admitted=lambda x: int((x == 'accept').sum()))

    def predict(self, score, gpa, program, session):
        if program not in self.programs or session not in self.rounds:
            return None
        s, g = _academic(float(score), float(gpa), self.kind)
        vector = np.asarray([1.0, s, g] + [float(program == p) for p in self.programs]
                            + [float(session == r) for r in self.rounds])
        academic_estimate = float(_sigmoid(vector @ self.coefficients + self.offset))
        method = SELECTED_METHOD[self.kind]
        estimate = (self.latest_prior if method == 'latest_cycle_prior'
                    else academic_estimate)
        def counts(table):
            if (program, session) not in table.index:
                return {'rows': 0, 'admitted': 0, 'rejected': 0}
            row = table.loc[(program, session)]
            n, a = int(row.total), int(row.admitted)
            return {'rows': n, 'admitted': a, 'rejected': n-a}
        fit = counts(self.fit_context)
        latest = counts(self.calibration_context)
        return {
            'estimate': round(estimate, 4),
            'method': method,
            'model_version': MODEL_VERSION,
            'personalized_academic_estimate': method == 'calibrated_academic',
            'target': 'historical self-reported program outcome, not current admission chance',
            'status': 'experimental_unvalidated_for_2027_28',
            'fit_cycles': list(self.fit_cycles),
            'calibration_cycle': self.calibration_cycle,
            'fit_context': fit, 'latest_context': latest,
            'latest_overall': {'rows': self.calibration_rows,
                               'admitted': self.calibration_admitted},
            'academic_slopes': {'test': round(float(self.coefficients[1]), 5),
                                'gpa': round(float(self.coefficients[2]), 5)},
            'quality': 'self-selected reports; identity, category, GPA conversion and program equivalence unverified',
        }


# Chosen by four score/GPA/cycle/round fingerprint-grouped folds on 24-25/25-26
# only. The already inspected 26-27 cycle did not determine these penalties.
STANDARDIZED_ALPHA = {'SAT': 0.03, 'Bocconi': 0.01}
STANDARDIZED_VERSION = 'historical-report-standardized-v2-2026-10-13'
HINGE_VERSION = 'historical-report-monotone-hinge-v3-2026-10-13'
MIN_CONTEXT_ROWS = 20
MIN_CONTEXT_GROUPS = 8
MIN_OUTCOME_ROWS = 2
MODERATE_EXTRAPOLATION_SD = 0.5
SEVERE_JOINT_DISTANCE_FLOOR_SD = 1.0
SEVERE_JOINT_DISTANCE_MULTIPLIER = 3.0


def _leave_one_out_radius(points):
    if len(points) <= 1:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.quantile(distances.min(axis=1), .9))


def _standard_design(frame, field, programs, rounds, mean, sd):
    academic = frame[[field, 'GPA']].to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(frame)), (academic-mean)/sd,
        *[(frame.Course == p).to_numpy(dtype=float) for p in programs],
        *[(frame.Session == r).to_numpy(dtype=float) for r in rounds],
    ])


def _standard_fit(x, y, alpha, weights=None):
    """Mean Bernoulli loss + alpha/2 times squared non-intercept coefficients."""
    n, d = x.shape
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    theta = np.zeros(d)
    theta[0] = math.log((y.sum()+1)/(n-y.sum()+1))
    penalty = np.eye(d)*alpha
    penalty[0, 0] = 0
    for _ in range(200):
        p = _sigmoid(x @ theta)
        gradient = x.T @ (weights*(p-y)) + penalty @ theta
        hessian = x.T @ ((weights*p*(1-p))[:, None]*x) + penalty + np.eye(d)*1e-9
        candidate = theta - np.linalg.solve(hessian, gradient)
        candidate[1:3] = np.maximum(candidate[1:3], 0)
        if np.max(np.abs(candidate-theta)) < 1e-10:
            theta = candidate
            break
        theta = candidate
    return theta


def _hinge_design(frame, field, programs, rounds, mean, sd, knots):
    academic=(frame[[field,'GPA']].to_numpy(dtype=float)-mean)/sd
    score,grade=academic[:,0],academic[:,1]
    return np.column_stack([
        np.ones(len(frame)),score,grade,
        np.maximum(0,score-knots[0]),np.maximum(0,grade-knots[1]),
        *[(frame.Course==p).to_numpy(dtype=float) for p in programs],
        *[(frame.Session==r).to_numpy(dtype=float) for r in rounds],
    ])


def _hinge_objective(x,y,theta,penalty,weights):
    z=x@theta
    return float(np.sum(weights*(np.logaddexp(0,z)-y*z))+
                 .5*np.sum(penalty*theta*theta))


def _fit_hinge(x,y,alpha,weights=None):
    """Convex mean-loss ridge fit with nonnegative academic base/hinge slopes."""
    n,d=x.shape
    weights=np.ones(n) if weights is None else np.asarray(weights,dtype=float)
    weights=weights/weights.sum()
    penalty=np.full(d,alpha)
    penalty[0]=0
    penalty[3:5]=4*alpha
    theta=np.zeros(d)
    theta[0]=math.log((y.sum()+1)/(n-y.sum()+1))
    constrained=(1,2,3,4)
    for _ in range(400):
        p=_sigmoid(x@theta)
        gradient=x.T@(weights*(p-y))+penalty*theta
        active=[j for j in constrained if theta[j]<=1e-12 and gradient[j]>0]
        free=[j for j in range(d) if j not in active]
        projected=gradient.copy()
        projected[active]=0
        if float(np.max(np.abs(projected)))<1e-9:
            break
        hessian=x.T@((weights*p*(1-p))[:,None]*x)+np.diag(penalty)+np.eye(d)*1e-10
        direction=np.zeros(d)
        direction[free]=-np.linalg.solve(hessian[np.ix_(free,free)],gradient[free])
        old=_hinge_objective(x,y,theta,penalty,weights)
        accepted=False
        scale=1.0
        for _ in range(40):
            candidate=theta+scale*direction
            candidate[list(constrained)]=np.maximum(candidate[list(constrained)],0)
            if _hinge_objective(x,y,candidate,penalty,weights)<old-1e-14:
                theta=candidate
                accepted=True
                break
            scale*=.5
        if not accepted:
            break
    p=_sigmoid(x@theta)
    gradient=x.T@(weights*(p-y))+penalty*theta
    kkt=max(abs(gradient[j]) if j not in constrained or theta[j]>1e-9
            else max(0.0,-gradient[j]) for j in range(d))
    if not np.isfinite(theta).all() or kkt>1e-6:
        raise RuntimeError(f'Monotone hinge fit failed KKT check: {kkt}')
    return theta


class StandardizedHistoricalModel:
    """Personalized historical-report estimate with graded training support."""

    def __init__(self, data, kind, weighting='row', family='linear'):
        if weighting not in ('row', 'proxy_group'):
            raise ValueError('Unknown row weighting')
        if family not in ('linear','monotone_hinge'):
            raise ValueError('Unknown academic family')
        self.kind = kind
        self.weighting = weighting
        self.family = family
        self.fit_cycles = FIT_CYCLES
        self.calibration_cycle = CALIBRATION_CYCLE
        self.score_field = 'SAT' if kind == 'SAT' else 'Bocconi_Test'
        data = data.drop_duplicates(subset=[c for c in data.columns if c != 'source_row']).copy()
        data = data.sort_values([self.score_field,'GPA','Course','Decision',
                                 'Academic_Year','Session'], kind='stable')
        train = data[data.Academic_Year.isin(FIT_CYCLES)].copy()
        calibration = data[data.Academic_Year == CALIBRATION_CYCLE].copy()
        if train.empty or calibration.empty:
            raise ValueError('Fit and calibration cycles need rows.')
        self.programs = sorted(train.Course.unique())
        self.rounds = sorted(train.Session.unique())
        key_columns = [self.score_field,'GPA','Academic_Year','Session']
        train['proxy_weight'] = 1/train.groupby(key_columns)[self.score_field].transform('size')
        calibration['proxy_weight'] = 1/calibration.groupby(key_columns)[self.score_field].transform('size')
        academic = train[[self.score_field, 'GPA']].to_numpy(dtype=float)
        self.mean = academic.mean(axis=0)
        self.sd = academic.std(axis=0)
        if (self.sd <= 0).any():
            raise ValueError('Training features need variation.')
        self.knots=np.median((academic-self.mean)/self.sd,axis=0)
        self.design=(lambda frame: _hinge_design(frame,self.score_field,
                       self.programs,self.rounds,self.mean,self.sd,self.knots)
                     if family=='monotone_hinge' else
                     _standard_design(frame,self.score_field,self.programs,
                                      self.rounds,self.mean,self.sd))
        x = self.design(train)
        y = (train.Decision == 'accept').to_numpy(dtype=float)
        self.alpha = STANDARDIZED_ALPHA[kind]
        fit_weights = (train.proxy_weight.to_numpy(dtype=float)
                       if weighting == 'proxy_group' else None)
        self.coefficients = (_fit_hinge(x,y,self.alpha,fit_weights)
                             if family=='monotone_hinge' else
                             _standard_fit(x,y,self.alpha,fit_weights))
        z = self.design(calibration)
        outcomes = (calibration.Decision == 'accept').to_numpy(dtype=float)
        calibration_weights = (calibration.proxy_weight.to_numpy(dtype=float)
                               if weighting == 'proxy_group' else None)
        self.offset = _calibrate_intercept(_sigmoid(z @ self.coefficients),
                                            outcomes, calibration_weights)
        self.fit_rows = len(train)
        self.calibration_rows = len(calibration)
        self.calibration_admitted = int(outcomes.sum())
        self.latest_prior = (self.calibration_admitted+1)/(self.calibration_rows+2)
        def context_table(frame):
            return frame.groupby(['Course','Session']).apply(
                lambda part: {'rows':len(part),
                              'admitted':int((part.Decision=='accept').sum()),
                              'rejected':int((part.Decision=='reject').sum()),
                              'approx_groups':round(float(part.proxy_weight.sum()),3)})
        self.fit_context = context_table(train)
        self.latest_context = context_table(calibration)
        global_points = (academic-self.mean)/self.sd
        self.global_support = {
            'score_min':float(academic[:,0].min()),
            'score_max':float(academic[:,0].max()),
            'gpa_min':float(academic[:,1].min()),
            'gpa_max':float(academic[:,1].max()),
            'joint_radius_standardized':round(_leave_one_out_radius(global_points),6),
            'points':global_points,
        }
        self.support = {}
        for key, part in train.groupby(['Course','Session']):
            points = (part[[self.score_field,'GPA']].to_numpy(dtype=float)-self.mean)/self.sd
            radius = _leave_one_out_radius(points)
            self.support[key] = {
                'score_min': float(part[self.score_field].min()),
                'score_max': float(part[self.score_field].max()),
                'gpa_min': float(part.GPA.min()),
                'gpa_max': float(part.GPA.max()),
                'joint_radius_standardized':round(radius,6),
            }
            self.support[key]['points'] = points

    @staticmethod
    def _counts(table, key):
        if key not in table.index:
            return {'rows':0,'admitted':0,'rejected':0,'approx_groups':0.0}
        return dict(table.loc[key])

    def frozen_state(self):
        """Explicit JSON state, exported offline. No pickle or executable content."""
        contexts = []
        for key, bounds in sorted(self.support.items()):
            contexts.append({'program': key[0], 'session': key[1],
                             'support': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                         for k, v in bounds.items()},
                             'fit': self._counts(self.fit_context, key),
                             'latest': self._counts(self.latest_context, key)})
        return {
            'kind': self.kind, 'family': self.family, 'weighting': self.weighting,
            'fit_cycles': list(self.fit_cycles), 'calibration_cycle': self.calibration_cycle,
            'programs': list(self.programs), 'rounds': list(self.rounds),
            'mean': self.mean.tolist(), 'sd': self.sd.tolist(), 'knots': self.knots.tolist(),
            'coefficients': self.coefficients.tolist(), 'offset': self.offset, 'alpha': self.alpha,
            'fit_rows': self.fit_rows, 'calibration_rows': self.calibration_rows,
            'calibration_admitted': self.calibration_admitted, 'latest_prior': self.latest_prior,
            'global_support': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in self.global_support.items()},
            'contexts': contexts,
        }

    @classmethod
    def from_frozen_state(cls, state):
        """Instantiate validated state without calling __init__ or fitting."""
        import pandas as pd
        obj = cls.__new__(cls)
        for field in ('kind', 'family', 'weighting', 'calibration_cycle', 'programs',
                      'rounds', 'offset', 'alpha', 'fit_rows', 'calibration_rows',
                      'calibration_admitted', 'latest_prior'):
            setattr(obj, field, state[field])
        obj.fit_cycles = tuple(state['fit_cycles'])
        obj.score_field = 'SAT' if obj.kind == 'SAT' else 'Bocconi_Test'
        for field in ('mean', 'sd', 'knots', 'coefficients'):
            setattr(obj, field, np.asarray(state[field], dtype=float))
        obj.design = (lambda frame: _hinge_design(frame, obj.score_field,
                          obj.programs, obj.rounds, obj.mean, obj.sd, obj.knots)
                      if obj.family == 'monotone_hinge' else
                      _standard_design(frame, obj.score_field, obj.programs,
                                       obj.rounds, obj.mean, obj.sd))
        obj.global_support = {**state['global_support'],
                              'points': np.asarray(state['global_support']['points'], dtype=float)}
        obj.support = {}
        fit, latest, keys = [], [], []
        for context in state['contexts']:
            key = (context['program'], context['session'])
            keys.append(key)
            obj.support[key] = {**context['support'],
                               'points': np.asarray(context['support']['points'], dtype=float)}
            fit.append(context['fit'])
            latest.append(context['latest'])
        index = pd.MultiIndex.from_tuples(keys, names=['Course', 'Session'])
        obj.fit_context = pd.Series(fit, index=index, dtype=object)
        obj.latest_context = pd.Series(latest, index=index, dtype=object)
        return obj

    def predict(self, score, gpa, program, session):
        key = (program, session)
        if program not in self.programs or session not in self.rounds or key not in self.support:
            return None
        score, gpa = float(score), float(gpa)
        bounds = self.support[key]
        marginal = (bounds['score_min'] <= score <= bounds['score_max'] and
                    bounds['gpa_min'] <= gpa <= bounds['gpa_max'])
        point = (np.asarray([score,gpa])-self.mean)/self.sd
        nearest = float(np.linalg.norm(bounds['points']-point,axis=1).min())
        dense_joint = nearest <= bounds['joint_radius_standardized'] + 1e-6
        context_below = np.maximum((np.asarray([bounds['score_min'],bounds['gpa_min']])
                            - np.asarray([score,gpa]))/self.sd,0)
        context_above = np.maximum((np.asarray([score,gpa])
                            - np.asarray([bounds['score_max'],bounds['gpa_max']]))/self.sd,0)
        context_excess = context_below+context_above
        global_bounds = self.global_support
        global_below = np.maximum((np.asarray([global_bounds['score_min'],global_bounds['gpa_min']])
                                   - np.asarray([score,gpa]))/self.sd,0)
        global_above = np.maximum((np.asarray([score,gpa])
                                   - np.asarray([global_bounds['score_max'],global_bounds['gpa_max']]))/self.sd,0)
        global_excess = global_below+global_above
        global_nearest = float(np.linalg.norm(global_bounds['points']-point,axis=1).min())
        severe_joint_limit = max(SEVERE_JOINT_DISTANCE_FLOOR_SD,
                                 SEVERE_JOINT_DISTANCE_MULTIPLIER *
                                 global_bounds['joint_radius_standardized'])
        severe = (bool((global_excess > MODERATE_EXTRAPOLATION_SD + 1e-9).any())
                  or global_nearest > severe_joint_limit + 1e-6)
        if severe:
            support_class = 'severe_out_of_distribution'
        elif not marginal:
            # A narrow *pooled* envelope does not imply a narrow local excess.
            support_class = ('context_extrapolation' if bool(
                (context_excess > MODERATE_EXTRAPOLATION_SD + 1e-9).any())
                else 'modest_extrapolation')
        elif dense_joint:
            support_class = 'nearby_interpolation'
        else:
            support_class = 'sparse_interpolation'
        fit_context = self._counts(self.fit_context,key)
        latest_context = self._counts(self.latest_context,key)
        context_ok = (fit_context['rows'] >= MIN_CONTEXT_ROWS and
                      fit_context['approx_groups'] >= MIN_CONTEXT_GROUPS and
                      min(fit_context['admitted'],fit_context['rejected']) >= MIN_OUTCOME_ROWS and
                      min(latest_context['admitted'],latest_context['rejected']) >= 1)
        evidence_level = ('very_limited' if support_class in
                          ('modest_extrapolation','context_extrapolation','severe_out_of_distribution')
                          or min(fit_context['admitted'],fit_context['rejected']) == 0
                          else 'moderate_unverified' if support_class == 'nearby_interpolation'
                          and context_ok else 'limited')
        public_bounds = {k:v for k,v in bounds.items() if k != 'points'}
        public_global_bounds = {k:v for k,v in global_bounds.items() if k != 'points'}
        result = {
            'estimate': None, 'method':('standardized_monotone_hinge_academic'
                                       if self.family=='monotone_hinge' else
                                       'standardized_monotone_academic'),
            'personalized_academic_estimate': True,
            'model_version':(HINGE_VERSION if self.family=='monotone_hinge'
                             else STANDARDIZED_VERSION),
            'target':'historical self-reported program outcome, not current admission chance',
            'status':('experimental_unvalidated_for_2027_28' if not severe
                      else 'abstained_severe_out_of_distribution'),
            'abstention_reason':(None if not severe else
                                 'Academic inputs are beyond the pooled training domain and its declared modest margin.'),
            'support_bounds':public_bounds,
            'global_support_bounds':public_global_bounds,
            'support_class':support_class,
            'evidence_level':evidence_level,
            'support_definition':'exact-context nearby/sparse interpolation; pooled training envelope plus at most 0.5 training SD and global nearest distance at most max(1 SD, 3× global 90th-percentile leave-one-out distance); identities unverified',
            'joint_nearest_distance_standardized':round(nearest,6),
            'joint_support':bool(dense_joint),
            'global_joint_nearest_distance_standardized':round(global_nearest,6),
            'severe_global_joint_distance_limit_standardized':round(severe_joint_limit,6),
            'context_marginal_excess_standardized':{'score':round(float(context_excess[0]),6),
                                                    'gpa':round(float(context_excess[1]),6)},
            'context_marginal_excess_raw':{'score':round(float(context_excess[0]*self.sd[0]),6),
                                          'gpa':round(float(context_excess[1]*self.sd[1]),6)},
            'context_excess_direction':{'score':('below' if score < bounds['score_min'] else
                                                'above' if score > bounds['score_max'] else 'within'),
                                        'gpa':('below' if gpa < bounds['gpa_min'] else
                                               'above' if gpa > bounds['gpa_max'] else 'within')},
            'global_marginal_excess_standardized':{'score':round(float(global_excess[0]),6),
                                                   'gpa':round(float(global_excess[1]),6)},
            'global_marginal_excess_raw':{'score':round(float(global_excess[0]*self.sd[0]),6),
                                         'gpa':round(float(global_excess[1]*self.sd[1]),6)},
            'context_quality':('local_supported' if context_ok else 'pooled_or_insufficient'),
            'context_policy':{'min_fit_rows_for_moderate':MIN_CONTEXT_ROWS,
                              'min_approx_groups':MIN_CONTEXT_GROUPS,
                              'min_each_fit_outcome':MIN_OUTCOME_ROWS,
                              'latest_requires_both_outcomes':True},
            'fit_cycles':list(FIT_CYCLES), 'calibration_cycle':CALIBRATION_CYCLE,
            'fit_context':fit_context,
            'latest_context':latest_context,
            'latest_overall':{'rows':self.calibration_rows,'admitted':self.calibration_admitted},
            'latest_cycle_prior':round(self.latest_prior,4),
            'alpha':self.alpha,
            'academic_slopes_per_raw_point':{
                'test':round(float(self.coefficients[1]/self.sd[0]),6),
                'gpa':round(float(self.coefficients[2]/self.sd[1]),6)},
            'academic_hinge_terms':(
                {'test_knot_raw':round(float(self.mean[0]+self.knots[0]*self.sd[0]),6),
                 'gpa_knot_raw':round(float(self.mean[1]+self.knots[1]*self.sd[1]),6),
                 'test_hinge_slope_per_raw_point':round(float(self.coefficients[3]/self.sd[0]),6),
                 'gpa_hinge_slope_per_raw_point':round(float(self.coefficients[4]/self.sd[1]),6)}
                if self.family=='monotone_hinge' else None),
            'quality':'self-selected reports; identity, category, GPA conversion and program equivalence unverified',
        }
        if not severe:
            score_z=(score-self.mean[0])/self.sd[0]
            gpa_z=(gpa-self.mean[1])/self.sd[1]
            academic=[score_z,gpa_z]
            if self.family=='monotone_hinge':
                academic.extend((max(0,score_z-self.knots[0]),
                                 max(0,gpa_z-self.knots[1])))
            vector = np.asarray([1.0]+academic+
                                [float(program==p) for p in self.programs] +
                                [float(session==r) for r in self.rounds])
            result['estimate'] = round(float(_sigmoid(vector @ self.coefficients+self.offset)),4)
        return result
