#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:53:16 2018

@author: ian
"""

import datetime as dt
import numpy as np
import pandas as pd
from lmfit import Model
import pdb

#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class partition():
    """
    Class for fitting of respiration parameters and estimation of respiration
    WARNING - NO FILTERING OR DATA INTEGRITY CHECKS APPLIED HERE!!!

    Args:
        * dataframe (pd.dataframe): containing a minimum of temperature, solar
          radiation, VPD and CO2 flux.
    Kwargs:
        * names_dict (dict): maps the variable names used in the dataset to
          common names (keys must be 'air_temperature', 'soil_temperature',
          'insolation', 'Cflux'); if None, defaults to the internal
          specification, which works for PyFluxPro.
        * weights_air_soil (str, or list of ints / floats): if str, must be
          either 'air' or 'soil', which determines which temperature series is
          used for the fit; if list is supplied, it must have two numbers (ints
          or floats), which are used for the weighting in the ratio air:soil
          e.g. choice of [3, 1] would cause weighting of 3:1 in favour of air
          temperature, or e.g. [1, 3] would result in the reverse.
    """
    def __init__(self, dataframe, names_dict=None, weights_air_soil='air',
                 noct_threshold=10):

        self.internal_names = _define_default_internal_names()
        self.external_names = _define_default_external_names(names_dict)
        self.interval = _check_continuity(dataframe.index)
        self.variable_map = {self.external_names[key]: self.internal_names[key]
                             for key in self.internal_names}
        _check_weights_format(weights_air_soil)
        if not isinstance(noct_threshold, (int, float)):
            raise TypeError('Arg "noct_threshold" must be of type int or float')
        self.noct_threshold = noct_threshold
        self.weighting = weights_air_soil
        self.df = make_formatted_df(dataframe, self.variable_map,
                                    self.weighting)
        self.noct_threshold = noct_threshold
#------------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Methods
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_daytime_parameters(self, Eo=None, window_size=4,
                                    window_step=4, fit_daytime_rb=False):

        # return self._estimate_parameters(mode='day', window_size=window_size,
        #                                  window_step=window_step,
        #                                  fit_daytime_rb=fit_daytime_rb)

        base_priors_dict = self.get_prior_parameter_estimates()
        update_priors_dict = base_priors_dict.copy()
        result_list, date_list = [], []
        if not Eo:
            Eo = self.estimate_Eo()
        if not fit_daytime_rb:
            rb_df = self.estimate_nocturnal_parameters(Eo=Eo)
        print('Processing the following dates (day mode): ')
        date_iterator = self.make_date_iterator(window=window_size,
                                                step=window_step)
        for date in date_iterator.index:
            try:
                rb = rb_df.loc[date, 'rb']
            except NameError:
                rb = None
            df = get_subset(self.df,
                            start=date_iterator.loc[date, 'Start'],
                            end=date_iterator.loc[date, 'End'])
            print((date.date()), end=' ')
            try:
                result = (
                    _fit_day_params(df, Eo, update_priors_dict,
                                    self.noct_threshold, rb=rb)
                    )
                pdb.set_trace()
                update_priors_dict['alpha'] = result['alpha']
                result_list.append(result)
                date_list.append(date)
            except RuntimeError as e:
                update_priors_dict['alpha'] = base_priors_dict['alpha']
                print('- {}'.format(e))
                continue
            print ()

    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_Eo(self, window_size=15, window_step=5):

        """Estimate the activation energy type parameter for the L&T Arrhenius
           style equation using nocturnal data"""

        Eo_list = []
        date_iterator = self.make_date_iterator(window=window_size,
                                                step=window_step)
        for date in date_iterator.index:
            df = get_subset(self.df,
                            start=date_iterator.loc[date, 'Start'],
                            end=date_iterator.loc[date, 'End'])
            df = df.loc[df.Fsd < self.noct_threshold]
            if not len(df) > 6:
                continue
            if not df.TC.max() - df.TC.min() >= 5:
                continue
            f = Lloyd_and_Taylor
            model = Model(f, independent_vars = ['t_series'])
            params = model.make_params(
                rb = 1, Eo = self.get_prior_parameter_estimates()['Eo']
                )
            result = model.fit(df.NEE,
                               t_series = df.TC,
                               params = params)
            if not 50 < result.params['Eo'].value < 400:
                continue
            if result.params['Eo'].stderr > result.params['Eo'].value / 2.0:
                continue
            Eo_list.append([result.params['Eo'].value,
                            result.params['Eo'].stderr])
        if len(Eo_list) == 0:
            raise RuntimeError(
                'Could not find any valid estimates of Eo! Exiting...'
                )
        print('Found {} valid estimates of Eo'.format(str(len(Eo_list))))
        Eo_array = np.array(Eo_list)
        Eo = ((Eo_array[:, 0] / (Eo_array[:, 1])).sum() /
              (1 / Eo_array[:, 1]).sum())
        if not 50 < Eo < 400:
            raise RuntimeError(
                'Eo value {} outside acceptable parameter range '
                '(50-400)! Exiting...'.format(str(round(Eo, 2)))
                )
        return Eo
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_er_time_series(self, params_df=None):

        """Get the complete time series of modelled ecosystem respiration"""

        if not isinstance(params_df, pd.core.frame.DataFrame):
            params_df = self.estimate_parameters(mode = 'night')
        resp_series = pd.Series()
        for date in params_df.index:
            params = params_df.loc[date]
            str_date = dt.datetime.strftime(date, '%Y-%m-%d')
            data = self.df.loc[str_date, 'TC']
            resp_series = resp_series.append(Lloyd_and_Taylor
                                             (t_series=data.to_numpy(),
                                              Eo=params.Eo, rb=params.rb))
        return resp_series
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_gpp_time_series(self, params_df=None) -> pd.Series:

        """Get the complete time series of modelled gross primary production"""

        if not isinstance(params_df, pd.core.frame.DataFrame):
            params_df = self.estimate_parameters(mode='day')
        gpp_series = pd.Series()
        for date in params_df.index:
            params = params_df.loc[date]
            str_date = dt.datetime.strftime(date, '%Y-%m-%d')
            data = self.df.loc[str_date, ['PPFD', 'VPD']]
            gpp_series = gpp_series.append(rectangular_hyperbola
                                           (par_series=data.PPFD.to_numpy(),
                                            vpd_series=data.VPD.to_numpy(),
                                            alpha=params.alpha,
                                            beta=params.beta,
                                            k=params.k))
        return gpp_series
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_nee_time_series(self, params_df=None, splice_with_obs=False):

        """Get the complete time series of modelled net ecosystem exchange"""

        nee = (self.estimate_gpp_time_series(params_df) +
               self.estimate_er_time_series(params_df))
        if splice_with_obs:
            return pd.where(~np.isnan(self.df.NEE), other=nee)
        return nee
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _estimate_parameters(self, mode, window_size, window_step, Eo=None,
                             fit_daytime_rb=False):

        """Get the time series of parameter estimates for the given window
           and step"""

        if not mode in ['day', 'night']:
            raise KeyError('mode parameter must be either "day" or "night"')
        base_priors_dict = self.get_prior_parameter_estimates()
        update_priors_dict = base_priors_dict.copy()
        if not Eo:
            Eo = self.estimate_Eo()
        result_list, date_list = [], []
        print('Processing the following dates ({} mode): '.format(mode))
        date_iterator = self.make_date_iterator(window=window_size,
                                                step=window_step)
        for date in date_iterator.index:
            df = get_subset(self.df,
                            start=date_iterator.loc[date, 'Start'],
                            end=date_iterator.loc[date, 'End'])
            print((date.date()), end=' ')
            try:
                if mode == 'day':
                    result = (_fit_day_params(df, Eo, update_priors_dict,
                                              self.noct_threshold,
                                              fit_daytime_rb))
                    update_priors_dict['alpha'] = result['alpha']
                elif mode == 'night':
                    result = (
                        _fit_nocturnal_params(df, Eo, update_priors_dict,
                                              self.noct_threshold)
                        )
                result_list.append(result)
                date_list.append(date)
                print ()
            except RuntimeError as e:
                update_priors_dict['alpha'] = base_priors_dict['alpha']
                print('- {}'.format(e))
                continue
        full_date_list = np.unique(self.df.index.date)
        flag = pd.Series(0, index=date_list, name='Fill_flag')
        flag = flag.reindex(pd.date_range(full_date_list[0], full_date_list[-1],
                                          freq='D'), fill_value=1)
        out_df = pd.DataFrame(result_list, index = date_list)
        out_df = out_df.resample('D').interpolate()
        out_df = out_df.reindex(np.unique(self.df.index.date))
        out_df.fillna(method = 'bfill', inplace = True)
        out_df.fillna(method = 'ffill', inplace = True)
        return out_df.join(flag)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_nocturnal_parameters(self, Eo=None, window_size=4,
                                      window_step=4):

        priors_dict = self.get_prior_parameter_estimates()
        if not Eo:
            Eo = self.estimate_Eo()
        result_list, date_list = [], []
        print('Processing the following dates: ')
        date_iterator = self.make_date_iterator(window=window_size,
                                                step=window_step)
        for date in date_iterator.index:
            print((date.date()), end=' ')
            df = get_subset(self.df,
                            start=date_iterator.loc[date, 'Start'],
                            end=date_iterator.loc[date, 'End'])
            try:
                result_list.append(_fit_nocturnal_params(df, Eo, priors_dict,
                                                         self.noct_threshold))
                date_list.append(date)
            except RuntimeError:
                continue
            print()
        return self._reindex_results(result_list, date_list)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_dates(self):

        return self.df.index.date
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_prior_parameter_estimates(self):

        """Get dictionary with prior parameter estimates for initialisation"""

        return {'rb': self.df.loc[self.df.Fsd <
                                  self.noct_threshold, 'NEE'].mean(),
        'Eo': 100,
        'alpha': -0.01,
        'beta': (
            self.df.loc[self.df.Fsd >
                        self.noct_threshold, 'NEE'].quantile(0.03) -
            self.df.loc[self.df.Fsd >
                        self.noct_threshold, 'NEE'].quantile(0.97)
            ),
        'k': 0}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def make_date_iterator(self, window, step):

        """Create a reference dataframe containing the requisite date steps and
           corresponding window start and end"""

        start, end = self.df.index[0], self.df.index[-1]
        start_date = (
            start.to_pydatetime().date() + dt.timedelta(window / 2)
            )
        end_date = (
            end.to_pydatetime().date() - dt.timedelta(window / 2)
            )
        date_df = (
            pd.DataFrame(index=pd.date_range(start_date, end_date,
                                             freq = '{}D'.format(str(step))),
                         columns=['Start', 'End'])
            )
        for this_date in date_df.index:
            ref_date = this_date + dt.timedelta(0.5)
            date_df.loc[this_date, 'Start'] = (
                ref_date - dt.timedelta(window / 2.0 - self.interval / 1440.0)
                )
            date_df.loc[this_date, 'End'] = ref_date + dt.timedelta(window / 2.0)
        return date_df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _reindex_results(self, result_list, date_list):

        full_date_range = pd.date_range(self.df.index[0].date(),
                                        self.df.index.date[-1],
                                        freq='D')
        temp_df = pd.DataFrame(result_list, index=date_list)
        temp_df = temp_df.reindex(full_date_range)
        params_df = temp_df[temp_df.columns[:-1]].interpolate()
        params_df.fillna(method = 'bfill', inplace = True)
        params_df.fillna(method = 'ffill', inplace = True)
        flag_df = temp_df[temp_df.columns[-1]].fillna(1)
        return params_df.join(flag_df)
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _check_weights_format(weighting):

    """Check the input format of the weights supplied by the user"""

    if not isinstance(weighting, (str, list)):
        raise TypeError('"weighting" kwarg must be either string or list')
    if isinstance(weighting, str):
        if not weighting in ['air', 'soil']:
            raise TypeError('if str passed for "weighting" kwarg, it must '
                            'be either "air" or "soil"')
    if isinstance(weighting, list):
        try:
            assert len(weighting) == 2
            for x in weighting:
                assert isinstance(x, (int, float))
        except AssertionError:
            raise TypeError('if list passed for weighting kwarg, it must '
                            'conists of only 2 elements, each of which must '
                            'be of type int or float')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_internal_names():

    """Map the variable names in the internal dataset to generic variable
       references"""

    return {'Cflux': 'NEE',
            'air_temperature': 'Ta',
            'soil_temperature': 'Ts',
            'insolation': 'Fsd',
            'vapour_pressure_deficit': 'VPD',
            'PPFD': 'PPFD'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_external_names(names_dict=None):

    """Map the variable names in the external dataset to generic variable
       references, and cross-check formatting of dictionary if passed"""

    default_externals = {'Cflux': 'Fc',
                         'air_temperature': 'Ta',
                         'soil_temperature': 'Ts',
                         'insolation': 'Fsd',
                         'vapour_pressure_deficit': 'VPD',
                         'PPFD': None}

    if not names_dict:
        return default_externals
    [default_externals[x] for x in names_dict.keys()]
    return names_dict
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_fill_flags():

    """Define a schema for status flags in the params_df"""

    return {'All parameters fitted': 0,
            'All parameters interpolated': 1,
            'k_fixed_to_default': 2,
            'alpha_fixed_to_prior_or_default': 3}
#------------------------------------------------------------------------------

# #------------------------------------------------------------------------------
# def _fit_day_params(df, Eo, priors_dict, noct_threshold, fit_daytime_rb):

#     """Optimise daytime parameters (rb) for a single data window
#        (Eo must be passed as kwarg after running estimate_Eo function)"""

#     def model_fit(these_params):
#         return model.fit(day_df.NEE.to_numpy(),
#                          par_series=day_df.PPFD.to_numpy(),
#                          vpd_series=day_df.VPD.to_numpy(),
#                          t_series=day_df.TC.to_numpy(),
#                          params=these_params)

#     day_df = df.loc[df.Fsd > noct_threshold]
#     if fit_daytime_rb:
#         rb_prior = priors_dict['rb']
#     else:
#         rb_prior = _fit_nocturnal_params(df, Eo, priors_dict,
#                                          noct_threshold)['rb']
#     beta_prior = priors_dict['beta']
#     if not len(day_df) > 4:
#         raise RuntimeError('insufficient data for fit')
#     f = NEE_model
#     model = Model(f, independent_vars=['par_series', 'vpd_series', 't_series'])
#     params = model.make_params(rb=rb_prior, Eo=Eo,
#                                alpha=priors_dict['alpha'],
#                                beta=beta_prior,
#                                k=priors_dict['k'])
#     rmse_list, params_list = [], []
#     for this_beta in [beta_prior, beta_prior / 2, beta_prior * 2]:
#         params['beta'].value = this_beta
#         params['Eo'].vary = False
#         params['rb'].vary = fit_daytime_rb
#         result = model_fit(these_params=params)
#         if result.params['rb'] < 0:
#             raise RuntimeError('rb parameter out of range')
#         if not 0 <= result.params['k'].value <= 10:
#             params['k'].value = priors_dict['k']
#             params['k'].vary = False
#             result = model_fit(these_params = params)
#         if not -0.22 <= result.params['alpha'].value <= 0:
#             params['alpha'].value = priors_dict['alpha']
#             params['alpha'].vary = False
#             result = model_fit(these_params = params)
#         if not -100 <= result.params['beta'].value <= 0:
#             raise RuntimeError('beta parameter out of range')
#         rmse_list.append(
#             np.sqrt(((day_df.NEE.to_numpy() - result.best_fit)**2).sum())
#             )
#         params_list.append(result.best_values)
#     idx = rmse_list.index(min(rmse_list))
#     return params_list[idx]
# #------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _fit_day_params(df, Eo, priors_dict, noct_threshold=10, rb=None):

    """Optimise daytime parameters (rb) for a single data window
       (Eo must be passed as kwarg after running estimate_Eo function)"""

    def model_fit(these_params):
        return model.fit(day_df.NEE.to_numpy(),
                         par_series=day_df.PPFD.to_numpy(),
                         vpd_series=day_df.VPD.to_numpy(),
                         t_series=day_df.TC.to_numpy(),
                         params=these_params)

    day_df = df.loc[df.Fsd > noct_threshold]
    if rb:
        rb_prior = rb
        fit_daytime_rb = False
    else:
        rb_prior = priors_dict['rb']
        fit_daytime_rb = True
    beta_prior = priors_dict['beta']
    if not len(day_df) > 4:
        raise RuntimeError('insufficient data for fit')
    f = NEE_model
    model = Model(f, independent_vars=['par_series', 'vpd_series', 't_series'])
    params = model.make_params(rb=rb_prior, Eo=Eo,
                               alpha=priors_dict['alpha'],
                               beta=beta_prior,
                               k=priors_dict['k'])
    rmse_list, params_list = [], []
    for this_beta in [beta_prior, beta_prior / 2, beta_prior * 2]:
        params['beta'].value = this_beta
        params['Eo'].vary = False
        params['rb'].vary = fit_daytime_rb
        result = model_fit(these_params=params)
        if result.params['rb'] < 0:
            raise RuntimeError('rb parameter out of range')
        if not 0 <= result.params['k'].value <= 10:
            params['k'].value = priors_dict['k']
            params['k'].vary = False
            result = model_fit(these_params = params)
        if not -0.22 <= result.params['alpha'].value <= 0:
            params['alpha'].value = priors_dict['alpha']
            params['alpha'].vary = False
            result = model_fit(these_params = params)
        if not -100 <= result.params['beta'].value <= 0:
            raise RuntimeError('beta parameter out of range')
        rmse_list.append(
            np.sqrt(((day_df.NEE.to_numpy() - result.best_fit)**2).sum())
            )
        params_list.append(result.best_values)
    idx = rmse_list.index(min(rmse_list))
    return params_list[idx]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _fit_nocturnal_params(df, Eo, priors_dict, noct_threshold):

    """Optimise nocturnal parameters (rb) for a single data window
       (Eo must be passed as kwarg after running estimate_Eo function)"""

    noct_df = df.loc[df.Fsd < noct_threshold]
    if not len(noct_df) > 2:
        raise RuntimeError('insufficient data for fit')
    f = Lloyd_and_Taylor
    model = Model(f, independent_vars = ['t_series'])
    params = model.make_params(rb=priors_dict['rb'], Eo=Eo)
    params['Eo'].vary = False
    result = model.fit(noct_df.NEE.to_numpy(), t_series=noct_df.TC.to_numpy(),
                       params=params)
    if result.params['rb'].value < 0:
        result.best_values['rb'] = np.nan
        result.best_values.update({'nocturnal_fit_status': 1})
    else:
        result.best_values.update({'nocturnal_fit_status': 0})
    return result.best_values
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _check_continuity(index):

    """Check file continuity and return interval in integer minutes"""

    freq_dict = {'30T': 30, 'H': 60}
    interval = pd.infer_freq(index)
    if not interval in freq_dict:
        raise RuntimeError(
            'Unrecognised or non-continuous dataframe DateTime index'
            )
    return freq_dict[interval]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_subset(df, start, end):

    """Get the data subset from the complete dataset"""

    return df.loc[start: end, ['NEE', 'PPFD', 'TC', 'VPD', 'Fsd']].dropna()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_weighted_temperature(df, weighting):

    """Weight air and soil temperatures according to user input"""

    soil_T_flag = 'Ts' in df.columns
    if weighting == 'air':
        s = df.Ta.copy()
    if weighting == 'soil':
        if soil_T_flag:
            s = df.Ts.copy()
        else:
            print ('No soil temperature variable specified in input data... '
                   'defaulting to air temperature!')
            s = df.Ta.copy()
    if isinstance(weighting, list):
        if soil_T_flag:
            s = (df.Ta * weighting[0] + df.Ts * weighting[1]) / sum(weighting)
        else:
            print ('Cannot weight air and soil temperatures without soil '
                   'temperature! defaulting to air temperature!')
            s = df.Ta.copy()
    s.name = 'TC'
    return s
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def Lloyd_and_Taylor(t_series, rb, Eo):

    """Arrhenius style equation as used in Lloyd and Taylor 1994"""

    return rb  * np.exp(Eo * (1 / (10 + 46.02) - 1 / (t_series + 46.02)))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def make_formatted_df(df, variable_map, weighting):

    """Do renaming and variable conversions"""

    sub_df = df.rename(variable_map, axis='columns')
    if not 'PPFD' in variable_map:
        sub_df['PPFD'] = sub_df['Fsd'] * 0.46 * 4.6
    weighted_T_series = get_weighted_temperature(sub_df, weighting)
    return sub_df.join(weighted_T_series)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def rectangular_hyperbola(par_series, vpd_series, alpha, beta, k):

    """Rectangular hyperbola as used in Lasslop et al 2010"""

    beta_VPD = beta * np.exp(-k * (vpd_series - 1))
    index = vpd_series <= 1
    beta_VPD[index] = beta
    GPP = (alpha * par_series) / (1 - (par_series / 2000) +
           (alpha * par_series / beta_VPD))
    index = par_series < 5
    GPP[index] = 0
    return GPP
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def NEE_model(par_series, vpd_series, t_series, rb, Eo, alpha, beta, k):

    """Complete model containing both temperature and light response functions"""

    return (rectangular_hyperbola(par_series, vpd_series, alpha, beta, k) +
            Lloyd_and_Taylor(t_series, rb, Eo))
#------------------------------------------------------------------------------
