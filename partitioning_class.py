#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:53:16 2018

@author: ian
"""

import datetime as dt
from lmfit import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

#------------------------------------------------------------------------------
### CONSTANTS ###
#------------------------------------------------------------------------------

noct_threshold = 10

#------------------------------------------------------------------------------
# Init
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
                 fit_daytime_rb=False):

        interval = int(''.join([x for x in pd.infer_freq(dataframe.index)
                                if x.isdigit()]))
        assert interval % 30 == 0
        self.interval = interval
        self.variable_map = get_variable_map(names_dict)
        self.weighting = _check_weights_format(weights_air_soil)
        self.df = make_formatted_df(dataframe, self.variable_map,
                                    self.weighting)
        self._fit_daytime_rb = fit_daytime_rb
#------------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Methods
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_Eo(self, window_size=15, window_step=5):

        """Estimate the activation energy type parameter for the L&T Arrhenius
           style equation using nocturnal data"""

        Eo_list = []
        date_iterator = make_date_iterator(self.df, window_size, window_step,
                                           self.interval)
        for date in date_iterator.index:
            df = get_subset(self.df,
                            start=date_iterator.loc[date, 'Start'],
                            end=date_iterator.loc[date, 'End'])
            df = df.loc[df.Fsd < noct_threshold]
            if not len(df) > 6: continue
            if not df.TC.max() - df.TC.min() >= 5: continue
            f = Lloyd_and_Taylor
            model = Model(f, independent_vars = ['t_series'])
            params = model.make_params(rb = 1,
                                       Eo = _get_prior_parameter_estimates(df)['Eo'])
            result = model.fit(df.NEE,
                               t_series = df.TC,
                               params = params)
            if not 50 < result.params['Eo'].value < 400: continue
            if result.params['Eo'].stderr > result.params['Eo'].value / 2.0:
                continue
            Eo_list.append([result.params['Eo'].value,
                            result.params['Eo'].stderr])
        if len(Eo_list) == 0: raise RuntimeError('Could not find any valid '
                                                 'estimates of Eo! Exiting...')
        print('Found {} valid estimates of Eo'.format(str(len(Eo_list))))
        Eo_array = np.array(Eo_list)
        Eo = ((Eo_array[:, 0] / (Eo_array[:, 1])).sum() /
              (1 / Eo_array[:, 1]).sum())
        if not 50 < Eo < 400: raise RuntimeError('Eo value {} outside '
                                                 'acceptable parameter range '
                                                 '(50-400)! Exiting...'
                                                 .format(str(round(Eo, 2))))
        return Eo
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_er_time_series(self, params_df=False):

        if not isinstance(params_df, pd.core.frame.DataFrame):
            params_df = self.estimate_parameters(mode = 'night')
        resp_series = pd.Series()
        for date in params_df.index:
            params = params_df.loc[date]
            str_date = dt.datetime.strftime(date, '%Y-%m-%d')
            data = self.df.loc[str_date, 'TC']
            resp_series = resp_series.append(Lloyd_and_Taylor
                                             (t_series = data,
                                              Eo = params.Eo, rb = params.rb))
        return resp_series
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_gpp_time_series(self, params_df=False):

        if not isinstance(params_df, pd.core.frame.DataFrame):
            params_df = self.estimate_parameters(mode = 'day')
        gpp_series = pd.Series()
        for date in params_df.index:
            params = params_df.loc[date]
            str_date = dt.datetime.strftime(date, '%Y-%m-%d')
            data = self.df.loc[str_date, ['PPFD', 'VPD']]
            gpp_series = gpp_series.append(rectangular_hyperbola
                                           (par_series = data.PPFD,
                                            vpd_series = data.VPD,
                                            alpha = params.alpha,
                                            beta = params.beta,
                                            k = params.k))
        return gpp_series
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_nee_time_series(self, params_df=False, splice_with_obs=False):
        return (self.estimate_gpp_time_series(params_df) +
                self.estimate_er_time_series(params_df))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_parameters(self, mode, Eo=None, window_size=4, window_step=4):

        base_priors_dict = self.get_prior_parameter_estimates()
        update_priors_dict = base_priors_dict.copy()
        if not Eo: Eo = self.estimate_Eo()
        result_list, date_list = [], []
        print('Processing the following dates ({} mode): '.format(mode))
        date_iterator = make_date_iterator(self.df, window_size, window_step,
                                           self.interval)
        for date in date_iterator.index:
            df = get_subset(self.df,
                            start=date_iterator.loc[date, 'Start'],
                            end=date_iterator.loc[date, 'End'])
            print((date.date()), end=' ')
            try:
                if mode == 'day': result = (
                    _fit_day_params(df, Eo, update_priors_dict,
                                                self._fit_daytime_rb)
                    )
                elif mode == 'night': result = (
                        _fit_nocturnal_params(df, Eo, update_priors_dict)
                        )
                result_list.append(result)
                date_list.append(date)
                update_priors_dict['alpha'] = result['alpha']
                print ()
            except RuntimeError as e:
                update_priors_dict['alpha'] = base_priors_dict['alpha']
                print('- {}'.format(e))
                continue
        full_date_list = np.unique(self.df.index.date)
        flag = pd.Series(0, index = date_list, name = 'Fill_flag')
        flag = flag.reindex(pd.date_range(full_date_list[0], full_date_list[-1],
                                          freq='D'))
        flag.fillna(1, inplace = True)
        out_df = pd.DataFrame(result_list, index = date_list)
        out_df = out_df.resample('D').interpolate()
        out_df = out_df.reindex(np.unique(self.df.index.date))
        out_df.fillna(method = 'bfill', inplace = True)
        out_df.fillna(method = 'ffill', inplace = True)
        return out_df.join(flag)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def plot_er(self, date_start, date_end, Eo=None):

        df = get_subset(self.df, start=date_start, end=date_end)
        assert len(df) > 0
        if not Eo: Eo = self.estimate_Eo()
        results_dict = {}
        try:
            results_dict['night'] = (
                _fit_nocturnal_params(
                    df, Eo, self.get_prior_parameter_estimates())['rb']
                )
        except RuntimeError as e:
            print('Fit of nocturnal rb failed with the following message {}'
                  .format(e))
        try:
            self._fit_daytime_rb = True
            results_dict['day'] = (
                    _fit_day_params(
                        df, Eo, self.get_prior_parameter_estimates(),
                        self._fit_daytime_rb)['rb']
                    )
        except RuntimeError as e:
            print('Fit of daytime rb failed with the following message {}'
                  .format(e))
        df = df.loc[df.Fsd < noct_threshold]
        fig, ax = plt.subplots(1, 1, figsize = (14, 8))
        fig.patch.set_facecolor('white')
        ax.axhline(0, color = 'black')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis = 'y', labelsize = 14)
        ax.tick_params(axis = 'x', labelsize = 14)
        # ax.set_title(dt.datetime.strftime(date, '%Y-%m-%d'), fontsize = 18)
        ax.set_xlabel('$Temperature\/(^oC)$', fontsize = 18)
        ax.set_ylabel('$NEE\/(\mu molC\/m^{-2}\/s^{-1})$', fontsize = 18)
        labels_dict = {'night': 'Night Eo and rb', 'day': 'Night Eo, day rb'}
        styles_dict = {'night': '--', 'day': ':'}
        ax.plot(df.TC, df.NEE, color = 'None', marker = 'o',
                mfc = 'grey', mec = 'black', ms = 8, alpha = 0.5,
                label = 'Observations')
        df['TC_alt'] = np.linspace(df.TC.min(), df.TC.max(), len(df))
        for key in list(results_dict.keys()):
            s = Lloyd_and_Taylor(t_series=df.TC_alt, rb=results_dict[key],
                                 Eo=Eo)
            ax.plot(df.TC_alt, s, color = 'black', ls = styles_dict[key],
                    label = labels_dict[key])
        ax.legend(loc = [0.05, 0.8], fontsize = 12)
        return fig
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def plot_nee(self, date, window_size = 15, Eo = None):

        state = self._fit_daytime_rb
        df = self.get_subset(date, size = window_size, mode = 'day')
        assert len(df) > 0
        if not Eo: Eo = self.estimate_Eo()
        results_dict = {}
        try:
            self._fit_daytime_rb = False
            results_dict['night'] = (self._day_params(date, Eo, window_size,
                                      self.prior_parameter_estimates()))
        except RuntimeError as e:
            print('Fit of daytime parameters and nocturnal rb failed with '
                  'the following message {}'.format(e))
        try:
            self._fit_daytime_rb = True
            results_dict['day'] = (self._day_params(date, Eo, window_size,
                                    self.prior_parameter_estimates()))
        except RuntimeError as e:
            print('Fit of daytime parameters and rb failed with the '
                  'following message {}'.format(e))
        self._fit_daytime_rb = state
        fig, ax = plt.subplots(1, 1, figsize = (14, 8))
        fig.patch.set_facecolor('white')
        ax.axhline(0, color = 'black')
        ax.set_xlim([0, df.PPFD.max() * 1.05])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis = 'y', labelsize = 14)
        ax.tick_params(axis = 'x', labelsize = 14)
        ax.set_title(dt.datetime.strftime(date, '%Y-%m-%d'), fontsize = 18)
        ax.set_xlabel('$PPFD\/(\mu mol\/photons\/m^{-2}\/s^{-1})$',
                      fontsize = 18)
        ax.set_ylabel('$NEE\/(\mu molC\/m^{-2}\/s^{-1})$', fontsize = 18)
        labels_dict = {'night': 'Night Eo and rb', 'day': 'Night Eo, day rb'}
        markers_dict = {'night': '+', 'day': 'x'}
        colors_dict = {'night': 'blue', 'day': 'magenta'}
        ax.plot(df.PPFD, df.NEE, color = 'None', marker = 'o',
                mfc = 'grey', mec = 'black', ms = 8, alpha = 0.5,
                label = 'Observations')
        for key in list(results_dict.keys()):
            params = results_dict[key]
            s = NEE_model(par_series=df.PPFD, vpd_series=df.VPD,
                            t_series=df.TC, rb = params['rb'],
                            Eo = params['Eo'], alpha = params['alpha'],
                            beta = params['beta'], k = params['k'])
            ax.plot(df.PPFD, s, color = colors_dict[key],
                    marker = markers_dict[key], label = labels_dict[key],
                    ls = 'None')
        ax.legend(loc = [0.05, 0.1], fontsize = 12)
        return fig
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_prior_parameter_estimates(self):

        return _get_prior_parameter_estimates(self.df)
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _check_weights_format(weighting):

    """Check the input format of the weights supplied by the user"""

    try:
        assert isinstance(weighting, (str, list))
    except AssertionError:
        raise TypeError('"weighting" kwarg must be either string or list')
    try:
        if isinstance(weighting, str):
            assert weighting in ['air', 'soil']
            return weighting
    except AssertionError:
        raise TypeError('if str passed for "weighting" kwarg, it must '
                           'be either "air" or "soil"')
    try:
        if isinstance(weighting, list):
            assert len(weighting) == 2
            for x in weighting:
                assert isinstance(x, (int, float))
            return weighting
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
def _define_default_external_names():

    """Map the variable names in the external dataset to generic variable
       references"""

    return {'Cflux': 'Fc',
            'air_temperature': 'Ta',
            'soil_temperature': 'Ts',
            'insolation': 'Fsd',
            'vapour_pressure_deficit': 'VPD',
            'PPFD': None}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _fit_day_params(df, Eo, priors_dict, fit_daytime_rb):

    """Optimise daytime parameters (rb) for a single data window
       (Eo must be passed as kwarg after running estimate_Eo function)"""

    def model_fit(these_params):
        return model.fit(day_df.NEE, par_series=day_df.PPFD,
                         vpd_series=day_df.VPD, t_series=day_df.TC,
                         params=these_params)

    day_df = df.loc[df.Fsd > noct_threshold]
    if fit_daytime_rb: rb_prior = priors_dict['rb']
    else: rb_prior = _fit_nocturnal_params(df, Eo, priors_dict)['rb']
    beta_prior = priors_dict['beta']
    if not len(df) > 4:
        raise RuntimeError('insufficient data for fit')
    f = NEE_model
    model = Model(f, independent_vars = ['par_series', 'vpd_series',
                                         't_series'])
    params = model.make_params(rb = rb_prior, Eo = Eo,
                               alpha = priors_dict['alpha'],
                               beta = beta_prior,
                               k = priors_dict['k'])
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
        rmse_list.append(np.sqrt(((df.NEE - result.best_fit)**2).sum()))
        params_list.append(result.best_values)
    idx = rmse_list.index(min(rmse_list))
    return params_list[idx]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _fit_nocturnal_params(df, Eo, priors_dict):

    """Optimise nocturnal parameters (rb) for a single data window
       (Eo must be passed as kwarg after running estimate_Eo function)"""

    noct_df = df.loc[df.Fsd < noct_threshold]
    if not len(noct_df) > 2: raise RuntimeError('insufficient data for fit')
    f = Lloyd_and_Taylor
    model = Model(f, independent_vars = ['t_series'])
    params = model.make_params(rb = priors_dict['rb'], Eo = Eo)
    params['Eo'].vary = False
    result = model.fit(noct_df.NEE, t_series = noct_df.TC, params = params)
    if result.params['rb'].value < 0: raise RuntimeError('rb parameter '
                                                         'out of range')
    return result.best_values
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_prior_parameter_estimates(df):

    """Get initial parameter estimates"""

    return {'rb': df.loc[df.Fsd < noct_threshold, 'NEE'].mean(),
            'Eo': 100,
            'alpha': -0.01,
            'beta': (df.loc[df.Fsd > noct_threshold, 'NEE'].quantile(0.03) -
                     df.loc[df.Fsd > noct_threshold, 'NEE'].quantile(0.97)),
            'k': 0}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_variable_map(external_names=None):

    """Convert external names to internal names"""

    internal_names = _define_default_internal_names()
    if not external_names: external_names = _define_default_external_names()
    return {external_names[key]: internal_names[key] for key in internal_names}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_subset(df, start, end):

    """Get the data subset from the complete dataset"""

    return df.loc[start: end, ['NEE', 'PPFD', 'TC', 'VPD', 'Fsd']].dropna()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_weighted_temperature(df, weighting):

    """Weight air and soil temperatures according to user input"""

    soil_T_flag = True if 'Ts' in df.columns else False
    if weighting == 'air': s = df.Ta.copy()
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
def make_date_iterator(df, size, step, interval):

    """Create a reference dataframe containing the requisite date steps and
       corresponding window start and end"""

    start_date = (df.index[0].to_pydatetime().date() +
                  dt.timedelta(size / 2))
    end_date = (df.index[-1].to_pydatetime().date() -
                dt.timedelta(size / 2))
    date_df = (
        pd.DataFrame(index=pd.date_range(start_date, end_date,
                                         freq = '{}D'.format(str(step))),
                     columns=['Start', 'End'])
        )
    for this_date in date_df.index:
        ref_date = this_date + dt.timedelta(0.5)
        date_df.loc[this_date, 'Start'] = (
            ref_date - dt.timedelta(size / 2.0 - interval / 1440.0)
            )
        date_df.loc[this_date, 'End'] = ref_date + dt.timedelta(size / 2.0)
    return date_df
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