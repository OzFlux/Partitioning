#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:53:16 2018

@author: ian
"""

import datetime as dt
import matplotlib.pyplot as plt
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
        self.weighting = weights_air_soil
        if not isinstance(noct_threshold, (int, float)):
            raise TypeError('Arg "noct_threshold" must be of type int or float')
        self.noct_threshold = noct_threshold
        self.df = make_formatted_df(dataframe, self.variable_map,
                                    self.weighting)
        self.noct_threshold = noct_threshold

    #--------------------------------------------------------------------------
    # Methods
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_day_parameters(self, Eo=None, window_size=4, window_step=4,
                                fit_daytime_rb=False) -> pd.DataFrame:

        base_priors_dict = self.get_prior_parameter_estimates()
        update_priors_dict = base_priors_dict.copy()
        if not Eo:
            Eo = self.estimate_Eo()
        result_list, date_list = [], []
        print('Processing the following dates (day mode): ')
        for date in self.get_date_steps(step=window_step, window=window_size):
            print((date.date()), end=' ')
            result = (
                self.fit_day_data_window(
                    date=date, window=window_size, Eo=Eo,
                    priors_dict=update_priors_dict,
                    fit_daytime_rb=fit_daytime_rb)
                )
            if convert_fault_integer_to_bitmap(result['fault_flag'])[1]:
                update_priors_dict['alpha'] = base_priors_dict['alpha']
            else:
                update_priors_dict['alpha'] = result['alpha']
            result_list.append(result)
            date_list.append(date)
            print ()
        return self._reindex_results(result_list, date_list)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_Eo(self, window_size=15, window_step=5):

        """Estimate the activation energy type parameter for the L&T Arrhenius
           style equation using nocturnal data"""

        Eo_list = []
        priors_dict = self.get_prior_parameter_estimates()
        for date in self.get_date_steps(step=window_step, window=window_size):
            df = self.get_data_window(date=date, window=window_size)
            df = df.loc[df.Fsd < self.noct_threshold]
            if not len(df) > 6:
                continue
            if not df.TC.max() - df.TC.min() >= 5:
                continue
            f = Lloyd_and_Taylor
            model = Model(f, independent_vars = ['t_series'])
            params = model.make_params(
                rb = priors_dict['rb'], Eo = priors_dict['Eo']
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
            params_df = self.estimate_night_parameters()
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
            params_df = self.estimate_day_parameters()
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
    def estimate_night_parameters(self, Eo=None, window_size=4,
                                      window_step=4):

        priors_dict = self.get_prior_parameter_estimates()
        if not Eo:
            Eo = self.estimate_Eo()
        result_list, date_list = [], []
        print('Processing the following dates: ')
        for date in self.get_date_steps(step=window_step, window=window_size):
            print((date.date()), end=' ')
            result_list.append(
                self.fit_night_data_window(date=date, window=window_size,
                                           Eo=Eo, priors_dict=priors_dict)
                )
            date_list.append(date)
            print()
        return self._reindex_results(result_list, date_list)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def fit_day_data_window(self, date, window, priors_dict=None, Eo=None,
                            fit_daytime_rb=False):

        """Get the daytime parameter fit for a specific date and window"""

        if not priors_dict:
            priors_dict = self.get_prior_parameter_estimates()
        if not Eo:
            Eo = self.estimate_Eo()
        df = self.get_data_window(date=date, window=window)
        if not fit_daytime_rb:
            noct_params = (
                _fit_night_params(df, Eo, priors_dict, self.noct_threshold)
                )
            if convert_fault_integer_to_bitmap(noct_params['fault_flag'])[1]:
                return noct_params
            rb=noct_params['rb']
        else:
            rb=None
        return _fit_day_params(df, Eo, priors_dict, self.noct_threshold,rb=rb)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def fit_night_data_window(self, date, window, priors_dict=None, Eo=None):

        """Get the nocturnal parameter fit for a specific date and window"""

        if not priors_dict:
            priors_dict = self.get_prior_parameter_estimates()
        if not Eo:
            Eo = self.estimate_Eo()
        df = self.get_data_window(date=date, window=window)
        return _fit_night_params(df, Eo, priors_dict, self.noct_threshold)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_date_steps(self, step, window) -> np.ndarray:

        """Get the stepped date index over which the fit function iterates"""

        start, end = self.df.index[0], self.df.index[-1]
        start_date = start.to_pydatetime().date() + dt.timedelta(window / 2)
        end_date = end.to_pydatetime().date() - dt.timedelta(window / 2)
        return (
            pd.date_range(start_date, end_date, freq='{}D'.format(str(step)))
            .to_pydatetime()
            )
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_data_window(self, date, window, dates_only=False) -> dict:

        """Get the relevant date range centred on a given date for a given
           window (in days); if not dates_only, get the (fit-relevant)
           data itself"""

        if not isinstance(date, dt.date):
            if not isinstance(date, str):
                raise TypeError('Format must be either python date or str')
            date = dt.datetime.strptime(date, '%Y-%m-%d').date()
        if not self.df.index[0] <= date <= self.df.index[-1]:
            raise RuntimeError('Date outside range of available data')
        ref_date = dt.datetime.combine(date, dt.time(12))
        start = ref_date - dt.timedelta(window / 2.0 - self.interval / 1440.0)
        end = ref_date + dt.timedelta(window / 2.0)
        if dates_only:
            return start, end
        return (
            self.df.loc[start: end, ['NEE', 'PPFD', 'TC', 'VPD', 'Fsd']]
            .dropna()
            )
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_prior_parameter_estimates(self) -> dict:

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
    def plot_er(self, date, window=15, Eo=None, fit_daytime_rb=False):

        if not fit_daytime_rb:
            params_dict = (
                self.fit_night_data_window(date=date, window=window, Eo=Eo)
                )
        else:
            params_dict = (
                self.fit_day_data_window(date=date, window=window, Eo=Eo,
                                         fit_daytime_rb=fit_daytime_rb)
                )
        error_state, fatal = convert_fault_integer_to_bitmap(
                fault_integer=params_dict['fault_flag'], as_bin=False
                )
        if fatal:
            print('Fitting failed with the following error: {}'
                  .format(error_state))
            return
        df = self.get_data_window(date=date, window=window)
        df = df.loc[df.Fsd < self.noct_threshold]
        fig, ax = plt.subplots(1, 1, figsize = (14, 8))
        fig.patch.set_facecolor('white')
        ax.axhline(0, color = 'black')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis = 'y', labelsize = 14)
        ax.tick_params(axis = 'x', labelsize = 14)
        ax.set_title(date, fontsize = 18)
        ax.set_xlabel('$Temperature\/(^oC)$', fontsize = 18)
        ax.set_ylabel('$NEE\/(\mu molC\/m^{-2}\/s^{-1})$', fontsize = 18)
        ax.plot(df.TC, df.NEE, color = 'None', marker = 'o',
                mfc = 'grey', mec = 'black', ms = 8, alpha = 0.5,
                label = 'Observations')
        df['TC_alt'] = np.linspace(df.TC.min(), df.TC.max(), len(df))
        synth_series = Lloyd_and_Taylor(t_series=df.TC_alt,
                                        rb=params_dict['rb'],
                                        Eo=params_dict['Eo'])
        ax.plot(df.TC_alt, synth_series, color = 'black', label='Model')
        ax.legend(loc = [0.05, 0.8], fontsize = 12)
        return fig
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def plot_nee(self, date, window=15, Eo=None, fit_daytime_rb=False):

        params_dict = (
            self.fit_day_data_window(date=date, window=window, Eo=Eo,
                                     fit_daytime_rb=fit_daytime_rb)
            )
        error_state, fatal = convert_fault_integer_to_bitmap(
                fault_integer=params_dict['fault_flag'], as_bin=False
                )
        if fatal:
            print('Fitting failed with the following error: {}'
                  .format(error_state))
            return
        error_str = error_state if error_state else 'none'
        error_str = 'Diagnostic state: {}'.format(error_str)
        df = self.get_data_window(date=date, window=window)
        df = df.loc[df.Fsd > self.noct_threshold]
        fig, ax = plt.subplots(1, 1, figsize = (14, 8))
        fig.patch.set_facecolor('white')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlim([0, df.PPFD.max() * 1.05])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis = 'y', labelsize = 14)
        ax.tick_params(axis = 'x', labelsize = 14)
        ax.set_title(date, fontsize = 18)
        ax.set_xlabel('$PPFD\/(\mu mol\/photons\/m^{-2}\/s^{-1})$',
                      fontsize = 18)
        ax.set_ylabel('$NEE\/(\mu molC\/m^{-2}\/s^{-1})$', fontsize = 18)
        im=ax.scatter(df.PPFD, df.NEE, marker='o', s=80, c=df.VPD, alpha=0.5,
                   label='Observations')
        df['PPFD_alt'] = np.linspace(0, df.PPFD.max(), len(df))
        synth_series = (
            NEE_model(par_series=df.PPFD, vpd_series=df.VPD,
                      t_series=df.TC, rb = params_dict['rb'],
                      Eo=params_dict['Eo'], alpha=params_dict['alpha'],
                      beta=params_dict['beta'], k=params_dict['k'])
            )
        ax.plot(df.PPFD, synth_series, color='None', marker='+',
                mec='black', ms=8, alpha=0.5, label='Model')
        ax.legend(loc=[0.05, 0.1], fontsize=12)
        ax.text(0.5, 0.95, error_str, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes,
                fontsize=12)
        fig.colorbar(im, ax=ax)
        return fig
    # #--------------------------------------------------------------------------

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
                            'consist of only 2 elements, each of which must '
                            'be of type int or float')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def convert_fault_integer_to_bitmap(fault_integer, as_bin=True):

    """Convert fault integer to bitmap or text"""

    fatal = True if np.bitwise_and(fault_integer, 7) else False
    if as_bin: return bin(fault_integer)[2:].zfill(5), fatal
    fault_dict = define_fault_flags()
    inverse_dict = dict(zip(fault_dict.values(), fault_dict.keys()))
    fault_list = []
    for this_bit in inverse_dict:
        try:
            bit_as_int = np.bitwise_and(this_bit, fault_integer)
            fault_list.append(inverse_dict[bit_as_int])
        except KeyError:
            continue
    return ', '.join(fault_list), fatal
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
def define_fault_flags(key=None):

    """Define a schema for diagnostic flags in the params_df"""

    def_dict = {'insufficient data for fit': 2**0,
                'rb out of range': 2**1,
                'beta out of range': 2**2,
                'k fixed to default': 2**3,
                'alpha fixed to prior or default': 2**4}

    if key in def_dict:
        return def_dict[key]
    return def_dict
#------------------------------------------------------------------------------

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

    fail_dict = {'rb': np.nan, 'Eo': Eo, 'alpha': np.nan, 'beta': np.nan,
                 'k': np.nan}
    day_df = df.loc[df.Fsd > noct_threshold]
    if rb:
        rb_prior = rb
        fit_daytime_rb = False
    else:
        rb_prior = priors_dict['rb']
        fit_daytime_rb = True
    beta_prior = priors_dict['beta']
    if not len(day_df) > 4:
        fail_dict.update(
            {'fault_flag': define_fault_flags('insufficient data for fit')}
            )
        return fail_dict
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
        fault_flag = 0
        result = model_fit(these_params=params)
        if result.params['rb'] < 0:
            fault_flag = define_fault_flags('rb out of range')
            continue
        if not 0 <= result.params['k'].value <= 10:
            params['k'].value = priors_dict['k']
            params['k'].vary = False
            result = model_fit(these_params=params)
            fault_flag += define_fault_flags('k fixed to default')
        if not -0.22 <= result.params['alpha'].value <= 0:
            params['alpha'].value = priors_dict['alpha']
            params['alpha'].vary = False
            result = model_fit(these_params=params)
            fault_flag += define_fault_flags('alpha fixed to prior or default')
        if not -100 <= result.params['beta'].value <= 0:
            fault_flag += define_fault_flags('beta out of range')
            continue
        rmse_list.append(
            np.sqrt(((day_df.NEE.to_numpy() - result.best_fit)**2).sum())
            )
        result.best_values.update({'fault_flag': fault_flag})
        params_list.append(result.best_values)
    if not rmse_list:
        fail_dict.update({'fault_flag': fault_flag})
        return fail_dict
    idx = rmse_list.index(min(rmse_list))
    return params_list[idx]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _fit_night_params(df, Eo, priors_dict, noct_threshold=10):

    """Optimise nocturnal parameters (rb) for a single data window
       (Eo must be passed as kwarg after running estimate_Eo function)"""

    fail_dict = {'rb': np.nan, 'Eo': Eo}
    noct_df = df.loc[df.Fsd < noct_threshold]
    if not len(noct_df) > 2:
        fail_dict.update(
            {'fault_flag':
             define_fault_flags('insufficient data for fit')}
            )
        return fail_dict
    model = Model(Lloyd_and_Taylor, independent_vars=['t_series'])
    params = model.make_params(rb=priors_dict['rb'], Eo=Eo)
    params['Eo'].vary = False
    result = model.fit(
        noct_df.NEE.to_numpy(), t_series=noct_df.TC.to_numpy(), params=params
        )
    if result.params['rb'].value < 0:
        fail_dict.update(
            {'fault_flag': define_fault_flags('rb out of range')}
            )
        return fail_dict
    result.best_values.update({'fault_flag': 0})
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
