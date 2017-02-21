#!/usr/bin/env python
#########################################################################################
#
# Compute T1 map from MP2RAGE images.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Riccardo Metere, Simon Levy
# Modified: 2017-01-17
#
# About the license: see the file LICENSE.TXT
#########################################################################################
from __future__ import (division, absolute_import, print_function, unicode_literals)
from fractions import Fraction

from msct_parser import Parser
from msct_image import Image
import sys
import sct_utils as sct
import pymrt.computation as pmc
from pymrt.sequences import mp2rage


class Param:
    # Constructor
    def __init__(self):
        self.verbose = 1
        self.D_SEQ_PARAMS = {
            'eff': 1.0,  # #
            'num': 160,  # # n_gre
            'tr_gre': 7.0,  # ms
            'a_1': 4.0,  # a1 deg
            'a_2': 5.0,  # a2 deg
            # 'tr_seq': 8000.0,  # ms
            # 'ti1': 1000.0,  # ms
            # 'ti2': 3300.0,  # ms
            't_a': 440.0,  # ta ms
            't_b': 1180.0,  # tb ms
            't_c': 4140.0,  # tc ms
            }

# Acquisition parameters (default values are in the __init__ function)
class AcqParam(object):
    def __init__(self,
                 ti=(700, 1500),
                 alpha=(7.0, 5.0),
                 tr_gre=3.4,
                 tr_seq=4000.,
                 matrix_sizes=(184, 184, 160),
                 grappa_factors=(1, 2, 1),
                 grappa_refs=(0, 32, 0),
                 part_fourier_factors=(1.0, 1.0, 6 / 8),
                 bandwidths=(750,),
                 sl_pe_swap=False,
                 eff=0.95
                 ):

        self.ti = ti
        self.alpha = alpha
        self.tr_gre = tr_gre
        self.tr_seq = tr_seq
        self.matrix_sizes = matrix_sizes
        self.grappa_factors = grappa_factors
        self.grappa_refs = grappa_refs
        self.part_fourier_factors = part_fourier_factors
        self.bandwidths = bandwidths
        self.sl_pe_swap = sl_pe_swap
        self.eff = eff

    # update constructor with user's parameters
    def update(self, param_user):
        # list_params = seq_param_user.split(':')
        if len(param_user) < 2:
            sct.printv('Please check parameter -param.', 1, type='error')
        list_key_val = param_user.split('=')
        if type(getattr(self, list_key_val[0])) == tuple:
            # get type of the param
            type_param = type(getattr(self, list_key_val[0])[0])
            list_val = list_key_val[1].split(',')
            setattr(self, list_key_val[0], tuple(convert_to_proper_type(val, type_param) for val in list_val))
        else:
            # get type of the param
            type_param = type(getattr(self, list_key_val[0]))
            setattr(self, list_key_val[0], type_param(list_key_val[1]))



# main
#=======================================================================================================================
def main():

    # Initialization
    verbose = param.verbose
    default_seq_params = param.D_SEQ_PARAMS
    acq_param = AcqParam()

    # Parse input parameters
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    rho_fname = arguments['-i']
    output_fname = arguments['-o']
    t1range = arguments['-t1range']
    if "-acqparam" in arguments:
        acq_param_user = arguments['-acqparam']
        for paramKey in acq_param_user:
            acq_param.update(paramKey)


    # Load data
    rho = Image(rho_fname)

    # Output initialization
    T1map = rho.copy()
    T1map.setFileName(output_fname[0])
    RHOmap_out = rho.copy()
    RHOmap_out.setFileName(output_fname[1])

    # Define acquisition params
    # acq_params = dict(
    #     matrix_sizes=(184, 184, 160),
    #     grappa_factors=(1, 2, 1),
    #     grappa_refs=(0, 32, 0),
    #     part_fourier_factors=(1.0, 6 / 8, 6 / 8),
    #     bandwidths=None,
    #     sl_pe_swap=False,
    #     tr_seq=4000,
    #     ti=(700, 1500),
    #     alpha=(7.0, 5.0),
    #     tr_gre=4.0
    # )
    acq_params = acq_param.__dict__
    # # Determine sequence parameters from acquisition parameters
    # seq_params = acq_to_seq_params(**acq_param_kws)[0]
    # seq_param_kws = {'eff': 0.96, 'num': seq_params[0], 'tr_gre': seq_params[1], 'tr_seq': 4000., 'ti_1': 700., 'ti_2': 1500.,
    #                  'a_1': 7., 'a_2': 5.}
    # seq_param_kws = {'eff': 0.96, 'num': seq_params[0], 'tr_gre': seq_params[1], 'tr_seq': 4000., 'ti_1': 700., 'ti_2': 1500.,
    #                  'a_1': 7., 'a_2': 5., 't_a': seq_params[2], 't_b': seq_params[3], 't_c': seq_params[4]}
    # seq_param_kws = {'eff': 0.96, 'num': 160, 'tr_gre': 7.0, 'tr_seq': 4000., 'ti_1': 700., 'ti_2': 1500.,
    #               'a_1': 7., 'a_2': 5.}
    # seq_param_kws = default_seq_params
    print('Sequence parameters:\n' + str(mp2rage.acq_to_seq_params(**acq_params)))


    # Compute T1 map
    # T1map.data = t1_mp2rage(t1_value_range=(100, 1500), rho_arr=T1w.data, **seq_param_kws)
    T1map.data, RHOmap_out.data = pmc.t1_mp2rage(rho_arr=rho.data, t1_value_range=t1range, **acq_params)

    # Output T1 map
    T1map.save()
    RHOmap_out.save()

    # To view results
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview '+T1map.file_name+' '+RHOmap_out.file_name+' &\n', verbose, 'info')


# ==========================================================================================
def convert_to_proper_type(s, type_param):
    """Handle the case of fractions for partial Fourier factors."""

    try:
        return type_param(s)
    except ValueError:
        return float(Fraction(s))


# ==========================================================================================
def get_parser(acq_param=None):

    if acq_param is None:
        acq_param = AcqParam()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute T1 map from the combined MP2RAGE T1-weighted image.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="T1-weighted image (called RHO) resulting from the combination of the two MP2RAGE inversion images.",
                      mandatory=True,
                      example='uni.nii.gz')
    parser.add_option(name="-acqparam",
                      type_value=[[':'], 'str'],
                      description="Sequence parameters used. Separate values with \",\". Separate parameters with \":\".\n"
                                  "ti: <int tuple> Inversion times (in ms) for the two images. Example: ti=800,2200. Default=" + str(getattr(acq_param, 'ti')) + "\n"
                                  "alpha: <int tuple> Flip angle (in deg) for the two images. Example: alpha=7,5. Default=" + str(getattr(acq_param, 'alpha')) + "\n"
                                  "tr_seq: <float> Total repetition time (in ms) for the whole MP2RAGE sequence. Example: tr_seq=5800. Default=" + str(getattr(acq_param, 'tr_seq')) + "\n"
                                  "tr_gre: <float> Repetition time (in ms) for the GRE blocks. Example: tr_gre=6.25. Default=" + str(getattr(acq_param, 'tr_gre')) + "\n"
                                  "matrix_sizes: <int tuple> X,Y,Z dimensions of the acquisition matrix. Example: matrix_sizes=320,280,256. Default=" + str(getattr(acq_param, 'matrix_sizes')) + "\n"
                                  "grappa_factors: <int tuple> iPAT factors used in dimensions X,Y,Z. Example: grappa_factors=1,2,1. Default=" + str(getattr(acq_param, 'grappa_factors')) + "\n"
                                  "grappa_refs: <int tuple> Reference line for iPAT in dimensions X,Y,Z. Example: grappa_refs=0,32,0. Default=" + str(getattr(acq_param, 'grappa_refs')) + "\n"
                                  "part_fourier_factors: <float tuple> Partial Fourier factors in dimensions X,Y,Z. Example: part_fourier_factors=1,6/8,6/8. Default=" + str(getattr(acq_param, 'part_fourier_factors')) + "\n",
                      mandatory=False,
                      example="ti=800,2200:alpha=7,5:tr_seq=5800:tr_gre=6.25:matrix_sizes=320,280,256:grappa_factors=1,2,1:part_fourier_factors=1,6/8,6/8")
    parser.add_option(name="-t1range",
                      type_value=[[','], 'float'],
                      description="Range for T1 values (used to compute the interpolation function). ",
                      mandatory=False,
                      default_value="100,5000",
                      example="700,1500")
    parser.add_option(name="-o",
                      type_value=[",", "file_output"],
                      description="Output file names for the T1 and RHO maps separated by \",\".",
                      mandatory=False,
                      default_value="MP2RAGE_T1.nii.gz,MP2RAGE_RHO.nii.gz",
                      example="T1map.nii.gz,RHOmap.nii.gz")

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    main()
