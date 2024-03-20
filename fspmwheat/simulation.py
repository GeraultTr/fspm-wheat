# -*- coding: latin-1 -*-

import os
import numpy as np
import pandas as pd

from dataclasses import dataclass, fields

from alinea.adel.adel_dynamic import AdelDyn
from alinea.adel.echap_leaf import echap_leaves
from fspmwheat import caribu_facade
from fspmwheat import cnwheat_facade
from fspmwheat import elongwheat_facade
from fspmwheat import farquharwheat_facade
from fspmwheat import fspmwheat_facade
from fspmwheat import growthwheat_facade
from fspmwheat import senescwheat_facade



@dataclass
class WheatFSPM:
    """
    TODO : Add description
    """

    def __init__(self, meteo, inputs_dataframes, 
                 HIDDENZONES_INITIAL_STATE_FILENAME = 'hiddenzones_initial_state.csv', ELEMENTS_INITIAL_STATE_FILENAME = 'elements_initial_state.csv', 
                 AXES_INITIAL_STATE_FILENAME = 'axes_initial_state.csv', update_parameters_all_models = None, HOUR_TO_SECOND_CONVERSION_FACTOR = 3600, 
                 ORGANS_INITIAL_STATE_FILENAME = 'organs_initial_state.csv', SOILS_INITIAL_STATE_FILENAME = 'soils_initial_state.csv', INPUTS_DIRPATH='inputs', 
                 PLANT_DENSITY=None, tillers_replications=None, stored_times = None, N_fertilizations=None, heterogeneous_canopy=True, show_3Dplant = False, option_static = False, 
                 cnwheat_roots=True, UPDATE_SHARED_DF=False, 
                 CARIBU_TIMESTEP = 4, FARQUHARWHEAT_TIMESTEP = 1, ELONGWHEAT_TIMESTEP = 1, GROWTHWHEAT_TIMESTEP = 1, CNWHEAT_TIMESTEP = 1, SENESCWHEAT_TIMESTEP = 1):
        
        # SELF STORAGE FOR LOOP PARAMETERS
        self.meteo = meteo

        # time steps
        self.CARIBU_TIMESTEP = CARIBU_TIMESTEP
        self.SENESCWHEAT_TIMESTEP = SENESCWHEAT_TIMESTEP
        self.FARQUHARWHEAT_TIMESTEP = FARQUHARWHEAT_TIMESTEP
        self.ELONGWHEAT_TIMESTEP = ELONGWHEAT_TIMESTEP
        self.GROWTHWHEAT_TIMESTEP = GROWTHWHEAT_TIMESTEP
        self.CNWHEAT_TIMESTEP = CNWHEAT_TIMESTEP

        # canopy parameters
        self.PLANT_DENSITY = PLANT_DENSITY
        self.N_fertilizations = N_fertilizations

        # plant parameters
        self.tillers_replications = tillers_replications

        # logging and data structures
        self.stored_times = stored_times
        self.shared_elements_inputs_outputs_df = pd.DataFrame()
        self.shared_elements_inputs_outputs_df = pd.DataFrame()
        self.shared_hiddenzones_inputs_outputs_df = pd.DataFrame()
        self.shared_organs_inputs_outputs_df = pd.DataFrame()
        self.shared_soils_inputs_outputs_df = pd.DataFrame()
        self.shared_axes_inputs_outputs_df = pd.DataFrame()
        self.all_simulation_steps = []
        self.axes_all_data_list = []
        self.organs_all_data_list = []
        self.hiddenzones_all_data_list = []
        self.elements_all_data_list = []
        self.soils_all_data_list = []


        # boolean choices
        self.show_3Dplant = show_3Dplant
        self.heterogeneous_canopy = heterogeneous_canopy
        self.option_static = option_static

        # -- ADEL and MTG CONFIGURATION --

        # read adelwheat inputs at t0
        self.adel_wheat = AdelDyn(seed=1, scene_unit='m', leaves=echap_leaves(xy_model='Soissons_byleafclass'))
        self.g = self.adel_wheat.load(dir=INPUTS_DIRPATH)

        # ---------------------------------------------
        # ----- CONFIGURATION OF THE FACADES -------
        # ---------------------------------------------

        # -- ELONGWHEAT (created first because it is the only facade to add new metamers) --
        # Initial states
        elongwheat_hiddenzones_initial_state = inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME][
            elongwheat_facade.converter.HIDDENZONE_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.HIDDENZONE_INPUTS if i in
                                                                    inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME].columns]].copy()
        elongwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
            elongwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.ELEMENT_INPUTS if i in
                                                                    inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()
        elongwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
            elongwheat_facade.converter.AXIS_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.AXIS_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

        phytoT_df = pd.read_csv(os.path.join(INPUTS_DIRPATH, 'phytoT.csv'))

        # Update parameters if specified
        if update_parameters_all_models and 'elongwheat' in update_parameters_all_models:
            update_parameters_elongwheat = update_parameters_all_models['elongwheat']
        else:
            update_parameters_elongwheat = None

        # Facade initialisation
        self.elongwheat_facade = elongwheat_facade.ElongWheatFacade(self.g,
                                                                ELONGWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                                elongwheat_axes_initial_state,
                                                                elongwheat_hiddenzones_initial_state,
                                                                elongwheat_elements_initial_state,
                                                                self.shared_axes_inputs_outputs_df,
                                                                self.shared_hiddenzones_inputs_outputs_df,
                                                                self.shared_elements_inputs_outputs_df,
                                                                self.adel_wheat, phytoT_df,
                                                                update_parameters_elongwheat,
                                                                update_shared_df=UPDATE_SHARED_DF)

        # -- CARIBU --
        self.caribu_facade_ = caribu_facade.CaribuFacade(self.g,
                                                    self.shared_elements_inputs_outputs_df,
                                                    self.adel_wheat,
                                                    update_shared_df=UPDATE_SHARED_DF)

        # -- SENESCWHEAT --
        # Initial states    
        senescwheat_roots_initial_state = inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].loc[inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME]['organ'] == 'roots'][
            senescwheat_facade.converter.ROOTS_TOPOLOGY_COLUMNS +
            [i for i in senescwheat_facade.converter.SENESCWHEAT_ROOTS_INPUTS if i in inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].columns]].copy()

        senescwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
            senescwheat_facade.converter.ELEMENTS_TOPOLOGY_COLUMNS +
            [i for i in senescwheat_facade.converter.SENESCWHEAT_ELEMENTS_INPUTS if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

        senescwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
            senescwheat_facade.converter.AXES_TOPOLOGY_COLUMNS +
            [i for i in senescwheat_facade.converter.SENESCWHEAT_AXES_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

        # Update parameters if specified
        if update_parameters_all_models and 'senescwheat' in update_parameters_all_models:
            update_parameters_senescwheat = update_parameters_all_models['senescwheat']
        else:
            update_parameters_senescwheat = None

        # Facade initialisation
        self.senescwheat_facade_ = senescwheat_facade.SenescWheatFacade(self.g,
                                                                SENESCWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                                senescwheat_roots_initial_state,
                                                                senescwheat_axes_initial_state,
                                                                senescwheat_elements_initial_state,
                                                                self.shared_organs_inputs_outputs_df,
                                                                self.shared_axes_inputs_outputs_df,
                                                                self.shared_elements_inputs_outputs_df,
                                                                update_parameters_senescwheat,
                                                                update_shared_df=UPDATE_SHARED_DF,
                                                                cnwheat_roots=cnwheat_roots)

        # -- FARQUHARWHEAT --
        # Initial states    
        farquharwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
            farquharwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS +
            [i for i in farquharwheat_facade.converter.FARQUHARWHEAT_ELEMENTS_INPUTS if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

        farquharwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
            farquharwheat_facade.converter.AXIS_TOPOLOGY_COLUMNS +
            [i for i in farquharwheat_facade.converter.FARQUHARWHEAT_AXES_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

        # Use the initial version of the photosynthesis sub-model (as in Barillot et al. 2016, and in Gauthier et al. 2020)
        update_parameters_farquharwheat = {'SurfacicProteins': False, 'NSC_Retroinhibition': False}

        # Facade initialisation
        self.farquharwheat_facade_ = farquharwheat_facade.FarquharWheatFacade(self.g,
                                                                        farquharwheat_elements_initial_state,
                                                                        farquharwheat_axes_initial_state,
                                                                        self.shared_elements_inputs_outputs_df,
                                                                        update_parameters_farquharwheat,
                                                                        update_shared_df=UPDATE_SHARED_DF)

        # -- GROWTHWHEAT --
        # Initial states    
        growthwheat_hiddenzones_initial_state = inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME][
            growthwheat_facade.converter.HIDDENZONE_TOPOLOGY_COLUMNS +
            [i for i in growthwheat_facade.simulation.HIDDENZONE_INPUTS if i in inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME].columns]].copy()

        growthwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
            growthwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS +
            [i for i in growthwheat_facade.simulation.ELEMENT_INPUTS if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

        growthwheat_root_initial_state = inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].loc[inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME]['organ'] == 'roots'][
            growthwheat_facade.converter.ROOT_TOPOLOGY_COLUMNS +
            [i for i in growthwheat_facade.simulation.ROOT_INPUTS if i in inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].columns]].copy()

        growthwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
            growthwheat_facade.converter.AXIS_TOPOLOGY_COLUMNS +
            [i for i in growthwheat_facade.simulation.AXIS_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

        # Update parameters if specified
        if update_parameters_all_models and 'growthwheat' in update_parameters_all_models:
            update_parameters_growthwheat = update_parameters_all_models['growthwheat']
        else:
            update_parameters_growthwheat = None

        # Facade initialisation
        self.growthwheat_facade_ = growthwheat_facade.GrowthWheatFacade(self.g,
                                                                GROWTHWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                                growthwheat_hiddenzones_initial_state,
                                                                growthwheat_elements_initial_state,
                                                                growthwheat_root_initial_state,
                                                                growthwheat_axes_initial_state,
                                                                self.shared_organs_inputs_outputs_df,
                                                                self.shared_hiddenzones_inputs_outputs_df,
                                                                self.shared_elements_inputs_outputs_df,
                                                                self.shared_axes_inputs_outputs_df,
                                                                update_parameters_growthwheat,
                                                                update_shared_df=UPDATE_SHARED_DF,
                                                                cnwheat_roots=cnwheat_roots)

        # -- CNWHEAT --
        # Initial states    
        cnwheat_organs_initial_state = inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME][
            [i for i in cnwheat_facade.cnwheat_converter.ORGANS_VARIABLES if i in inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].columns]].copy()

        cnwheat_hiddenzones_initial_state = inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME][
            [i for i in cnwheat_facade.cnwheat_converter.HIDDENZONE_VARIABLES if i in inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME].columns]].copy()

        cnwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
            [i for i in cnwheat_facade.cnwheat_converter.ELEMENTS_VARIABLES if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

        cnwheat_soils_initial_state = inputs_dataframes[SOILS_INITIAL_STATE_FILENAME][
            [i for i in cnwheat_facade.cnwheat_converter.SOILS_VARIABLES if i in inputs_dataframes[SOILS_INITIAL_STATE_FILENAME].columns]].copy()

        # Update parameters if specified
        if update_parameters_all_models and 'cnwheat' in update_parameters_all_models:
            update_parameters_cnwheat = update_parameters_all_models['cnwheat']
        else:
            update_parameters_cnwheat = {}

        # Force solver separation if a different root model has been chosen
        if not cnwheat_roots:
            isolated_roots = True

        # Facade initialisation
        self.cnwheat_facade_ = cnwheat_facade.CNWheatFacade(self.g,
                                                    CNWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                    PLANT_DENSITY,
                                                    update_parameters_cnwheat,
                                                    cnwheat_organs_initial_state,
                                                    cnwheat_hiddenzones_initial_state,
                                                    cnwheat_elements_initial_state,
                                                    cnwheat_soils_initial_state,
                                                    self.shared_axes_inputs_outputs_df,
                                                    self.shared_organs_inputs_outputs_df,
                                                    self.shared_hiddenzones_inputs_outputs_df,
                                                    self.shared_elements_inputs_outputs_df,
                                                    self.shared_soils_inputs_outputs_df,
                                                    update_shared_df=UPDATE_SHARED_DF,
                                                    isolated_roots=isolated_roots,
                                                    cnwheat_roots=cnwheat_roots)

        # Run cnwheat with constant nitrates concentration in the soil if specified
        if N_fertilizations is not None and 'constant_Conc_Nitrates' in N_fertilizations.keys():
            self.cnwheat_facade_.soils[(1, 'MS')].constant_Conc_Nitrates = True  # TODO: make (1, 'MS') more general
            self.cnwheat_facade_.soils[(1, 'MS')].nitrates = N_fertilizations['constant_Conc_Nitrates'] * self.cnwheat_facade_.soils[(1, 'MS')].volume

        # -- FSPMWHEAT --
        # Facade initialisation
        self.fspmwheat_facade_ = fspmwheat_facade.FSPMWheatFacade(self.g)

        # Update geometry
        self.adel_wheat.update_geometry(self.g)
        if show_3Dplant:
            self.adel_wheat.plot(self.g)

    def __call__(self, t_caribu):
        # run Caribu
        PARi = self.meteo.loc[t_caribu, ['PARi']].iloc[0]
        DOY = self.meteo.loc[t_caribu, ['DOY']].iloc[0]
        hour = self.meteo.loc[t_caribu, ['hour']].iloc[0]
        PARi_next_hours = self.meteo.loc[range(t_caribu, t_caribu + self.CARIBU_TIMESTEP), ['PARi']].sum().values[0]

        if (t_caribu % self.CARIBU_TIMESTEP == 0) and (PARi_next_hours > 0):
            run_caribu = True
        else:
            run_caribu = False

        self.caribu_facade_.run(run_caribu, energy=PARi, DOY=DOY, hourTU=hour, latitude=48.85, sun_sky_option='sky', heterogeneous_canopy=self.heterogeneous_canopy, plant_density=self.PLANT_DENSITY[1])

        for t_senescwheat in range(t_caribu, t_caribu + self.SENESCWHEAT_TIMESTEP, self.SENESCWHEAT_TIMESTEP):
            # run SenescWheat
            self.senescwheat_facade_.run()

            # Test for dead plant # TODO: adapt in case of multiple plants
            if not self.shared_elements_inputs_outputs_df.empty and \
                    np.nansum(self.shared_elements_inputs_outputs_df.loc[self.shared_elements_inputs_outputs_df['element'].isin(['StemElement', 'LeafElement1']), 'green_area']) == 0:
                # append the inputs and outputs at current step to global lists
                self.all_simulation_steps.append(t_senescwheat)
                self.axes_all_data_list.append(self.shared_axes_inputs_outputs_df.copy())
                self.organs_all_data_list.append(self.shared_organs_inputs_outputs_df.copy())
                self.hiddenzones_all_data_list.append(self.shared_hiddenzones_inputs_outputs_df.copy())
                self.elements_all_data_list.append(self.shared_elements_inputs_outputs_df.copy())
                self.soils_all_data_list.append(self.shared_soils_inputs_outputs_df.copy())
                break

            # Run the rest of the model if the plant is alive
            for t_farquharwheat in range(t_senescwheat, t_senescwheat + self.SENESCWHEAT_TIMESTEP, self.FARQUHARWHEAT_TIMESTEP):
                # get the meteo of the current step
                Ta, ambient_CO2, RH, Ur = self.meteo.loc[t_farquharwheat, ['air_temperature', 'ambient_CO2', 'humidity', 'Wind']]

                # run FarquharWheat
                self.farquharwheat_facade_.run(Ta, ambient_CO2, RH, Ur)

                for t_elongwheat in range(t_farquharwheat, t_farquharwheat + self.FARQUHARWHEAT_TIMESTEP, self.ELONGWHEAT_TIMESTEP):
                    # run ElongWheat
                    Tair, Tsoil = self.meteo.loc[t_elongwheat, ['air_temperature', 'soil_temperature']]
                    self.elongwheat_facade_.run(Tair, Tsoil, option_static=self.option_static)

                    # Update geometry
                    self.adel_wheat.update_geometry(self.g)
                    if self.show_3Dplant:
                        self.adel_wheat.plot(self.g)

                    for t_growthwheat in range(t_elongwheat, t_elongwheat + self.ELONGWHEAT_TIMESTEP, self.GROWTHWHEAT_TIMESTEP):
                        # run GrowthWheat
                        self.growthwheat_facade_.run()

                        for t_cnwheat in range(t_growthwheat, t_growthwheat + self.GROWTHWHEAT_TIMESTEP, self.CNWHEAT_TIMESTEP):
                            print('t cnwheat is {}'.format(t_cnwheat))

                            # N fertilization if any
                            if self.N_fertilizations is not None and len(self.N_fertilizations) > 0:
                                if t_cnwheat in self.N_fertilizations.keys():
                                    self.cnwheat_facade_.soils[(1, 'MS')].nitrates += self.N_fertilizations[t_cnwheat]

                            if t_cnwheat > 0:
                                # run CNWheat
                                Tair = self.meteo.loc[t_elongwheat, 'air_temperature']
                                Tsoil = self.meteo.loc[t_elongwheat, 'soil_temperature']
                                self.cnwheat_facade_.run(Tair, Tsoil, self.tillers_replications)

                            # append outputs at current step to global lists
                            if (self.stored_times == 'all') or (t_cnwheat in self.stored_times):
                                axes_outputs, elements_outputs, hiddenzones_outputs, organs_outputs, soils_outputs = self.fspmwheat_facade_.build_outputs_df_from_MTG()

                                self.all_simulation_steps.append(t_cnwheat)
                                self.axes_all_data_list.append(axes_outputs)
                                self.organs_all_data_list.append(organs_outputs)
                                self.hiddenzones_all_data_list.append(hiddenzones_outputs)
                                self.elements_all_data_list.append(elements_outputs)
                                self.soils_all_data_list.append(soils_outputs)
