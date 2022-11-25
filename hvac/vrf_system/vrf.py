from typing import Callable
from enum import Enum
from inspect import signature
from hvac import Quantity
from hvac.logging import ModuleLogger
from .model import VRFDataModel


Q_ = Quantity
logger = ModuleLogger.get_logger(__name__)


class WorkingMode(Enum):
    COOLING = 'cooling'
    HEATING = 'heating'


class VRFSystem:

    def __init__(
        self,
        Q_rated: Quantity,
        W_rated: Quantity,
        CR_system: float,
        Leq_pipe: Quantity,
        h_pipe: Quantity = Q_(0, 'm'),
        working_mode: WorkingMode = WorkingMode.COOLING,
        vrf_model_heating: VRFDataModel | None = None,
        vrf_model_cooling: VRFDataModel | None = None
    ):
        """
        Create VRF system.

        Parameters
        ----------
        Q_rated: Quantity
            The available (full-load) cooling/heating capacity of the system at
            rated conditions.
            If working_mode is set to WorkingMode.COOLING, Q_rated should be the
            rated cooling capacity. If working_mode is set to WorkingMode.HEATING,
            Q_rated should be the rated heating capacity.
        W_rated: Quantity
            The input power taken up by the VRV-system at rated cooling/heating
            conditions. This value must also be adapted according to the set
            working mode.
        CR_system: float
            The installed combination ratio of the VRF-system. This is the
            ratio of the installed total indoor unit capacity (i.e. the sum of
            the model sizes of the indoor units) to the rated capacity of the
            outdoor unit (also based on the model size of the outdoor unit).
        Leq_pipe: Quantity
            The maximum equivalent length of piping, i.e. the equivalent length
            (which must take the number of bends in the piping trajectory
            into account) between the outdoor unit and the farthest indoor unit.
        h_pipe: Quantity, default None
            The maximum of the heights between the outdoor unit and the indoor
            units or between indoor units.
        working_mode: WorkingMode, default WorkingMode.COOLING
            The working mode of the VRF system: either cooling, or heating.
        vrf_model_heating: VRFDataModel, default None
            The VRF model that applies to heating mode.
            The VRF model contains the functions that correct available capacity
            and input power depending on indoor air temperature, outdoor air
            temperature, equivalent piping length, combination ratio, part
            load ratio (PLR), and defrost cycli (only for heating). If working
            mode is set to heating, then the vrf_model_heating parameter must
            be set to a valid VRFDataModel object and cannot be left to None.
        vrf_model_cooling: VRFDataModel, default None
            The VRF model that applies to cooling mode. If working
            mode is set to cooling, then the vrf_model_cooling parameter must
            be set to a valid VRFDataModel object and cannot be left to None.
        """
        self.Q_rated = Q_rated
        self.W_rated = W_rated
        self.CR_system = CR_system
        self.Leq_pipe = Leq_pipe
        self.h_pipe = h_pipe
        self.working_mode = working_mode
        self.vrf_model_heating = vrf_model_heating
        self.vrf_model_cooling = vrf_model_cooling
        if (
            (self.working_mode == WorkingMode.COOLING) and
            (self.vrf_model_cooling is not None)
        ):
            self.vrf_model = self.vrf_model_cooling
        elif (
            (self.working_mode == WorkingMode.HEATING) and
            (self.vrf_model_heating is not None)
        ):
            self.vrf_model = self.vrf_model_heating
        else:
            raise TypeError('no valid VRF model attached')

    @staticmethod
    def _check_number_of_arguments(f: Callable) -> int:
        sig = signature(f)
        num_args = len(sig.parameters)
        return num_args

    def get_available_capacity(self, Tia_avg: Quantity, Toa: Quantity) -> Quantity:
        """
        Get the available cooling/heating capacity of the VRF system at
        the given (load-weighted average) indoor air temperature Tai_avg and
        outdoor air temperature Tao. In cooling mode, indoor air temperature is
        wet-bulb, and outdoor air temperature is dry-bulb. In heating mode,
        indoor air temperature is dry-bulb, and outdoor air temperature wet-bulb.
        """
        Tia_avg = Tia_avg.to(self.vrf_model.units['temperature']).m
        Toa = Toa.to(self.vrf_model.units['temperature']).m
        
        # Temperature correction
        if self._check_number_of_arguments(self.vrf_model.CAPFT_fun) == 2:
            CAPFT = self.vrf_model.CAPFT_fun(Tia_avg, Toa)
        else:
            CAPFT = self.vrf_model.CAPFT_fun(Toa)

        # Correction if rated combination ratio (ratio of total rated indoor
        # unit capacity or sum of model sizes to outdoor unit capacity or model
        # size) is greater than 100 %.
        if self.CR_system > 1:
            CR_corr = self.vrf_model.CR_corr_fun(self.CR_system)
        else:
            CR_corr = 1.0

        # Correction for equivalent piping length and piping height
        Leq_pipe_corr = 1.0
        Leq_pipe = self.Leq_pipe.to(self.vrf_model.units['length']).m
        h_pipe = self.h_pipe.to(self.vrf_model.units['height']).m

        if self.working_mode == WorkingMode.COOLING:
            if self._check_number_of_arguments(self.vrf_model.Leq_pipe_corr_fun) == 3:
                Leq_pipe_corr = self.vrf_model.Leq_pipe_corr_fun(Leq_pipe, self.CR_system, h_pipe)
            else:
                Leq_pipe_corr = self.vrf_model.Leq_pipe_corr_fun(Leq_pipe, self.CR_system)
        if self.working_mode == WorkingMode.HEATING:
            if self._check_number_of_arguments(self.vrf_model.Leq_pipe_corr_fun) == 2:
                Leq_pipe_corr = self.vrf_model.Leq_pipe_corr_fun(Leq_pipe, h_pipe)
            else:
                Leq_pipe_corr = self.vrf_model.Leq_pipe_corr_fun(Leq_pipe)

        logger.debug(f"calculate available capacity with 'Tia_avg' = {Tia_avg} and 'Toa' = {Toa}")
        logger.debug(f"CAPFT: {CAPFT}")
        logger.debug(f"CR_corr: {CR_corr}")
        logger.debug(f"Leq_pipe_corr: {Leq_pipe_corr}")

        Q_available = self.Q_rated * CAPFT * CR_corr * Leq_pipe_corr
        return Q_available

    def get_input_power(self, Tia_avg: Quantity, Toa: Quantity, PLR: float) -> Quantity:
        """
        Get the input power taken up by VRF system at the given (load-weighted
        average) indoor air temperature Tai_avg and outdoor air temperature Tao,
        when the part-load ratio equals PLR.
        """
        Tia_avg = Tia_avg.to(self.vrf_model.units['temperature']).m
        Toa = Toa.to(self.vrf_model.units['temperature']).m
        if self._check_number_of_arguments(self.vrf_model.CAPFT_fun) == 2:
            CAPFT = self.vrf_model.CAPFT_fun(Tia_avg, Toa)
        else:
            CAPFT = self.vrf_model.CAPFT_fun(Toa)
        EIRFT = self.vrf_model.EIRFT_fun(Tia_avg, Toa)
        EIRFPLR = self.vrf_model.EIRFPLR_fun(PLR)
        HPRTF = self.vrf_model.HPRTF_fun(PLR)
        logger.debug(f"calculate input power with 'Tia_avg' = {Tia_avg}, 'Toa' = {Toa}, and 'PLR' = {PLR}")
        logger.debug(f"CAPFT: {CAPFT}")
        logger.debug(f"EIRFT: {EIRFT}")
        logger.debug(f"EIRFPLR: {EIRFPLR}")
        logger.debug(f"HPRTF: {HPRTF}")
        W_input = self.W_rated * CAPFT * EIRFT * EIRFPLR * HPRTF
        return W_input
