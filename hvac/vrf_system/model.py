from typing import List, Callable, Optional, Dict
from scipy import interpolate
import dill as pickle
import pandas as pd
from hvac.logging import ModuleLogger

logger = ModuleLogger.get_logger(__name__)


def get_Tia_avg(
    Ql_zones: List[float | int],
    Tia_zones: List[float | int]
) -> float:
    """
    Get the load-weighted average indoor air temperature.

    Parameters
    ----------
    Ql_zones: List[float | int]
        A list with the zone loads.
    Tia_zones: List[float | int]
        A list with the indoor air temperatures of the zones (wet-bulb
        when cooling, dry-bulb when heating).

    Returns
    -------
    The load-weighted average indoor air temperature (wet-bulb
    when cooling, dry-bulb when heating).
    """
    Ql_tot = sum(Ql_zones)
    Tia_avg = sum(
        Tia * Ql_zone
        for Tia, Ql_zone in zip(Tia_zones, Ql_zones)
    ) / Ql_tot
    return Tia_avg


def get_CR_system(
    Qiu_rated_list: List[float | int],
    Qou_rated: float | int
) -> float:
    """
    Get the combination ratio (model size ratio) of the VRF system, i.e. the ratio
    of the sum of the rated capacities (or model sizes) of the indoor units in the
    VRF system to the rated total indoor unit capacity (or model size) assigned
    to the outdoor unit.

    Parameters
    ----------
    Qiu_rated_list: List[float | int]
        A list of the rated capacities of the indoor units in the VRF system
        (based on indoor unit model size).
    Qou_rated: float | int
        The rated total indoor unit capacity assigned to the outdoor unit
        (based on the outdoor unit model size).

    Returns
    -------
    The combination ratio (or model size ratio) of the VRF-system.
    """
    Q_iu_tot = sum(Qiu_rated_list)
    CR_system = Q_iu_tot / Qou_rated
    return CR_system


class VRFDataModel:
    default_units = {
        'temperature': 'degC',
        'length': 'm',
        'height': 'm'
    }

    def __init__(self, units: Dict[str, str] | None = None):
        """
        Create VRF data model. This object contains the functions that a
        VRFSystem object will use to correct available capacity and input power
        depending on temperature and part-load conditions, and also on
        installation conditions (combination ratio, equivalent piping length,
        piping height).
        A separate model has to be created for cooling and heating using the
        manufacturer's data.

        Parameters
        ----------
        units: Dict[str, str], optional
            Most input data to the model will be dimensionless. Only temperature,
            piping length, and piping height are input quantities to the model.
            The default units for temperature, length, and height are degrees
            Celsius, meter, and meter respectively. If this would not be the
            case, parameter `units` must be given a dictionary with keys
            'temperature', 'length', and 'height' together with their
            appropriate measuring unit.
        """
        self.units = self.default_units if units is None else units
        self.CAPFT_fun: Optional[Callable] = None
        self.EIRFT_fun: Optional[Callable] = None
        self.EIRFPLR_fun: Optional[Callable] = None
        self.Leq_pipe_corr_fun: Optional[Callable] = None
        self.CR_corr_fun: Optional[Callable] = None
        self.HPRTF_fun: Optional[Callable] = None

    def set_CAPFT_fun(self, CAP_data: pd.Series | pd.DataFrame) -> None:
        """
        Set the function that returns the capacity ratio of the outdoor unit for a
        given indoor air temperature and a given outdoor air temperature (i.e. the
        ratio of available full-load capacity at the given temperatures to the
        rated full-load capacity).

        Parameters
        ----------
        CAP_data: pd.Series | pd.DataFrame
            - A Pandas Series object in case the capacity ratio should be independent
            of indoor air temperatures. In that case the index is composed of the
            outdoor air temperatures and the values are the corresponding capacity
            ratios.
            - A Pandas DataFrame object if the capacity ratio is also dependent of
            indoor air temperatures. In that case the index is composed of the indoor
            air temperatures and the columns are the outdoor air temperatures.

        Notes
        -----
        If a Pandas DataFrame object was passed in, CAPFT_fun is a function object
        with calling signature CAPFT_fun(Tia, Toa), wherein Tia is the indoor air
        temperature and Toa is the outdoor air temperature, that returns the
        capacity ratio that corresponds with the two given temperatures.

        If a Pandas Series object was passed in, CAPFT_fun is a function object
        with calling signature CAPFT_fun(Toa), wherein Toa is the outdoor air
        temperature, that returns the capacity ratio that corresponds with this
        outdoor air temperature.
        """
        if isinstance(CAP_data, pd.DataFrame):
            CAP_data = CAP_data.transpose()
            CAPFT_fun = interpolate.interp2d(
                x=CAP_data.columns,
                y=CAP_data.index,
                z=CAP_data.to_numpy()
            )
            Tia_min = min(CAP_data.columns[0], CAP_data.columns[-1])
            Tia_max = max(CAP_data.columns[0], CAP_data.columns[-1])
            Toa_min = min(CAP_data.index[0], CAP_data.index[-1])
            Toa_max = max(CAP_data.index[0], CAP_data.index[-1])
            
            def f(Tia: float, Toa: float) -> float:
                if not (Tia_min <= Tia <= Tia_max):
                    logger.warning(f"Indoor air temperature {Tia} outside interpolation domain")
                if not (Toa_min <= Toa <= Toa_max):
                    logger.warning(f"Outdoor air temperature {Toa} outside interpolation domain")
                return CAPFT_fun(Tia, Toa)[0]

            self.CAPFT_fun = f

        else:
            CAPFT_fun = interpolate.interp1d(
                x=CAP_data.index,
                y=CAP_data.values,
                bounds_error=False,
                fill_value='extrapolate'
            )
            Toa_min = min(CAP_data.index[0], CAP_data.index[-1])
            Toa_max = max(CAP_data.index[0], CAP_data.index[-1])
            
            def f(Toa: float) -> float:
                if not (Toa_min <= Toa <= Toa_max):
                    logger.warning(f"Outdoor air temperature {Toa} outside interpolation domain")
                return CAPFT_fun(Toa)

            self.CAPFT_fun = f

    def set_EIRFT_fun(self, EIR_data: pd.DataFrame) -> None:
        """
        Set the function that returns the ratio of the energy input ratio (EIR) for
        a given indoor air temperature and a given outdoor air temperature to the
        rated energy input ratio (EIR_rated) (i.e. the ratio of input power at
        rated full-load capacity to this rated full-load capacity).

        Parameters
        ----------
        EIR_data: Pandas DataFrame object
            A Pandas DataFrame with columns the outdoor air temperatures, with
            index the indoor air temperatures, and with data the EIR ratios
            derived from the manufacturer's data (EIR-ratio = input-power-ratio /
            capacity-ratio)

        Notes
        -----
        EIRFT_fun is a function object with calling signature EIRFT_fun(Tia, Toa),
        wherein Tia is the indoor air temperature and Toa is the outdoor air
        temperature, that returns the ratio EIR to EIR_rated that corresponds
        with the given temperatures.
        """
        EIR_data = EIR_data.transpose()
        EIRFT_fun = interpolate.interp2d(
            x=EIR_data.columns,
            y=EIR_data.index,
            z=EIR_data.to_numpy(),
        )
        Tia_min = min(EIR_data.columns[0], EIR_data.columns[-1])
        Tia_max = max(EIR_data.columns[0], EIR_data.columns[-1])
        Toa_min = min(EIR_data.index[0], EIR_data.index[-1])
        Toa_max = max(EIR_data.index[0], EIR_data.index[-1])
        
        def f(Tia: float, Toa: float) -> float:
            if not (Tia_min <= Tia <= Tia_max):
                logger.warning(f"Indoor air temperature {Tia} outside interpolation domain")
            if not (Toa_min <= Toa <= Toa_max):
                logger.warning(f"Outdoor air temperature {Toa} outside interpolation domain")
            return EIRFT_fun(Tia, Toa)[0]

        self.EIRFT_fun = f

    def set_EIRFPLR_fun(self, EIRFPLR_data: pd.Series) -> None:
        """
        Set the function that returns the ratio of actual input power to rated input
        power for a given part-load ratio (PLR).

        Parameters
        ----------
        EIRFPLR_data: pd.Series
            A Pandas Series of which the index consists of PLR values and the data
            (values) of the corresponding input power ratios.

        Notes
        -----
        1. EIRFPLR_fun is a function object with calling signature EIRFPLR_fun(PLR),
        wherein PLR is the part-load ratio, that returns the ratio of actual
        input power to rated input power.

        2. The term EIRFPLR was badly chosen, as it has nothing to do with EIR
        (energy input ratio). It returns an input power ratio, and not
        a ratio of input power to capacity, which is the true definition of EIR .

        3. The EIRFPLR_data can be deduced from manufacturer's data that gives the
        ratio of power input in function of total capacity of indoor units. The
        capacity of the indoor units will finally balance with the load on the
        indoor units, and a such they can be considered the same. Consequently, the
        PLR can also be considered as a ratio of the actual total capacity of indoor
        units to the rated total capacity of indoor units.
        """
        EIRFPLR_fun = interpolate.interp1d(
            x=EIRFPLR_data.index,
            y=EIRFPLR_data.values,
            bounds_error=False,
            fill_value='extrapolate'
        )
        PLR_min = min(EIRFPLR_data.index[0], EIRFPLR_data.index[-1])
        PLR_max = max(EIRFPLR_data.index[0], EIRFPLR_data.index[-1])

        def f(PLR: float) -> float:
            if not (PLR_min <= PLR <= PLR_max):
                logger.warning(f"PLR-value {PLR} outside interpolation domain")
            return EIRFPLR_fun(PLR)

        self.EIRFPLR_fun = f

    def set_CR_corr_fun(self, CR_corr_data: pd.Series) -> None:
        """
        Set the combination ratio correction function to correct for outdoor unit
        capacity.

        Parameters
        ----------
        CR_corr_data: pd.Series
            A Pandas Series object of which the index consists of combination ratio
            (CR) values and the data are the corresponding capacity ratio's.

        Notes
        -----
        1. CR_corr_fun is a function object with calling signature CR_corr_fun(CR),
        wherein CR is the fixed combination ratio (or model size ratio) of the
        system, that returns the capacity correction factor for the given
        combination ratio.

        2. The CR_corr_data can be deduced from manufacturer's data that gives the
        ratio of capacity of the outdoor unit in function of total capacity of
        indoor units.
        """
        CR_corr_fun = interpolate.interp1d(
            x=CR_corr_data.index,
            y=CR_corr_data.values,
            bounds_error=False,
            fill_value='extrapolate'
        )
        CR_min = min(CR_corr_data.index[0], CR_corr_data.index[-1])
        CR_max = max(CR_corr_data.index[0], CR_corr_data.index[-1])

        def f(CR: float) -> float:
            if not (CR_min <= CR <= CR_max):
                logger.warning(f"Combination ratio {CR} outside interpolation domain")
            return CR_corr_fun(CR)

        self.CR_corr_fun = f

    def set_Leq_pipe_corr_fun(
        self,
        L_eq_corr_data: pd.DataFrame | pd.Series,
        cf_height: float = 0.0
    ) -> None:
        """
        Set the equivalent piping length and height correction function for
        the correction of available capacity.

        Parameters
        ----------
        L_eq_corr_data: pd.DataFrame | pd.Series
            - A Pandas DataFrame object having a list of equivalent piping lengths as
            index, the columns are combination ratios and the data are the corresponding
            correction factors.
            - A Pandas Series objects having a list of equivalent piping lengths as
            index, the data are the corresponding correction factors.
        cf_height: float, default 0.0
            Correction factor for the height difference between outdoor unit and
            highest or lowest indoor unit.

        Notes
        -----
        If a Pandas DataFrame object is passed in (cooling mode) and cf_height is
        not zero, Leq_pipe_corr_fun is a function object with calling signature
        Leq_pipe_corr_fun(Leq_pipe, CR, h), wherein Leq_pipe is the equivalent
        pipe length between the outdoor unit and the farthest indoor unit, CR is
        the combination ratio (model size ratio) of the VRF system, and
        h is the height between the outdoor unit and the highest or lowest indoor unit
        or the height between the lowest and highest indoor unit in case indoor units
        are above and below the outdoor unit. The function returns the correction factor
        that must be applied to the available capacity of the outdoor unit.
        If cf_height is zero, no correction for piping height is incorporated and
        the calling signature is f(Leq_pipe, CR).

        If a Pandas Series is passed in (heating mode) and cf_height is
        not zero, the calling signature becomes Leq_pipe_corr_fun(Leq_pipe, h).
        If cf_height is zero, the calling signature will be Leq_pipe_corr_fun(Leq_pipe).
        """
        if isinstance(L_eq_corr_data, pd.DataFrame):
            L_eq_corr_data = L_eq_corr_data.transpose()
            Leq_corr_fun = interpolate.interp2d(
                x=L_eq_corr_data.columns,
                y=L_eq_corr_data.index,
                z=L_eq_corr_data.to_numpy()
            )
            Leq_min = min(L_eq_corr_data.columns[0], L_eq_corr_data.columns[-1])
            Leq_max = max(L_eq_corr_data.columns[0], L_eq_corr_data.columns[-1])
            CR_min = min(L_eq_corr_data.index[0], L_eq_corr_data.index[-1])
            CR_max = max(L_eq_corr_data.index[0], L_eq_corr_data.index[-1])

            if cf_height != 0.0:

                def f(Leq: float | int, CR: float, h: float | int) -> float:
                    if not (Leq_min <= Leq <= Leq_max):
                        logger.warning(f"Equivalent pipe length {Leq} outside interpolation domain")
                    if not (CR_min <= CR <= CR_max):
                        logger.warning(f"Combination ratio {CR} outside interpolation domain")
                    return Leq_corr_fun(Leq, CR)[0] + cf_height * h

                self.Leq_pipe_corr_fun = f
            else:

                def f(Leq: float | int, CR: float) -> float:
                    if not (Leq_min <= Leq <= Leq_max):
                        logger.warning(f"Equivalent pipe length {Leq} outside interpolation domain")
                    if not (CR_min <= CR <= CR_max):
                        logger.warning(f"Combination ratio {CR} outside interpolation domain")
                    return Leq_corr_fun(Leq, CR)[0]

                self.Leq_pipe_corr_fun = f
        else:
            Leq_corr_fun = interpolate.interp1d(
                x=L_eq_corr_data.index,
                y=L_eq_corr_data.values,
                bounds_error=False,
                fill_value='extrapolate'
            )
            Leq_min = min(L_eq_corr_data.index[0], L_eq_corr_data.index[-1])
            Leq_max = max(L_eq_corr_data.index[0], L_eq_corr_data.index[-1])

            if cf_height != 0.0:

                def f(Leq: float | int, h: float | int) -> float:
                    if not (Leq_min <= Leq <= Leq_max):
                        logger.warning(f"Equivalent pipe length {Leq} outside interpolation domain")
                    return Leq_corr_fun(Leq) + cf_height * h

                self.Leq_pipe_corr_fun = f
            else:

                def f(Leq: float | int) -> float:
                    if not (Leq_min <= Leq <= Leq_max):
                        logger.warning(f"Equivalent pipe length {Leq} outside interpolation domain")
                    return Leq_corr_fun(Leq)

                self.Leq_pipe_corr_fun = f

    def set_HPRTF_fun(self, PLR_min: float = 0.2) -> None:
        """
        Set the heat pump runtime fraction when PLR is smaller than PLR_min.

        Parameters
        ----------
        PLR_min: float, default 0.2
            Minimum part-load ratio at which capacity modulation by controlling
            compressor speed is possible. When PLR < PLR_min, the compressor
            will cycle.

        Notes
        -----
        HPRTF_fun is a function object with calling signature HPRTF_fun(PLR),
        wherein PLR is a part-load ratio between 0 and 1, that returns the
        heat pump runtime fraction.
        """
        def HPRTF_fun(PLR: float):
            cycling_ratio = PLR / PLR_min
            cycling_ratio_fraction = 0.85 + 0.15 * cycling_ratio
            HPRTF = min(1.0, cycling_ratio / cycling_ratio_fraction)
            return HPRTF

        self.HPRTF_fun = HPRTF_fun

    def save(self, file_path: str):
        """Save (pickle) the VRF-model to disk."""
        with open(file_path, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(file_path: str) -> 'VRFDataModel':
        """Load the VRF-model from disk."""
        with open(file_path, 'rb') as fh:
            vrf_model = pickle.load(fh)
            return vrf_model
