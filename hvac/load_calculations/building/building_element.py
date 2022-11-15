from typing import Dict, List, Any, Callable
import math
from scipy.interpolate import interp1d
from hvac import Quantity
from hvac.climate.sun import Surface, AnisotropicSkyModel
from hvac.climate.sun.solar_time import time_to_decimal_hour
from hvac.climate import ClimateData
from hvac.load_calculations.core import ConstructionAssembly
from hvac.load_calculations.core import ThermalNetwork, ThermalNetworkBuilder, ThermalNetworkSolver


Q_ = Quantity


class _ExteriorSurface(Surface):
    """
    Only for internal use. Is part of `ExteriorBuildingElement`.
    Used for calculating the daily hourly average sol air temperature on
    the exterior surface of a building element.
    """

    def __init__(
        self,
        azimuth: Quantity,
        tilt: Quantity,
        width: Quantity,
        height: Quantity,
        climate_data: ClimateData,
        surface_resistance: Quantity,
        surface_absorptance: Quantity
    ) -> None:
        super().__init__(azimuth, tilt, width, height)
        self.climate_data = climate_data
        self.R_surf = surface_resistance
        self.a_surf = surface_absorptance
        self.irradiance_profile = AnisotropicSkyModel.daily_profile(
            location=climate_data.location,
            surface=self,
            irradiance_hor_profile=climate_data.irradiance_profile
        )
        self.T_sol_profile = self._get_sol_air_temperature_profile()
        self._T_sol_fun = self._interpolate_sol_air_temperature()

    def _get_sol_air_temperature_profile(self) -> Dict[str, List[Any]]:
        tilt = self.tilt.to('rad').m
        dT_sky = 3.9 * math.cos(tilt)
        T_arr = [T.to('degC').m for T in self.climate_data.Tdb_profile['T']]
        I_glo_arr = [I.to('W / m ** 2').m for I in self.irradiance_profile['I_glo_sur']]
        R_surf = self.R_surf.to('m ** 2 * K / W').m
        a_surf = self.a_surf.to('frac').m
        Tsol_arr = [T + a_surf * R_surf * I_glo - dT_sky for T, I_glo in zip(T_arr, I_glo_arr)]
        d = {
            't': self.irradiance_profile['t'],
            'T': [Q_(Tsol, 'degC') for Tsol in Tsol_arr]
        }
        return d

    def _interpolate_sol_air_temperature(self) -> Callable[[float], float]:
        t_ax = [time_to_decimal_hour(dt.time()) * 3600.0 for dt in self.T_sol_profile['t']]
        T_ax = [T.to('degC').m for T in self.T_sol_profile['T']]
        interpolant = interp1d(x=t_ax, y=T_ax, kind='cubic')
        return interpolant

    def T_sol(self, t: float) -> float:
        """
        Get sol air temperature in degC at time *t* in seconds from 00:00:00.
        """
        return self._T_sol_fun(t)


class ExteriorBuildingElement:
    """
    An `ExteriorBuildingElement` is composed of:
    - an `ExteriorSurface` that defines the orientation and size of the
    building element and that is responsible for calculating the sol-air
    temperature under clear-sky conditions (based on `ClimateData`).
    - a `ConstructionAssembly` that defines the layered material composition of
    the building element and knows about the thermal resistance and thermal
    capacity of the building element.
    - a `ThermalNetwork` model of the construction assembly that is used to
    calculate the dynamic and steady heat transfer through the building element.
    """

    def __init__(self):
        self._exterior_surface: _ExteriorSurface | None = None
        self.construction_assembly: ConstructionAssembly | None = None
        self.thermal_network: ThermalNetwork | None = None
        self._T_int: Callable[[float], float] | None = None
        self._heat_transfer: Dict[str, List[Quantity]] | None = None

    @classmethod
    def create(
        cls,
        azimuth: Quantity,
        tilt: Quantity,
        width: Quantity,
        height: Quantity,
        climate_data: ClimateData,
        construction_assembly: ConstructionAssembly,
        surface_absorptance: Quantity,
        T_int_fun: Callable[[float], float]
    ) -> 'ExteriorBuildingElement':
        ext_building_element = cls()
        ext_building_element.construction_assembly = construction_assembly
        ext_building_element._exterior_surface = _ExteriorSurface(
            azimuth=azimuth,
            tilt=tilt,
            width=width,
            height=height,
            climate_data=climate_data,
            surface_resistance=construction_assembly.R_surf_ext,
            surface_absorptance=surface_absorptance
        )
        ext_building_element._T_int = T_int_fun
        return ext_building_element

    def get_heat_transfer(self, dt_hr: float = 1.0, n_cycles: int = 6) -> Dict[str, List[Quantity]]:
        if self._heat_transfer is None:
            self.thermal_network = ThermalNetworkBuilder.build(self.construction_assembly)
            self.thermal_network.T_ext = self._exterior_surface.T_sol
            self.thermal_network.T_int = self._T_int
            self.thermal_network = ThermalNetworkSolver.solve(self.thermal_network, dt_hr, n_cycles)
            dt = dt_hr * 3600
            self._heat_transfer = {
                'q_ext': [],
                'q_int': [],
                'q_steady': []
            }
            for k, T_node_list in enumerate(self.thermal_network.T_node_table):
                T_sol = Q_(self._exterior_surface.T_sol(k * dt), 'degC').to('K')
                T_node_ext = T_node_list[0].to('K')
                R_ext = self.thermal_network.R_ext
                q_ext = (T_sol - T_node_ext) / R_ext
                T_int = Q_(self._T_int(k * dt), 'degC').to('K')
                T_node_int = T_node_list[-1].to('K')
                R_int = self.thermal_network.R_int
                q_int = (T_node_int - T_int) / R_int
                R_tot = self.construction_assembly.R
                q_steady = (T_sol - T_int) / R_tot
                self._heat_transfer['q_ext'].append(q_ext)
                self._heat_transfer['q_int'].append(q_int)
                self._heat_transfer['q_steady'].append(q_steady)
        return self._heat_transfer

    @property
    def area_net(self) -> Quantity:
        return self._exterior_surface.area

    @property
    def area_gross(self) -> Quantity:
        return self._exterior_surface.area
