from typing import Dict, List, Callable
from datetime import datetime
import pandas as pd
from hvac import Quantity
from hvac.climate import ClimateData
from hvac.climate.sun.solar_time import time_from_decimal_hour
from .construction_assembly import ConstructionAssembly
from .thermal_network import (
    ThermalNetwork,
    ThermalNetworkBuilder,
    ThermalNetworkSolver
)
from .exterior_surface import ExteriorSurface
from .fenestration import (
    WindowThermalProperties,
    ExteriorShadingDevice,
    InteriorShadingDevice,
    Window
)


Q_ = Quantity


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
        self.ID: str = ''
        self._exterior_surface: ExteriorSurface | None = None
        self.construction_assembly: ConstructionAssembly | None = None
        self._thermal_network: ThermalNetwork | None = None
        self.T_int_fun: Callable[[float], float] | None = None
        self.F_rad: Quantity | None = None
        self._heat_transfer: Dict[str, List[Quantity]] | None = None
        self.windows: List[Window] = []
        self.doors: List['ExteriorBuildingElement'] = []

    @classmethod
    def create(
        cls,
        ID: str,
        azimuth: Quantity,
        tilt: Quantity,
        width: Quantity,
        height: Quantity,
        climate_data: ClimateData,
        construction_assembly: ConstructionAssembly,
        T_int_fun: Callable[[float], float],
        F_rad: Quantity = Q_(0.46, 'frac'),
        surface_absorptance: Quantity | None = None,
        surface_color: str = 'dark-colored'
    ) -> 'ExteriorBuildingElement':
        """
        Create an `ExteriorBuildingElement` object.

        Parameters
        ----------
        ID: str
            Identifier for the building element.
        azimuth: Quantity
            The azimuth angle of the exterior building element measured clockwise
            from North (East = 90°, South = 180°, West = 270 °, North = 0°)
        tilt: Quantity
            The tilt angle of the exterior surface with respect to the horizontal
            surface. A vertical wall has a tilt angle of 90°.
        width: Quantity
            The width of the exterior surface.
        height: Quantity
            The height of the exterior surface.
        climate_data: ClimateData
            Object that determines the outdoor dry- and wet-bulb temperature and
            solar radiation for each hour of the design day. See class
            ClimateData.
        construction_assembly: ConstructionAssembly
            Object that defines the construction of the building element. See
            class ConstructionAssembly.
        T_int_fun: Callable
            Function that returns the indoor air temperature for each hour of
            the day.
        F_rad: Quantity, default 0.46
            Fraction of conductive heat flow that is transferred by radiation
            to the internal mass of the space (ASHRAE Fundamentals 2017, Ch. 18,
            Table 14).
        surface_absorptance: Quantity | None (default)
            The absorptance of the exterior surface of the building element.
        surface_color: ['dark-colored' (default), 'light-colored']
            Indicate if the surface is either dark-colored or light-colored. You
            can use this instead of specifying `surface_absorptance`.

        Returns
        -------
        ExteriorBuildingElement
        """
        ext_building_element = cls()
        ext_building_element.ID = ID
        ext_building_element.construction_assembly = construction_assembly
        ext_building_element._exterior_surface = ExteriorSurface(
            azimuth=azimuth,
            tilt=tilt,
            width=width,
            height=height,
            climate_data=climate_data,
            surface_resistance=construction_assembly.R_surf_ext,
            surface_absorptance=surface_absorptance,
            surface_color=surface_color
        )
        ext_building_element.T_int_fun = T_int_fun
        ext_building_element.F_rad = F_rad.to('frac')
        return ext_building_element

    @property
    def thermal_network(self) -> ThermalNetwork:
        """
        Get thermal network.
        """
        if self._thermal_network is None:
            self._thermal_network = ThermalNetworkBuilder.build(self.construction_assembly)
            self._thermal_network.T_ext = self._exterior_surface.T_sol
            self._thermal_network.T_int = self.T_int_fun
        return self._thermal_network

    @property
    def area_net(self) -> Quantity:
        A_net = self._exterior_surface.area
        if self.windows:
            A_wnd = sum(window.area for window in self.windows)
            A_net -= A_wnd
        if self.doors:
            A_drs = sum(door.area_gross for door in self.doors)
            A_net -= A_drs
        return A_net

    @property
    def area_gross(self) -> Quantity:
        return self._exterior_surface.area

    @property
    def irr_profile(self) -> Dict[str, List[datetime] | List[Quantity]]:
        return self._exterior_surface.irr_profile

    @property
    def T_sol_profile(self) -> Dict[str, List[datetime] | List[Quantity]]:
        return self._exterior_surface.T_sol_profile

    def add_window(
        self,
        ID: str,
        width: Quantity,
        height: Quantity,
        therm_props: WindowThermalProperties,
        F_rad: Quantity = Q_(0.46, 'frac'),
        ext_shading_dev: ExteriorShadingDevice | None = None,
        int_shading_dev: InteriorShadingDevice | None = None
    ) -> None:
        window = Window.create(
            ID=ID,
            azimuth=self._exterior_surface.azimuth,
            tilt=self._exterior_surface.tilt,
            width=width,
            height=height,
            climate_data=self._exterior_surface.climate_data,
            therm_props=therm_props,
            T_int_fun=self.T_int_fun,
            F_rad=F_rad,
            ext_shading_dev=ext_shading_dev,
            int_shading_dev=int_shading_dev
        )
        self.windows.append(window)

    def add_door(
        self,
        ID: str,
        width: Quantity,
        height: Quantity,
        construction_assembly: ConstructionAssembly,
        F_rad: Quantity = Q_(0.46, 'frac'),
        surface_absorptance: Quantity | None = None,
        surface_color: str = 'dark-colored'
    ) -> None:
        door = ExteriorBuildingElement.create(
            ID=ID,
            azimuth=self._exterior_surface.azimuth,
            tilt=self._exterior_surface.tilt,
            width=width,
            height=height,
            climate_data=self._exterior_surface.climate_data,
            construction_assembly=construction_assembly,
            T_int_fun=self.T_int_fun,
            F_rad=F_rad,
            surface_absorptance=surface_absorptance,
            surface_color=surface_color
        )
        self.doors.append(door)

    def get_conductive_heat_gain(self, t: float) -> Dict[str, float]:
        # note: the **steady-state** conductive heat gain is calculated.
        U = self.construction_assembly.U.to('W / (m ** 2 * K)').m
        A = self.area_net.to('m ** 2').m
        T_int = self.T_int_fun(t)
        T_sol = self._exterior_surface.T_sol(t)
        Q = U * A * (T_sol - T_int)
        Q_rad = self.F_rad.m * Q
        Q_conv = Q - Q_rad
        return {'rad': Q_rad, 'conv': Q_conv}

    def get_heat_transfer(
        self,
        dt_hr: float = 1.0,
        n_cycles: int = 6,
        unit: str = 'W'
    ) -> pd.DataFrame:
        """
        Get the diurnal heat transfer cycle through the exterior building
        element.

        Parameters
        ----------
        dt_hr:
            The time step of the calculations in decimal hours.
        n_cycles:
            The number of repeated cycles before returning the result.
        unit: default 'W'
            The desired unit in which thermal power is to be expressed.

        Returns
        -------
        A Pandas DataFrame with the dynamic heat flows at the exterior side (key
        `Q_ext`) and interior side (key `Q_int`) of the building element, and
        also the calculated steady-state value (based on the temperature
        difference between exterior, sol-air temperature and indoor temperature
        and the thermal resistance of the building element) (key `Q_steady`).
        """
        if self._heat_transfer is None:
            tnw_solved = ThermalNetworkSolver.solve(
                self.thermal_network,
                dt_hr,
                n_cycles
            )
            dt = dt_hr * 3600
            A = self.area_net.to('m ** 2')
            self._heat_transfer = {
                'Q_ext': [],
                'Q_int': [],
                'Q_steady': []
            }
            t_ax = []
            for k, T_node_list in enumerate(tnw_solved.T_node_table):
                t_ax.append(time_from_decimal_hour(k * dt_hr))
                T_sol = Q_(self._exterior_surface.T_sol(k * dt), 'degC').to('K')
                T_node_ext = T_node_list[0].to('K')
                R_ext = tnw_solved.R_ext
                q_ext = (T_sol - T_node_ext) / R_ext
                Q_ext = A * q_ext
                T_int = Q_(self.T_int_fun(k * dt), 'degC').to('K')
                T_node_int = T_node_list[-1].to('K')
                R_int = tnw_solved.R_int
                q_int = (T_node_int - T_int) / R_int
                Q_int = A * q_int
                R_tot = self.construction_assembly.R
                q_steady = (T_sol - T_int) / R_tot
                Q_steady = A * q_steady
                self._heat_transfer['Q_ext'].append(Q_ext.to(unit).m)
                self._heat_transfer['Q_int'].append(Q_int.to(unit).m)
                self._heat_transfer['Q_steady'].append(Q_steady.to(unit).m)
            self._heat_transfer = pd.DataFrame(data=self._heat_transfer, index=t_ax)
        return self._heat_transfer


class InteriorBuildingElement:

    def __init__(self):
        self.ID: str = ''
        self.width: Quantity | None = None
        self.height: Quantity | None = None
        self.construction_assembly: Quantity | None = None
        self.T_int_fun: Callable[[float], float] | None = None
        self.T_adj_fun: Callable[[float], float] | None = None
        self.F_rad: Quantity = Q_(0.46, 'frac')
        self.doors: List['InteriorBuildingElement'] = []

    @classmethod
    def create(
        cls,
        ID: str,
        width: Quantity,
        height: Quantity,
        construction_assembly: Quantity,
        T_int_fun: Callable[[float], float],
        T_adj_fun: Callable[[float], float],
        F_rad: Quantity = Q_(0.46, 'frac')
    ) -> 'InteriorBuildingElement':
        int_build_elem = cls()
        int_build_elem.ID = ID
        int_build_elem.width = width
        int_build_elem.height = height
        int_build_elem.construction_assembly = construction_assembly
        int_build_elem.T_int_fun = T_int_fun
        int_build_elem.T_adj_fun = T_adj_fun
        int_build_elem.F_rad = F_rad.to('frac')
        return int_build_elem

    @property
    def area_net(self) -> Quantity:
        A_net = self.area_gross
        if self.doors:
            A_drs = sum(door.area_gross for door in self.doors)
            A_net -= A_drs
        return A_net

    @property
    def area_gross(self) -> Quantity:
        return self.width * self.height

    def get_conductive_heat_gain(self, t: float) -> Dict[str, float]:
        U = self.construction_assembly.U.to('W / (m ** 2 * K)').m
        A = self.area_net.to('m ** 2').m
        T_int = self.T_int_fun(t)
        T_adj = self.T_adj_fun(t)
        Q = U * A * (T_adj - T_int)
        Q_rad = self.F_rad.m * Q
        Q_conv = Q - Q_rad
        return {'rad': Q_rad, 'conv': Q_conv}

    def add_door(
        self,
        ID: str,
        width: Quantity,
        height: Quantity,
        construction_assembly: ConstructionAssembly,
        F_rad: Quantity = Q_(0.46, 'frac')
    ) -> None:
        door = InteriorBuildingElement.create(
            ID=ID,
            width=width,
            height=height,
            construction_assembly=construction_assembly,
            F_rad=F_rad,
            T_int_fun=self.T_int_fun,
            T_adj_fun=self.T_adj_fun
        )
        self.doors.append(door)
