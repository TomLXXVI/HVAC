from typing import Callable, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd
from hvac import Quantity
from hvac.fluids import HumidAir
from hvac.climate import ClimateData
from hvac.climate.sun.solar_time import time_from_decimal_hour
from ..core import (
    ConstructionAssembly,
    ExteriorBuildingElement,
    InteriorBuildingElement,
    InternalHeatGain,
    ThermalStorageNode,
    ThermalNetwork,
    ThermalNetworkSolver,
    TemperatureSchedule
)


if TYPE_CHECKING:
    from .ventilation_zone import VentilationZone


Q_ = Quantity


@dataclass
class InternalThermalMass:
    A: Quantity
    R: Quantity
    C: Quantity


class Ventilation:

    def __init__(self):
        self.space: Optional['Space'] = None
        self.vz: VentilationZone | None = None

        self.n_min: float = 0.0
        self.V_open: float = 0.0
        self.V_ATD_d: float = 0.0
        self.V_sup: float = 0.0
        self.V_trf: float = 0.0
        self.V_exh: float = 0.0
        self.V_comb: float = 0.0

        self.T_sup: TemperatureSchedule | None = None
        self.T_trf: TemperatureSchedule | None = None
        self.T_int: TemperatureSchedule | None = None
        self.Tdb_ext: TemperatureSchedule | None = None
        self.Twb_ext: TemperatureSchedule | None = None

    @classmethod
    def create(
        cls,
        space: 'Space',
        ventilation_zone: 'VentilationZone',
        n_min: Quantity = Q_(0.5, '1 / hr'),
        V_open: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_ATD_d: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_sup: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_trf: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_exh: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_comb: Quantity = Q_(0.0, 'm ** 3 / hr'),
        T_sup: TemperatureSchedule | None = None,
        T_trf: TemperatureSchedule | None = None,
    ) -> 'Ventilation':
        obj = cls()
        obj.space = space
        obj.vz = ventilation_zone

        obj.n_min = n_min.to('1 / hr').m
        obj.V_open = V_open.to('m ** 3 / hr').m
        obj.V_ATD_d = V_ATD_d.to('m ** 3 / hr').m
        obj.V_sup = V_sup.to('m ** 3 / hr').m
        obj.V_trf = V_trf.to('m ** 3 / hr').m
        obj.V_exh = V_exh.to('m ** 3 / hr').m
        obj.V_comb = V_comb.to('m ** 3 / hr').m

        obj.T_sup = T_sup or space.climate_data.Tdb_ext
        obj.T_trf = T_trf or space.T_int_fun
        obj.T_int = space.T_int_fun
        obj.Tdb_ext = space.climate_data.Tdb_ext
        obj.Twb_ext = space.climate_data.Twb_ext

        return obj

    @property
    def V_leak_ATD(self) -> float:
        """
        External airflow rate into the space through leakages and ATDs.
        (EN 12831-1 eq. 19: the leakage airflow rate into the ventilation zone
        is divided among the spaces according to their envelope surface area
        and the airflow rate through ATDs into the ventilation zone is divided
        among the spaces according to their design airflow rate through ATDs).
        """
        try:
            return (
                self.vz.V_leak * (self.space.envelope_area.to('m ** 2').m / self.vz.A_env) +
                self.vz.V_ATD * (self.V_ATD_d / self.vz.V_ATD_d)
            )
        except ZeroDivisionError:
            try:
                return self.vz.V_leak * (self.space.envelope_area.to('m ** 2').m / self.vz.A_env)
            except ZeroDivisionError:
                return 0.0

    @property
    def V_env(self) -> float:
        """
        External airflow rate into the space through the envelope.
        (EN 12831-1 eq. 18)
        """
        try:
            V_env = (
                (self.vz.V_inf_add / self.vz.V_env)
                * min(self.vz.V_env, self.V_leak_ATD * self.vz.f_dir)
            )
            V_env += (
                (self.vz.V_env - self.vz.V_inf_add)
                / self.vz.V_env * self.V_leak_ATD
            )
        except ZeroDivisionError:
            return float('nan')
        else:
            return V_env

    @property
    def V_tech(self) -> float:
        """
        Technical airflow rate into the space.
        (EN 12831-1 eq. 23)
        """
        return max(self.V_sup + self.V_trf, self.V_exh + self.V_comb)

    @property
    def V_min(self) -> float:
        """
        Minimum required airflow rate of the space that needs to be ensured in
        order to maintain an appropriate level of air hygiene.
        (EN 12831-1 eq. 33)
        """
        return self.n_min * self.space.volume.to('m ** 3').m

    @property
    def V_outdoor(self) -> Quantity:
        """
        Volume flow rate of outdoor air that enters the space.
        """
        V_outside = max(self.V_env + self.V_open, self.V_min - self.V_tech)
        return Q_(V_outside, 'm ** 3 / hr')

    def Q_ven_sen(self, t: float) -> float:
        """
        Sensible infiltration/ventilation load of the space.
        (EN 12831-1 eq. 17)
        """
        VT_inf = (
            max(self.V_env + self.V_open, self.V_min - self.V_tech)
            * (self.Tdb_ext(t) - self.T_int(t))
        )
        VT_sup = self.V_sup * (self.T_sup(t) - self.T_int(t))
        VT_trf = self.V_trf * (self.T_trf(t) - self.T_int(t))
        Q_ven = 0.34 * (VT_inf + VT_sup + VT_trf)  # see EN 12831-1 B.2.8.
        return Q_ven

    def Q_ven_lat(self, t: float) -> float:
        """
        Latent infiltration/ventilation load of the space
        (Principles of Heating, Ventilation, and Air Conditioning in
        Buildings, Chapter 10.5).
        """
        h_o = HumidAir(
            Tdb=Q_(self.Tdb_ext(t), 'degC'),
            Twb=Q_(self.Twb_ext(t), 'degC')
        ).h.to('J / kg').m
        h_x = HumidAir(
            Tdb=Q_(self.Tdb_ext(t), 'degC'),
            RH=self.space.RH_int
        ).h.to('J / kg').m
        rho_o = HumidAir(
            Tdb=Q_(self.Tdb_ext(t), 'degC'),
            Twb=Q_(self.Twb_ext(t), 'degC')
        ).rho.to('kg / m ** 3').m
        V_inf = max(self.V_env + self.V_open, self.V_min - self.V_tech) / 3600.0  # m³/s
        Q_ven_lat = rho_o * V_inf * (h_o - h_x)
        return Q_ven_lat


class Space:

    def __init__(self):
        self.ID: str = ''
        self.height: Quantity | None = None
        self.width: Quantity | None = None
        self.length: Quantity | None = None
        self.ventilation_zone: VentilationZone | None = None
        self.ventilation: Ventilation | None = None
        self.climate_data: ClimateData | None = None
        self.T_int_fun: Callable[[float], float] | None = None
        self.RH_int: Quantity | None = None
        self.ext_building_elements: dict[str, ExteriorBuildingElement] = {}
        self.int_building_elements: dict[str, InteriorBuildingElement] = {}
        self.int_heat_gains: dict[str, InternalHeatGain] = {}
        self.int_thermal_mass: InternalThermalMass | None = None
        self.dt_hr: float = 1.0
        self.n_cycles: int = 6

    @classmethod
    def create(
        cls,
        ID: str,
        height: Quantity,
        width: Quantity,
        length: Quantity,
        climate_data: ClimateData,
        T_int_fun: Callable[[float], float],
        RH_int: Quantity = Q_(50, 'pct'),
        ventilation_zone: Union['VentilationZone', None] = None
    ) -> 'Space':
        space = cls()
        space.ID = ID
        space.height = height
        space.width = width
        space.length = length
        space.ventilation_zone = ventilation_zone
        space.climate_data = climate_data
        space.T_int_fun = T_int_fun
        space.RH_int = RH_int
        # default internal thermal mass
        space.int_thermal_mass = InternalThermalMass(
            A=2 * space.floor_area,
            R=Q_(1 / 25 + 0.1, 'm ** 2 * K / W'),
            C=Q_(200, 'kJ / (m ** 2 * K)')
        )
        return space

    def add_ext_building_element(
        self,
        ID: str,
        azimuth: Quantity,
        tilt: Quantity,
        width: Quantity,
        height: Quantity,
        construction_assembly: ConstructionAssembly,
        F_rad: Quantity = Q_(0.46, 'frac'),
        surface_absorptance: Quantity | None = None,
        surface_color: str = 'dark-colored'
    ) -> ExteriorBuildingElement:
        ebe = ExteriorBuildingElement.create(
            ID=ID,
            azimuth=azimuth,
            tilt=tilt,
            width=width,
            height=height,
            climate_data=self.climate_data,
            construction_assembly=construction_assembly,
            T_int_fun=self.T_int_fun,
            F_rad=F_rad,
            surface_absorptance=surface_absorptance,
            surface_color=surface_color
        )
        self.ext_building_elements[ebe.ID] = ebe
        return ebe

    def add_int_building_element(
        self,
        ID: str,
        width: Quantity,
        height: Quantity,
        construction_assembly: Quantity,
        T_adj_fun: Callable[[float], float],
        F_rad: Quantity = Q_(0.46, 'frac')
    ) -> InteriorBuildingElement:
        ibe = InteriorBuildingElement.create(
            ID=ID,
            width=width,
            height=height,
            construction_assembly=construction_assembly,
            T_int_fun=self.T_int_fun,
            T_adj_fun=T_adj_fun,
            F_rad=F_rad
        )
        self.int_building_elements[ibe.ID] = ibe
        return ibe

    def add_internal_heat_gain(self, ihg: InternalHeatGain) -> None:
        self.int_heat_gains[ihg.ID] = ihg

    def add_internal_thermal_mass(
        self,
        A: Quantity,
        R: Quantity,
        C: Quantity
    ) -> None:
        """
        Add thermal mass to the space interior.

        Parameters
        ----------
        A:
            Surface area of the internal thermal mass
        R:
            Thermal unit resistance between thermal mass and space air. It is
            the sum of convective heat transfer resistance and that of any floor
            surfaces such as carpeting. The convective heat transfer coefficient
            for the interior of buildings is 25 W/(m².K). The thermal resistance
            of carpeting ranges from about 0.1 to 0.4 m².K/W [Principles of
            Heating, Ventilation and Air Conditioning in Buildings, p. 276].
        C:
            The capacitance of the internal thermal mass. This is an aggregate
            value of capacitance that takes into account storage in the floor,
            ceiling, partitions, and furnishings. It is in the range of 100 to
            300 kJ/(m².K) of floor area [Principles of Heating, Ventilation and
            Air Conditioning in Buildings, p. 276].
        """
        self.int_thermal_mass = InternalThermalMass(A, R, C)

    def add_ventilation(
        self,
        n_min: Quantity = Q_(0.5, '1 / hr'),
        V_open: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_ATD_d: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_sup: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_trf: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_exh: Quantity = Q_(0.0, 'm ** 3 / hr'),
        V_comb: Quantity = Q_(0.0, 'm ** 3 / hr'),
        T_sup: Optional[TemperatureSchedule] = None,
        T_trf: Optional[TemperatureSchedule] = None
    ) -> None:
        self.ventilation = Ventilation.create(
            space=self,
            ventilation_zone=self.ventilation_zone,
            n_min=n_min,
            V_open=V_open,
            V_ATD_d=V_ATD_d,
            V_sup=V_sup,
            V_trf=V_trf,
            V_exh=V_exh,
            V_comb=V_comb,
            T_sup=T_sup,
            T_trf=T_trf
        )

    @property
    def envelope_area(self) -> Quantity:
        if self.ext_building_elements:
            A_env = sum(
                ebe.area_gross
                for ebe in self.ext_building_elements.values()
            )
        else:
            A_env = Q_(0.0, 'm ** 2')
        return A_env

    @property
    def floor_area(self) -> Quantity:
        return self.width * self.length

    @property
    def volume(self) -> Quantity:
        return self.floor_area * self.height

    def get_heat_gains(self, unit: str = 'W') -> pd.DataFrame:
        hbm = HeatBalanceMethod(self, self.dt_hr, self.n_cycles)
        df = hbm.get_heat_gains(unit)
        return df

    def get_thermal_storage_heat_flows(
        self,
        T_unit: str = 'degC',
        Q_unit: str = 'W'
    ) -> pd.DataFrame:
        hbm = HeatBalanceMethod(self, self.dt_hr, self.n_cycles)
        df = hbm.get_thermal_storage_heat_flows(T_unit, Q_unit)
        return df


class ThermalNetworkBuilder:
    """Builds the thermal network of the space exterior envelope."""

    @classmethod
    def build(cls, space: 'Space') -> ThermalNetwork | None:
        nodes = []
        int_surf_node_indices = []
        for ebe in space.ext_building_elements.values():
            cls._modify_int_surf_node(ebe)
            for node in ebe.thermal_network.nodes:
                node.A = ebe.area_net
            nodes.extend(ebe.thermal_network.nodes)
            int_surf_node_indices.append(len(nodes) - 1)
        tsn = cls._create_thermal_storage_node(space)
        nodes.append(tsn)
        thermal_network = ThermalNetwork(nodes, int_surf_node_indices)
        return thermal_network

    @classmethod
    def _create_thermal_storage_node(cls, space: 'Space'):
        int_surf_nodes = [
            ebe.thermal_network.nodes[-1]
            for ebe in space.ext_building_elements.values()
        ]
        R = [Q_(isn.R[-1], 'm ** 2 * K / W') for isn in int_surf_nodes]
        R.append(space.int_thermal_mass.R)
        int_thermal_mass_node = ThermalStorageNode(
            ID='internal thermal mass',
            R_list=R,
            C=space.int_thermal_mass.C,
            A=space.int_thermal_mass.A,
            T_input=space.T_int_fun,
            Q_input=cls._get_Q_input_fun(space)
        )
        return int_thermal_mass_node

    @classmethod
    def _get_Q_input_fun(cls, space: 'Space') -> Callable[[float], float]:
        windows = cls._get_windows(space)
        ext_doors = cls._get_ext_doors(space)
        int_doors = cls._get_int_doors(space)

        def _Q_input(t: float) -> float:
            """
            Get the radiative heat gain at time t in seconds from 00:00:00 from
            windows, interior building elements, doors and internal heat gains
            (people, equipment, lightning).
            """
            Q_rad_wnd = sum(
                wnd.get_conductive_heat_gain(t)['rad']
                for wnd in windows
            )
            Q_rad_wnd += sum(
                wnd.get_solar_heat_gain(t)['rad']
                for wnd in windows
            )
            Q_rad_ibe = sum(
                ibe.get_conductive_heat_gain(t)['rad']
                for ibe in space.int_building_elements.values()
            )
            Q_rad_idr = sum(
                door.get_conductive_heat_gain(t)['rad']
                for door in int_doors
            )
            Q_rad_edr = sum(
                door.get_conductive_heat_gain(t)['rad']
                for door in ext_doors
            )
            Q_rad_ihg = sum(
                int_heat_gain.Q_sen(t)['rad']
                for int_heat_gain in space.int_heat_gains.values()
            )
            Q_rad = Q_rad_wnd + Q_rad_ibe + Q_rad_idr + Q_rad_edr + Q_rad_ihg
            return Q_rad

        return _Q_input

    @staticmethod
    def _modify_int_surf_node(ebe) -> None:
        int_surf_node = ebe.thermal_network.nodes[-1]
        R_surf_film = int_surf_node.R[-1]
        R_rad = R_surf_film / ebe.F_rad.m
        R_conv = R_surf_film / (1.0 - ebe.F_rad.m)
        int_surf_node.R = [int_surf_node.R[0], R_conv, R_rad]

    @staticmethod
    def _get_windows(space: 'Space'):
        windows = [
            window
            for ebe in space.ext_building_elements.values()
            for window in ebe.windows if ebe.windows
        ]
        return windows

    @staticmethod
    def _get_ext_doors(space: 'Space'):
        ext_doors = [
            door
            for ebe in space.ext_building_elements.values()
            for door in ebe.doors if ebe.doors
        ]
        return ext_doors

    @staticmethod
    def _get_int_doors(space: 'Space'):
        int_doors = [
            door
            for ibe in space.int_building_elements.values()
            for door in ibe.doors if ibe.doors
        ]
        return int_doors


class HeatBalanceMethod:

    def __init__(self, space: Space, dt_hr: float = 1.0, n_cycles: int = 6):
        self.space = space
        thermal_network = ThermalNetworkBuilder.build(space)
        self._thermal_network = ThermalNetworkSolver.solve(thermal_network, dt_hr, n_cycles)
        self._dt = dt_hr * 3600

    def _calculate_conv_ext_build_elem_gain(self, k: int, T_nodes: list[float]) -> float:
        """
        Calculate heat flow from exterior building elements to space air at
        time index k
        """
        Q_env = 0.0
        T_int = self.space.T_int_fun(k * self._dt)
        for i in self._thermal_network.int_surf_node_indices:
            T_isn = T_nodes[i]
            R_conv = self._thermal_network.nodes[i].R[1]
            q_env = (T_isn - T_int) / R_conv
            Q_env += q_env * self._thermal_network.nodes[i].A
        return Q_env

    def _calculate_conv_therm_mass_gain(self, k: int, T_nodes: list[float]) -> float:
        """
        Calculate heat flow from interior thermal mass to space air at time
        index k
        """
        T_itm = T_nodes[-1]
        T_int = self.space.T_int_fun(k * self._dt)
        R_conv = self._thermal_network.nodes[-1].R[-1]
        q = (T_itm - T_int) / R_conv
        A = self._thermal_network.nodes[-1].A
        Q = A * q
        return Q

    def _calculate_conv_window_gain(self, k: int) -> float:
        """
        Calculate heat flow from windows to space air at time index k.
        """
        windows = [
            window
            for ebe in self.space.ext_building_elements.values()
            for window in ebe.windows if ebe.windows
        ]
        Q = sum(
            window.get_conductive_heat_gain(k * self._dt)['conv'] +
            window.get_solar_heat_gain(k * self._dt)['conv']
            for window in windows
        )
        return Q

    def _calculate_conv_int_build_elem_gain(self, k: int) -> float:
        """
        Calculate heat flow from interior building elements to space air at
        time index k
        """
        Q = sum(
            ibe.get_conductive_heat_gain(k * self._dt)['conv']
            for ibe in self.space.int_building_elements.values()
        )
        return Q

    def _calculate_conv_int_door_gain(self, k: int) -> float:
        """
        Calculate heat flow from interior doors to space air at time index k.
        """
        doors = [
            door
            for ibe in self.space.int_building_elements.values()
            for door in ibe.doors if ibe.doors
        ]
        Q = sum(
            door.get_conductive_heat_gain(k * self._dt)['conv']
            for door in doors
        )
        return Q

    def _calculate_conv_ext_door_gain(self, k: int) -> float:
        """
        Calculate heat flow from exterior doors to space air at time index k.
        """
        doors = [
            door
            for ebe in self.space.ext_building_elements.values()
            for door in ebe.doors if ebe.doors
        ]
        Q = sum(
            door.get_conductive_heat_gain(k * self._dt)['conv']
            for door in doors
        )
        return Q

    def _calculate_conv_int_heat_gain(self, k: int) -> float:
        """
        Calculate heat flow from internal heat gains to space air at time
        index k.
        """
        Q = sum(
            ihg.Q_sen(k * self._dt)['conv']
            for ihg in self.space.int_heat_gains.values()
        )
        return Q

    def _calculate_lat_int_heat_gain(self, k: int) -> float:
        """
        Calculate latent internal heat gains to space air at time index k.
        """
        Q = sum(
            ihg.Q_lat(k * self._dt)
            for ihg in self.space.int_heat_gains.values()
        )
        return Q

    def _calculate_thermal_storage_heat_flows(
        self,
        k: int,
        T_nodes: list[float]
    ) -> tuple[float, ...]:
        """
        Calculate heat flow into internal thermal mass, heat stored in internal
        thermal mass and heat flow out from internal thermal mass at time index
        k.
        """
        # heat flow from exterior building elements into internal thermal mass
        # at time index k
        Q_rad_ext = 0.0
        A_itm = self._thermal_network.nodes[-1].A
        T_itm = T_nodes[-1]
        for i in self._thermal_network.int_surf_node_indices:
            T_isn = T_nodes[i]
            R_rad = self._thermal_network.nodes[i].R[-1]
            Q_rad_ext += A_itm * (T_isn - T_itm) / R_rad

        # heat flow from windows, doors and interior building elements into
        # internal thermal mass at time index k
        Q_rad_other = self._thermal_network.nodes[-1].Q_input(k * self._dt)

        Q_in = Q_rad_ext + Q_rad_other

        # heat flow from internal thermal mass to space air at time index k
        T_int = self.space.T_int_fun(k * self._dt)
        R_itm = self._thermal_network.nodes[-1].R[-1]
        Q_out = A_itm * (T_itm - T_int) / R_itm

        # heat stored in internal thermal mass at time index k
        Q_sto = Q_in - Q_out

        return T_itm, Q_in, Q_out, Q_sto

    def _calculate_sen_vent_gain(self, k: int) -> float:
        """
        Calculate sensible heat gain to space air at time index k.
        """
        if self.space.ventilation is not None:
            Q = self.space.ventilation.Q_ven_sen(k * self._dt)
            return Q
        return 0.0

    def _calculate_lat_vent_gain(self, k: int) -> float:
        if self.space.ventilation is not None:
            Q = self.space.ventilation.Q_ven_lat(k * self._dt)
            return Q
        return 0.0

    def get_heat_gains(self, unit: str = 'W') -> pd.DataFrame:
        """
        Get Pandas `DataFrame` object with the heat gains to the space air and
        the cooling load of the space at each time moment of the design day
        in the measuring unit asked (default unit is Watts, 'W').
        """
        Q_gains = {
            'time': [],
            'Q_conv_ext': [],
            'Q_conv_int': [],
            'Q_conv_itm': [],
            'Q_conv_wnd': [],
            'Q_conv_ihg': [],
            'Q_sen_vent': [],
            'Q_sen_load': [],
            'Q_lat_ihg': [],
            'Q_lat_vent': [],
            'Q_lat_load': []
        }
        # noinspection PyProtectedMember
        for k, T_nodes in enumerate(self._thermal_network._T_node_table):

            Q_gains['time'].append(time_from_decimal_hour(k * self._dt / 3600))

            # heat flow from exterior building elements to space air at time
            # index k
            Q_gains['Q_conv_ext'].append(
                Q_(
                    self._calculate_conv_ext_build_elem_gain(k, T_nodes) +
                    self._calculate_conv_ext_door_gain(k),
                    'W'
                ).to(unit).m
            )

            # heat flow from interior building elements to space air at time
            # index k
            Q_gains['Q_conv_int'].append(
                Q_(
                    self._calculate_conv_int_build_elem_gain(k) +
                    self._calculate_conv_int_door_gain(k),
                    'W'
                ).to(unit).m
            )

            # heat flow from internal thermal mass to space air at time
            # index k
            Q_gains['Q_conv_itm'].append(
                Q_(
                    self._calculate_conv_therm_mass_gain(k, T_nodes),
                    'W'
                ).to(unit).m
            )

            # heat flow from windows to space air at time index k
            Q_gains['Q_conv_wnd'].append(
                Q_(
                    self._calculate_conv_window_gain(k),
                    'W'
                ).to(unit).m
            )

            # heat flow from internal heat gains to space air at time index k
            Q_gains['Q_conv_ihg'].append(
                Q_(
                    self._calculate_conv_int_heat_gain(k),
                    'W'
                ).to(unit).m
            )

            # heat flow from ventilation to space air at time index k
            Q_gains['Q_sen_vent'].append(
                Q_(
                    self._calculate_sen_vent_gain(k),
                    'W'
                ).to(unit).m
            )

            # sensible cooling load of space air at time index k
            Q_gains['Q_sen_load'].append(
                Q_gains['Q_conv_ext'][-1] +
                Q_gains['Q_conv_int'][-1] +
                Q_gains['Q_conv_itm'][-1] +
                Q_gains['Q_conv_wnd'][-1] +
                Q_gains['Q_conv_ihg'][-1] +
                Q_gains['Q_sen_vent'][-1]
            )

            # latent heat transfer from internal heat gains to space air at
            # time index k
            Q_gains['Q_lat_ihg'].append(
                Q_(
                    self._calculate_lat_int_heat_gain(k),
                    'W'
                ).to(unit).m
            )

            # latent heat transfer from ventilation to space air at time index
            # k
            Q_gains['Q_lat_vent'].append(
                Q_(
                    self._calculate_lat_vent_gain(k),
                    'W'
                ).to(unit).m
            )

            # latent cooling load of space air at time index k
            Q_gains['Q_lat_load'].append(
                Q_gains['Q_lat_ihg'][-1] +
                Q_gains['Q_lat_vent'][-1]
            )

        Q_gains = pd.DataFrame(Q_gains)
        Q_gains['Q_tot_load'] = Q_gains['Q_sen_load'] + Q_gains['Q_lat_load']
        return Q_gains

    def get_thermal_storage_heat_flows(
        self,
        T_unit: str = 'degC',
        Q_unit: str = 'W'
    ) -> pd.DataFrame:

        Q_therm_storage = {
            'time': [],
            'T_itm': [],
            'Q_in': [],
            'Q_out': [],
            'Q_sto': []
        }

        # noinspection PyProtectedMember
        for k, T_nodes in enumerate(self._thermal_network._T_node_table):
            tup = self._calculate_thermal_storage_heat_flows(k, T_nodes)
            Q_therm_storage['time'].append(
                time_from_decimal_hour(k * self._dt / 3600)
            )
            Q_therm_storage['T_itm'].append(
                Q_(tup[0], 'degC').to(T_unit).m
            )
            Q_therm_storage['Q_in'].append(
                Q_(tup[1], 'W').to(Q_unit).m
            )
            Q_therm_storage['Q_out'].append(
                Q_(tup[2], 'W').to(Q_unit).m
            )
            Q_therm_storage['Q_sto'].append(
                Q_(tup[3], 'W').to(Q_unit).m
            )

        Q_therm_storage = pd.DataFrame(Q_therm_storage)
        return Q_therm_storage
