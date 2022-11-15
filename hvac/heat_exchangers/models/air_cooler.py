from typing import Optional, List, Tuple, Union
import numpy as np
from scipy.optimize import fsolve
from hvac import Quantity
from hvac.logging import ModuleLogger
from hvac.fluids import HumidAir, FluidState
from .core import *

Q_ = Quantity
logger = ModuleLogger.get_logger(__name__)


class AirCooler:

    def __init__(
            self,
            coil_geometry: CoilGeometry,
            air_stream: AirStream,
            coolant_stream: CoolantStream,
            n_segments: int
    ) -> None:
        self.coil_geometry = coil_geometry
        self.air_stream = air_stream
        if self.air_stream.m is None:
            self.air_stream.m = self.air_stream.inlet.rho * self.air_stream.vfa * self.coil_geometry.Afa
        self.coolant_stream = coolant_stream
        self.n_segments = n_segments

        self.control_volumes: List[AbstractControlVolume] = []
        self.air_out: Optional[HumidAir] = None
        self.coolant_in: Optional[FluidState] = None
        self._adp: Optional[HumidAir] = None
        self._section_printer = AirCooler.SectionPrinter(self.n_segments)
        self._segment_printer = AirCooler.SegmentPrinter()

    def analyze(self, display_info: bool = True) -> Tuple[HumidAir, FluidState]:
        self.control_volumes = []
        if display_info:
            self._section_printer.print_header()
        Ao = (
                self.coil_geometry.Afa *
                self.coil_geometry.Nr *
                self.coil_geometry.Ao_to_Afa
        )
        delta_Ao = Ao / self.n_segments
        air_stream = AirStream(
            inlet=self.air_stream.inlet,
            outlet=None,
            m=self.air_stream.m,
            vfa=self.air_stream.vfa
        )
        coolant_stream = CoolantStream(
            inlet=None,
            outlet=self.coolant_stream.outlet,
            m=self.coolant_stream.m,
            m_tube=self.coolant_stream.m_tube,
            type=self.coolant_stream.type,
            e=self.coolant_stream.e
        )
        for i in range(self.n_segments):
            cv = ControlVolume.create(
                self.coil_geometry, air_stream,
                coolant_stream, delta_Ao
            )
            cv.analyze()
            self.control_volumes.append(cv)
            if display_info:
                self._section_printer.print_section(i, cv)
            air_stream = AirStream(
                inlet=cv.air_stream.outlet,
                outlet=None,
                m=cv.air_stream.m,
                vfa=cv.air_stream.vfa
            )
            coolant_stream = CoolantStream(
                inlet=None,
                outlet=cv.coolant_stream.inlet,
                m=cv.coolant_stream.m,
                m_tube=cv.coolant_stream.m_tube,
                type=cv.coolant_stream.type,
                e=cv.coolant_stream.e
            )
        self.air_out = self.control_volumes[-1].air_stream.outlet
        self.coolant_in = self.control_volumes[-1].coolant_stream.inlet
        return self.air_out, self.coolant_in

    def size(self, display_info: bool = True) -> Tuple[Quantity, Quantity, float]:
        self.control_volumes = []
        if display_info:
            self._segment_printer.print_header()
        hai = self.air_stream.inlet.h
        hao = self.air_stream.outlet.h
        dha = (hai - hao) / self.n_segments
        dhc = self.air_stream.m * dha / self.coolant_stream.m
        air_stream = AirStream(
            inlet=self.air_stream.inlet,
            outlet=HumidAir(h=hai - dha, W=self.air_stream.inlet.W),  # initially assume only sensible heat transfer
            m=self.air_stream.m,
            vfa=self.air_stream.vfa
        )
        coolant_stream = CoolantStream(
            inlet=self.coolant_stream.type(
                P=self.coolant_stream.outlet.P,
                h=self.coolant_stream.outlet.h - dhc
            ),
            outlet=self.coolant_stream.outlet,
            m=self.coolant_stream.m,
            m_tube=self.coolant_stream.m_tube,
            type=self.coolant_stream.type,
            e=self.coolant_stream.e
        )
        for i in range(self.n_segments):
            cv = ControlVolume.create(self.coil_geometry, air_stream, coolant_stream, None)
            cv.size()
            self.control_volumes.append(cv)
            if display_info:
                self._segment_printer.print_segment(i, cv)
            air_stream = AirStream(
                inlet=cv.air_stream.outlet,
                outlet=HumidAir(h=cv.air_stream.outlet.h - dha, W=cv.air_stream.outlet.W),
                # initially assume only sensible heat transfer
                m=cv.air_stream.m,
                vfa=cv.air_stream.vfa
            )
            coolant_stream = CoolantStream(
                inlet=cv.coolant_stream.type(
                    P=cv.coolant_stream.inlet.P,
                    h=cv.coolant_stream.inlet.h - dhc
                ),
                outlet=cv.coolant_stream.inlet,
                m=cv.coolant_stream.m,
                m_tube=cv.coolant_stream.m_tube,
                type=cv.coolant_stream.type,
                e=cv.coolant_stream.e
            )
        Ao = sum(cv.delta_Ao for cv in self.control_volumes)
        self.air_out = self.control_volumes[-1].air_stream.outlet
        self.coolant_in = self.control_volumes[-1].coolant_stream.inlet
        self.coil_geometry.Afa = self.air_stream.m / (self.air_stream.inlet.rho * self.air_stream.vfa)
        self.coil_geometry.Nr = float(Ao / (self.coil_geometry.Ao_to_Afa * self.coil_geometry.Afa))
        return Ao, self.coil_geometry.Afa, self.coil_geometry.Nr

    @property
    def Q(self) -> Quantity:
        """Total heat transfer."""
        Q = sum(cv.ht.delta_Q for cv in self.control_volumes)
        return Q

    @property
    def mw(self) -> Quantity:
        """Moisture removal rate."""
        mw = sum(
            cv.ht.delta_mw
            for cv in self.control_volumes
            if isinstance(cv, WetControlVolume)
        )
        return mw

    @property
    def Qs(self) -> Quantity:
        """Sensible heat transfer."""
        Qs_wet = sum(
            cv.ht.delta_Qs
            for cv in self.control_volumes
            if isinstance(cv, WetControlVolume)
        )
        Qs_dry = sum(
            cv.ht.delta_Q
            for cv in self.control_volumes
            if isinstance(cv, DryControlVolume)
        )
        return Qs_wet + Qs_dry

    @property
    def Ql(self) -> Quantity:
        """Latent heat transfer."""
        Ql = sum(
            cv.ht.delta_Ql
            for cv in self.control_volumes
            if isinstance(cv, WetControlVolume)
        )
        return Ql

    @property
    def SHR(self) -> Quantity:
        """Sensible heat ratio."""
        SHR = self.Qs / self.Q
        return SHR.to('frac')

    def get_air_states(self) -> List[HumidAir]:
        """Returns a list with the air states at the calculated sections."""
        air_state_list = [self.control_volumes[0].air_stream.inlet]
        air_state_list.extend(cv.air_stream.outlet for cv in self.control_volumes)
        return air_state_list

    def get_surface_temperatures(self) -> List[Quantity]:
        """Returns a list with the surface temperatures at the calculated sections."""
        surf_state_list = [self.control_volumes[0].surface.Ti]
        surf_state_list.extend(cv.surface.To for cv in self.control_volumes)
        return surf_state_list

    def get_coolant_states(self) -> List[FluidState]:
        """Returns a list with the coolant states at the calculated sections."""
        coolant_state_list = [self.control_volumes[0].coolant_stream.outlet]
        coolant_state_list.extend(cv.coolant_stream.inlet for cv in self.control_volumes)
        return coolant_state_list

    @property
    def ADP(self) -> HumidAir:
        """Apparatus Dew Point."""
        if self._adp is None:
            Tai = self.air_stream.inlet.Tdb.to('K')
            Wai = self.air_stream.inlet.W
            Tao = self.air_out.Tdb.to('K')
            Wao = self.air_out.W

            def eq(unknown: np.ndarray) -> np.ndarray:
                Tadp = Q_(unknown[0], 'K')
                Wadp = HumidAir(Tdb=Tadp, RH=Q_(100, 'pct')).W
                rhs = Tai - (Tai - Tao) / (Wai - Wao) * (Wai - Wadp)
                out = Tadp - rhs
                return out.m

            Tadp_ini = self.air_out.Tdp.to('K').m
            roots = fsolve(eq, np.array([Tadp_ini]))
            T_adp = Q_(roots[0], 'K')
            self._adp = HumidAir(Tdb=T_adp, RH=Q_(100, 'pct'))
        return self._adp

    @property
    def CF(self) -> Quantity:
        """Contact factor."""
        hai = self.air_stream.inlet.h
        hao = self.air_out.h
        h_adp = self.ADP.h
        cf = (hai - hao) / (hai - h_adp)
        return cf.to('frac')

    @property
    def BF(self) -> Quantity:
        """Bypass factor."""
        return 1 - self.CF

    @property
    def Uod_avg(self) -> Quantity:
        """Average global heat transfer coefficient of dry part."""
        Uod_list = []
        n_dry = 0
        for cv in self.control_volumes:
            if isinstance(cv, DryControlVolume):
                Uod_list.append(cv.htp.Uo)
                n_dry += 1
        if Uod_list:
            Uod_avg = sum(Uod_list) / n_dry
        else:
            Uod_avg = Q_(0.0, 'W / (m**2 * K)')
        return Uod_avg

    @property
    def Uow_avg(self) -> Quantity:
        """Average global heat transfer coefficient of wet part."""
        Uow_list = []
        n_wet = 0
        for cv in self.control_volumes:
            if isinstance(cv, WetControlVolume):
                Uow_list.append(cv.htp.Uo)
                n_wet += 1
        if Uow_list:
            Uow_avg = sum(Uow_list) / n_wet
        else:
            Uow_avg = Q_(0.0, 'kg / (m**2 * s)')
        return Uow_avg

    @property
    def hed_avg(self) -> Quantity:
        """Average external heat transfer coefficient of dry part."""
        hed_list = []
        n_dry = 0
        for cv in self.control_volumes:
            if isinstance(cv, DryControlVolume):
                hed_list.append(cv.htp.he)
                n_dry += 1
        if hed_list:
            hed_avg = sum(hed_list) / n_dry
        else:
            hed_avg = Q_(0.0, 'W / (m**2 * K)')
        return hed_avg

    @property
    def hew_avg(self) -> Quantity:
        """Average external heat transfer coefficient of wet part."""
        hew_list = []
        n_wet = 0
        for cv in self.control_volumes:
            if isinstance(cv, WetControlVolume):
                hew_list.append(cv.htp.he)
                n_wet += 1
        if hew_list:
            hew_avg = sum(hew_list) / n_wet
        else:
            hew_avg = Q_(0.0, 'kg / (m**2 * s)')
        return hew_avg

    @property
    def hid_avg(self) -> Quantity:
        """Average internal heat transfer coefficient of dry part."""
        hid_list = []
        n_dry = 0
        for cv in self.control_volumes:
            if isinstance(cv, DryControlVolume):
                hid_list.append(cv.htp.hi)
                n_dry += 1
        if hid_list:
            hid_avg = sum(hid_list) / n_dry
        else:
            hid_avg = Q_(0.0, 'W / (m**2 * K)')
        return hid_avg

    @property
    def hiw_avg(self) -> Quantity:
        """Average internal heat transfer coefficient of wet part."""
        hiw_list = []
        n_wet = 0
        for cv in self.control_volumes:
            if isinstance(cv, WetControlVolume):
                hiw_list.append(cv.htp.hi)
                n_wet += 1
        if hiw_list:
            hiw_avg = sum(hiw_list) / n_wet
        else:
            hiw_avg = Q_(0.0, 'W / (m**2 * K)')
        return hiw_avg

    @property
    def eta_sd_avg(self) -> Quantity:
        """Average surface efficiency of dry part."""
        eta_sd_list = []
        n_dry = 0
        for cv in self.control_volumes:
            if isinstance(cv, DryControlVolume):
                eta_sd_list.append(cv.htp.eta_surf)
                n_dry += 1
        if eta_sd_list:
            eta_sd_avg = sum(eta_sd_list) / n_dry
        else:
            eta_sd_avg = Q_(0.0, 'frac')
        return eta_sd_avg

    @property
    def eta_sw_avg(self) -> Quantity:
        """Average surface efficiency of wet part."""
        eta_sw_list = []
        n_wet = 0
        for cv in self.control_volumes:
            if isinstance(cv, WetControlVolume):
                eta_sw_list.append(cv.htp.eta_surf)
                n_wet += 1
        if eta_sw_list:
            eta_sw_avg = sum(eta_sw_list) / n_wet
        else:
            eta_sw_avg = Q_(0.0, 'frac')
        return eta_sw_avg

    @property
    def Q_max(self) -> Quantity:
        """Theoretical maximum achievable heat transfer."""
        if isinstance(self.control_volumes[-1], WetControlVolume):
            hao_min = HumidAir(Tdb=self.coolant_in.T, RH=Q_(100, 'pct')).h
        else:
            hao_min = HumidAir(Tdb=self.coolant_in.T, W=self.air_stream.inlet.W).h
        hco_max = self.coolant_stream.type(T=self.air_stream.inlet.Tdb, P=self.coolant_stream.outlet.P).h
        Q_max1 = self.air_stream.m * (self.air_stream.inlet.h - hao_min)
        Q_max2 = self.coolant_stream.m * (hco_max - self.coolant_in.h)
        Q_max = min(Q_max1, Q_max2)
        return Q_max

    @property
    def eps(self) -> Quantity:
        """Air cooler effectiveness."""
        return self.Q / self.Q_max

    @property
    def Ao(self) -> Quantity:
        """Total external heat transfer surface area."""
        return self.coil_geometry.Ao

    @property
    def Aod(self) -> Quantity:
        """External heat transfer area of dry part."""
        Aod_list = [cv.delta_Ao for cv in self.control_volumes if isinstance(cv, DryControlVolume)]
        if Aod_list:
            return sum(Aod_list)
        else:
            return Q_(0, 'm ** 2')

    @property
    def Aow(self) -> Quantity:
        """Internal heat transfer area of wet part."""
        Aow_list = [cv.delta_Ao for cv in self.control_volumes if isinstance(cv, WetControlVolume)]
        if Aow_list:
            return sum(Aow_list)
        else:
            return Q_(0, 'm ** 2')

    @property
    def dPa(self) -> Quantity:
        """Air-side pressure drop."""
        Afa_to_Ac = 1 / self.coil_geometry.Ac_to_Afa
        Ao_to_Afa = self.coil_geometry.Ao_to_Afa
        Ao_to_Ac = Afa_to_Ac * Ao_to_Afa
        rho_in = self.air_stream.inlet.rho
        rho_out = self.air_stream.outlet.rho
        rho_avg = 0.5 * (rho_in + rho_out)
        G = self.air_stream.m / self.coil_geometry.Ac  # mass velocity
        f_fan_avg = sum(cv.get_fanning_friction_factor_air() for cv in self.control_volumes) / len(self.control_volumes)
        f1 = G ** 2 / (2 * rho_in)
        t1 = (1 + self.coil_geometry.Ac_to_Afa ** 2) * (rho_in / rho_out - 1)
        t2 = f_fan_avg * self.coil_geometry.Nr * Ao_to_Ac * rho_in / rho_avg
        dP = f1 * (t1 + t2)
        return dP

    class SectionPrinter:

        def __init__(self, n_segments: int):
            self.layout = {
                'section': (15,),
                'Ta': (15, 'degC', 3),
                'Wa': (20, 'g / kg', 3),
                'ha': (20, 'kJ / kg', 3),
                'Ts': (15, 'degC', 3),
                'Tc': (15, 'degC', 3),
                'hc': (20, 'kJ / kg', 3),
                'xc': (20, 'pct', 3)
            }
            self.final_index = n_segments - 1

        def print_header(self):
            header = ''
            for k, v in self.layout.items():
                header += f"{k}".ljust(v[0])
            logger.info(header)

        def _print_inlet_section(
            self,
            index: int,
            cv: Union[WetControlVolume, DryControlVolume]
        ) -> None:
            inlet_section = {
                'section': index + 1,
                'Ta': cv.air_stream.inlet.Tdb,
                'Wa': cv.air_stream.inlet.W,
                'ha': cv.air_stream.inlet.h,
                'Ts': cv.surface.Ti,
                'Tc': cv.coolant_stream.outlet.T,
                'hc': cv.coolant_stream.outlet.h,
                'xc': cv.coolant_stream.outlet.x
            }
            info = ''
            for k, v in self.layout.items():
                if k == 'section':
                    field = f"{inlet_section[k]}".ljust(v[0])
                else:
                    field = f"{inlet_section[k].to(v[1]):~P.{v[2]}f}".ljust(v[0])
                info += field
            logger.info(info)

        def _print_outlet_section(
            self,
            index: int,
            cv: Union[WetControlVolume, DryControlVolume]
        ):
            outlet_section = {
                'section': index + 2,
                'Ta': cv.air_stream.outlet.Tdb,
                'Wa': cv.air_stream.outlet.W,
                'ha': cv.air_stream.outlet.h,
                'Ts': cv.surface.To,
                'Tc': cv.coolant_stream.inlet.T,
                'hc': cv.coolant_stream.inlet.h,
                'xc': cv.coolant_stream.inlet.x
            }
            info = ''
            for k, v in self.layout.items():
                if k == 'section':
                    field = f"{outlet_section[k]}".ljust(v[0])
                else:
                    field = f"{outlet_section[k].to(v[1]):~P.{v[2]}f}".ljust(v[0])
                info += field
            logger.info(info)

        def print_section(
            self,
            index: int,
            cv: Union[WetControlVolume, DryControlVolume]
        ):
            self._print_inlet_section(index, cv)
            if index == self.final_index:
                self._print_outlet_section(index, cv)

    class SegmentPrinter:

        def __init__(self):
            self.layout = {
                'segment': (10,),
                'condition': (10,),
                'Ao': (15, 'm**2', 3),
                'Ta': (15, 'degC', 3),
                'Wa': (15, 'g / kg', 3),
                'ha': (15, 'kJ / kg', 3),
                'Ts': (15, 'degC', 3),
                'Tc': (15, 'degC', 3),
                'hc': (15, 'kJ / kg', 3),
                'xc': (15, 'pct', 3)
            }

        def print_header(self):
            header = ''
            for k, v in self.layout.items():
                header += f"{k}".ljust(v[0])
            logger.info(header)

        def print_segment(
            self,
            index: int,
            cv: Union[WetControlVolume, DryControlVolume]
        ) -> None:
            Ta_avg = 0.5 * (cv.air_stream.inlet.Tdb.to('K') + cv.air_stream.outlet.Tdb.to('K'))
            Wa_avg = 0.5 * (cv.air_stream.inlet.W + cv.air_stream.outlet.W)
            ha_avg = 0.5 * (cv.air_stream.inlet.h + cv.air_stream.outlet.h)
            Ts_avg = 0.5 * (cv.surface.Ti + cv.surface.To)
            Tc_avg = 0.5 * (cv.coolant_stream.outlet.T.to('K') + cv.coolant_stream.inlet.T.to('K'))
            hc_avg = 0.5 * (cv.coolant_stream.outlet.h + cv.coolant_stream.inlet.h)
            xc_avg = 0.5 * (cv.coolant_stream.outlet.x + cv.coolant_stream.inlet.x)
            segment = {
                'segment': index + 1,
                'condition': 'wet' if isinstance(cv, WetControlVolume) else 'dry',
                'Ao': cv.delta_Ao,
                'Ta': Ta_avg,
                'Wa': Wa_avg,
                'ha': ha_avg,
                'Ts': Ts_avg,
                'Tc': Tc_avg,
                'hc': hc_avg,
                'xc': xc_avg
            }
            info = ''
            for k, v in self.layout.items():
                if k == 'segment' or k == 'condition':
                    field = f"{segment[k]}".ljust(v[0])
                else:
                    field = f"{segment[k].to(v[1]):~P.{v[2]}f}".ljust(v[0])
                info += field
            logger.info(info)
