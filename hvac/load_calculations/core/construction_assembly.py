from __future__ import annotations

from typing import Tuple, Dict, List
import math
from dataclasses import dataclass
from enum import Enum
from hvac import Quantity


Q_ = Quantity


@dataclass
class Material:
    """
    Dataclass that groups material properties.

    Attributes
    ----------
    k: Quantity
        Conductivity
    rho: Quantity
        Mass density
    c: Quantity
        Specific heat capacity
    """
    k: Quantity = Q_(0.0, 'W / (m * K)')
    rho: Quantity = Q_(0.0, 'kg / m ** 3')
    c: Quantity = Q_(0.0, 'J / (kg * K)')


@dataclass
class Geometry:
    """
    Dataclass that groups geometrical properties.

    Attributes
    ----------
    A: Quantity
        Area
    t: Quantity
        Thickness
    """
    A: Quantity = Q_(1.0, 'm ** 2')
    t: Quantity = Q_(0.0, 'm')


class HeatFlowDirection(Enum):
    HORIZONTAL = 'horizontal'
    UPWARDS = 'upwards'
    DOWNWARDS = 'downwards'


class MechanicalFastening:
    """Class for determining the effect of mechanical fasteners penetrating
    the insulation layer of a building component or element according to
    ISO 6946:2017, Annex F.3."""

    def __init__(self):
        self._t_ins: Quantity = Q_(10, 'cm')
        self._alpha: Quantity = Q_(0.8, 'frac')
        self._kf: Quantity = Q_(45, 'W / (m * K)')  # steel
        self._Af: Quantity = Q_(0, 'mm ** 2')
        self._nf: Quantity = Q_(3, '1 / m ** 2')

    @classmethod
    def create(
        cls,
        diameter: Quantity = Q_(3, 'mm'),
        number_per_unit_area: Quantity = Q_(3, '1 / m ** 2'),
        insulation_thickness: Quantity = Q_(10, 'cm'),
        length: Quantity = Q_(10, 'cm'),
        conductivity: Quantity = Q_(45, 'W / (m * K)')
    ):
        """
        Create `MechanicalFastening` object, used for calculating the correction
        factor due to mechanical fasteners penetrating the insulation layer of a
        building element according to ISO 6946:2017, F.3.

        Parameters
        ----------
        diameter: Quantity
            Diameter of the fastener.
        number_per_unit_area: int
            Number of fasteners per unit area.
        insulation_thickness: Quantity
            Thickness of the insulation layer.
        length: Quantity
            Length of the fastener that penetrates the insulation layer.
        conductivity: Quantity, default 45 W / (m.K) for steel
            Thermal conductivity of the fastener.

        Returns
        -------
        `MechanicalFastening` object. The correction factor can be accessed
        through property `correction_factor`.
        """
        fastener = cls()
        fastener._Af = math.pi * (diameter ** 2) / 4
        fastener._nf = number_per_unit_area
        fastener._alpha = 0.8 * length / insulation_thickness
        fastener._t_ins = insulation_thickness
        fastener._kf = conductivity
        return fastener

    @property
    def correction_factor(self) -> Quantity:
        """Get correction factor for thermal transmittance due to the presence
        of mechanical fasteners in the insulation layer of the building
        element."""
        U_cor_factor = self._alpha * self._kf * self._Af * self._nf / self._t_ins
        return U_cor_factor


class ThermalComponent:
    """Common base class for building components and alike, combining thermal
    resistance and capacitance into one object. Shouldn't be used directly.
    This object is created by the + operator and // operator.
    """
    
    def __init__(self):
        self.ID: str = ''
        self.geometry: Geometry = Geometry()
        self.R: Quantity = Q_(0.0, 'm ** 2 * K / W')
        self.C: Quantity = Q_(0.0, 'J / (K * m ** 2)')
        self.slices: int = 1
    
    @classmethod
    def create(cls, *args, **kwargs) -> ThermalComponent:
        therm_comp = cls()
        try:
            ID = args[0]
        except IndexError:
            ID = kwargs.get('ID', therm_comp.ID)
        try:
            geometry = args[1]
        except IndexError:
            geometry = kwargs.get('geometry', Geometry())
        try:
            R = args[2]
        except IndexError:
            R = kwargs.get('R', therm_comp.R)
        try:
            C = args[3]
        except IndexError:
            C = kwargs.get('C', therm_comp.C)
        therm_comp.ID = ID
        therm_comp.geometry = geometry
        therm_comp.R = R
        therm_comp.C = C
        return therm_comp

    def __add__(self, other: ThermalComponent) -> ThermalComponent:
        R = self.R + other.R
        C = self.C + other.C
        A = self.geometry.A
        t = self.geometry.t + other.geometry.t
        geometry = Geometry(A, t)
        ID = f"{self.ID} + {other.ID}"
        therm_obj = ThermalComponent.create(ID, geometry, R, C)
        return therm_obj

    def __radd__(self, other: ThermalComponent | int) -> ThermalComponent:
        if isinstance(other, int):
            return self
        else:
            return other.__add__(self)

    def __floordiv__(self, other: ThermalComponent) -> ThermalComponent:
        A = self.geometry.A + other.geometry.A
        U = (self.geometry.A / A) * self.U + (other.geometry.A / A) * other.U
        R = 1 / U
        C = (self.geometry.A / A) * self.C + (other.geometry.A / A) * other.C
        A = self.geometry.A + other.geometry.A
        t = self.geometry.t
        geometry = Geometry(A, t)
        ID = f"{self.ID} // {other.ID}"
        therm_comp = ThermalComponent.create(ID, geometry, R, C)
        return therm_comp

    def __str__(self):
        l1 = f"Component '{self.ID}'\n"
        l2 = f"\tR: {self.R.to('m ** 2 * K / W'):~P.2f}\n"
        l3 = f"\tC: {self.C.to('J / (m ** 2 * K)'):~P.2f}\n"
        l4 = f"\tslices: {self.slices}\n"
        return l1 + l2 + l3 + l4

    @property
    def U(self) -> Quantity:
        """Get specific transmittance."""
        return 1 / self.R

    @U.setter
    def U(self, v: Quantity) -> None:
        """Set specific transmittance."""
        self.R = 1 / v


class BuildingComponent(ThermalComponent):
    """Represents a flat building component made of solid material."""

    def __init__(self):
        super().__init__()
        self.material: Material | None = None

    @classmethod
    def create(
        cls,
        ID: str,
        geometry: Geometry,
        material: Material
    ) -> BuildingComponent:
        """
        Create `BuildingComponent` object.

        Parameters
        ----------
        ID: str
            Name to identify the building component.
        geometry: Geometry
            Data object that contains the area (A) and thickness (t) of the
            planar building component.
        material: Material
            Data object that contains the material properties of the building
            component (k, rho and c).

        Returns
        -------
        `BuildingComponent` object
        """
        build_comp = cls()
        build_comp.ID = ID
        build_comp.geometry = geometry
        build_comp.material = material
        build_comp.R = cls._calculate_resistance(material, geometry)
        build_comp.C = cls._calculate_capacitance(material, geometry)
        return build_comp
    
    @staticmethod
    def _calculate_resistance(material: Material, geometry: Geometry) -> Quantity:
        t = geometry.t
        k = material.k
        try:
            R = t / k
        except ZeroDivisionError:
            R = Q_(0.0, 'm ** 2 * K / W')
        return R
    
    @staticmethod
    def _calculate_capacitance(material: Material, geometry: Geometry) -> Quantity:
        t = geometry.t
        rho = material.rho
        c = material.c
        C = rho * c * t
        return C


class AirSpace(ThermalComponent):
    """Represents an unventilated airspace or air layer of which the thermal
    resistance is calculated according to ISO 6946:2017 annex D.2."""

    def __init__(self):
        super().__init__()

    @classmethod
    def create(
        cls,
        ID: str,
        geometry: Geometry,
        dT: Quantity,
        heat_flow_direction: HeatFlowDirection,
        Tmn: Quantity,
        surface_emissivities: Tuple[Quantity, Quantity] = (Q_(0.9, 'frac'), Q_(0.9, 'frac')),
        inclination_angle: Quantity | None = None
    ) -> AirSpace:
        """
        Create `AirSpace` object.

        Parameters
        ----------
        ID: str
            Name to identify the airspace in a building component.
        geometry: Geometry
        dT: Quantity
            The temperature difference across the airspace.
        heat_flow_direction: HeatFlowDirection
            Enum-type object indicating the direction of the heat flow
            perpendicular to the surface. Can be HORIZONTAL in case of a vertical
            surface, and UPWARDS or DOWNWARDS in case of a horizontal surface.
        Tmn: Quantity
            The mean thermodynamic temperature of the surfaces and the airspace.
        surface_emissivities: Tuple[Quantity, Quantity], (Q_(0.9, 'frac'), Q_(0.9, 'frac'))
            Emissivity of surfaces on both sides of the airspace.
        inclination_angle: Quantity
            Inclination angle of the airspace with respect to the horizontal.

        Returns
        -------
        `AirSpace` object.
        """
        air_space = cls()
        air_space.ID = ID
        air_space.geometry = geometry
        if inclination_angle is not None:
            a = inclination_angle.to('deg').m
            if 0 < a < 90:
                ha90 = cls._determine_ha(geometry, dT, HeatFlowDirection.HORIZONTAL)
                ha0 = cls._determine_ha(geometry, dT, HeatFlowDirection.UPWARDS)
                ha = ha90 + (ha90 - ha0) * (a - 90) / 90
            elif a == 0.0:
                ha = cls._determine_ha(geometry, dT, HeatFlowDirection.UPWARDS)
            else:
                ha = cls._determine_ha(geometry, dT, HeatFlowDirection.HORIZONTAL)
        else:
            ha = cls._determine_ha(geometry, dT, heat_flow_direction)
        hr = cls._determine_hr(Tmn, surface_emissivities)
        air_space.R = 1 / (ha + hr)
        return air_space

    @staticmethod
    def _determine_ha(
        geometry: Geometry,
        dT: Quantity,
        heat_flow_direction: HeatFlowDirection
    ) -> Quantity:
        t = geometry.t.to('m').m
        dT = dT.to('K').m
        ha = 0.0
        if dT <= 5:
            if heat_flow_direction == HeatFlowDirection.HORIZONTAL:
                ha = max(1.25, 0.025 / t)
            elif heat_flow_direction == HeatFlowDirection.UPWARDS:
                ha = max(1.95, 0.025 / t)
            elif heat_flow_direction == HeatFlowDirection.DOWNWARDS:
                ha = max(0.12 * (t ** -0.44), 0.025 / t)
        if dT > 5:
            if heat_flow_direction == HeatFlowDirection.HORIZONTAL:
                ha = max(0.73 * (dT ** (1 / 3)), 0.025 / t)
            elif heat_flow_direction == HeatFlowDirection.UPWARDS:
                ha = max(1.14 * (dT ** (1 / 3)), 0.025 / t)
            elif heat_flow_direction == HeatFlowDirection.DOWNWARDS:
                ha = max(0.09 * (dT ** 0.187) * (t ** -0.44), 0.025 / t)
        return Q_(ha, 'W / (m ** 2 * K)')

    @staticmethod
    def _determine_hr(Tmn: Quantity, surface_emissivities: Tuple[Quantity, Quantity]) -> Quantity:
        SIGMA = Q_(5.67e-8, 'W / (m ** 2 * K ** 4)')
        hro = 4 * SIGMA * Tmn.to('K') ** 3
        e1, e2 = surface_emissivities
        hr = hro / ((1 / e1) + (1 / e2) - 1)
        return hr


class SurfaceLayer(ThermalComponent):
    """
    Represents the internal or external surface film of a
    building component or element according to ISO 6946:2017, annex C.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def create(
        cls,
        ID: str,
        geometry: Geometry,
        heat_flow_direction: HeatFlowDirection,
        Tmn: Quantity = Q_(20, 'degC'),
        surface_emissivity: Quantity = Q_(0.9, 'frac'),
        internal_surface: bool = True,
        wind_speed: Quantity = Q_(4.0, 'm / s')
    ) -> SurfaceLayer:
        """
        Create a `SurfaceLayer` object.

        Parameters
        ----------
        ID: str
            Name to identify the surface layer.
        geometry: Geometry
            Data object with the area (A) and thickness (t) of the layer.
            Note: in case of surface layer, thickness has no meaning.
        heat_flow_direction: HeatFlowDirection
            Enum-type object indicating the direction of the heat flow
            perpendicular to the surface. Can be HORIZONTAL in case of a vertical
            surface, and UPWARDS or DOWNWARDS in case of a horizontal surface.
        Tmn: Quantity
            The mean thermodynamic temperature of the surface and of its
            surroundings (e.g. operative temperature for internal surfaces,
            external air temperature or sol-air temperature for external surfaces).
        surface_emissivity: Quantity, default 0.9
            The hemispherical emissivity of the surface.
        internal_surface: bool, default True
            Indicates if the surface is an internal surface or not.
        wind_speed: Quantity
            Wind speed. Only used for an external surface (`internal_surface
            = False`)

        Returns
        -------
        `SurfaceLayer` object.
        """
        surf_layer = cls()
        surf_layer.ID = ID
        surf_layer.geometry = geometry
        hr = cls._determine_hr(Tmn, surface_emissivity)
        hc = cls._determine_hc(heat_flow_direction, internal_surface, wind_speed)
        surf_layer.R = 1 / (hr + hc)
        return surf_layer

    @staticmethod
    def _determine_hr(Tmn: Quantity, e: Quantity) -> Quantity:
        SIGMA = Q_(5.67e-8, 'W / (m ** 2 * K ** 4)')
        hro = 4 * SIGMA * Tmn.to('K') ** 3
        hr = e * hro
        return hr

    @staticmethod
    def _determine_hc(heat_flow_direction: HeatFlowDirection, internal_surface: bool, v: Quantity) -> Quantity:
        if internal_surface:
            if heat_flow_direction == HeatFlowDirection.UPWARDS:
                hc = Q_(5.0, 'W / (m ** 2 * K)')
            elif heat_flow_direction == HeatFlowDirection.HORIZONTAL:
                hc = Q_(2.5, 'W / (m ** 2 * K)')
            else:
                hc = Q_(0.7, 'W / (m ** 2 * K)')
        else:
            hc = Q_(4 + 4 * v.to('m / s').m, 'W / (m ** 2 * K)')
        return hc


def apply_insulation_correction(
    insulation_layer: ThermalComponent,
    building_component: ThermalComponent,
    installation_quality_level: int = 0,
    mechanical_fastening: MechanicalFastening | None = None
) -> Quantity:
    """
    Apply correction to thermal resistance/transmittance to allow for the effects
    of air voids in insulation and mechanical fasteners penetrating the insulation
    layer according to ISO 6946:2017 Annex F.

    Parameters
    ----------
    insulation_layer: ThermalComponent
        The insulation layer within the building component or element.
    building_component: ThermalComponent
        The complete building component or element that contains the insulation
        layer.
    installation_quality_level: int, [0 (default), 1, or 2]
        Installation quality level of the insulation according to ISO 6946:2017,
        table F.1.
        - level 0: no air voids
        - level 1: air voids, but no air circulation
        - level 2: air voids with free air circulation between the warm and cold
        side of the insulation.
    mechanical_fastening: MechanicalFastening, optional
        Object that describes the mechanical fastening of the insulation layer.
        See `MechanicalFastening` class.

    Returns
    -------
    The correction term `delta_U` to be added to the thermal transmittance U of
    the building component or element.
    """
    correction_factor = Q_(0.0, 'W / (m ** 2 * K)')
    # correction for air voids
    if installation_quality_level == 1:
        correction_factor += Q_(0.01, 'W / (m ** 2 * K)')
    if installation_quality_level == 2:
        correction_factor += Q_(0.04, 'W / (m ** 2 * K)')
    # correction for mechanical fasteners
    if isinstance(mechanical_fastening, MechanicalFastening):
        correction_factor += mechanical_fastening.correction_factor
    # calculate delta_U
    R1 = insulation_layer.R
    Rtot = building_component.R
    delta_U = correction_factor * (R1 / Rtot) ** 2
    return delta_U


class ConstructionAssembly:
    """
    A flat construction assembly is composed of layered `ThermalComponent` objects (
    `BuildingComponent`, `AirSpace`, and/or `SurfaceLayer` objects).
    """

    def __init__(self):
        self.ID: str = ''
        self.layers: Dict[str, ThermalComponent] = {}
        self._thermal_component: ThermalComponent = ThermalComponent()

    @classmethod
    def create(cls, ID: str, layers: List[ThermalComponent]) -> ConstructionAssembly:
        """
        Create `ConstructionAssembly` object.

        Parameters
        ----------
        ID: str
            Name to identify the construction assembly
        layers: list of `ThermalComponent` objects
            A construction assembly is thought to be composed of successive layers,
            arranged from the outer to the inner environment.

        Returns
        -------
        `ConstructionAssembly` object
        """
        construction_assembly = cls()
        construction_assembly.ID = ID
        construction_assembly.layers = {layer.ID: layer for layer in layers}
        construction_assembly._thermal_component = sum(layers)
        return construction_assembly

    def apply_insulation_correction(
        self,
        insulation_layer_ID: str,
        insulation_level: int = 0,
        mechanical_fastening: MechanicalFastening | None = None
    ) -> None:
        """
        Apply a correction to the thermal resistance and transmittance of the
        construction assembly taking air voids and/or mechanical fasteners in the
        insulation layer into account.

        Parameters
        ----------
        insulation_layer_ID: str
            The ID of the insulation layer in the construction assembly.
        insulation_level: int, [0 (default), 1, 2]
            Insulation level according to ISO 6946:2017, Annex F, table F.1:
            - Level 0: No air voids within the insulation, or where only minor
            air voids are present that have no significant effect on the thermal
            transmittance.
            - Level 1: Air gaps bridging between the hot and cold side of the
            insulation, but not causing air circulation between the warm and
            cold side of the insulation.
            - Level 2: Air gaps bridging between the hot and cold side of the
            insulation, combined with cavities resulting in free air circulation
            between the warm and cold sides of the insulation.
        mechanical_fastening: MechanicalFastening, optional
            Object that describes the mechanical fastening of the insulation layer.
            See `MechanicalFastening` class.

        Returns
        -------
        None.
        """
        insulation_layer = self.layers.get(insulation_layer_ID)
        if insulation_layer:
            delta_U = apply_insulation_correction(
                insulation_layer, self._thermal_component,
                insulation_level, mechanical_fastening
            )
            insulation_layer.U += delta_U
            # also update R of construction assembly
            self._thermal_component.U += delta_U

    def apply_ventilated_layer_correction(
        self,
        ventilated_layer_ID: str,
        area_of_openings: Quantity,
        heat_flow_direction: HeatFlowDirection,
        Tmn: Quantity,
        surface_emissivity: Quantity = Q_(0.9, 'frac'),
    ) -> None:
        """
        Apply a correction to the thermal resistance and transmittance of the
        construction assembly due to a slightly or well-ventilated air layer
        according to ISO 6946:2017 6.9.3 and 6.9.4.

        Parameters
        ----------
        ventilated_layer_ID: str
            ID of the ventilated air layer.
        area_of_openings: Quantity
            A slightly ventilated air layer is one in which there is provision
            for limited air flow through it from the external environment by
            openings of area, A_ve, within the following ranges:
            ??? >500 mm2 but <1 500 mm2 per metre of length (in the horizontal
            direction) for vertical air layers;
            ??? >500 mm2 but <1 500 mm2 per square metre of surface area for
            horizontal air layers.
            A well-ventilated air layer is one for which the openings between
            the air layer and the external environment are equal to or exceed
            ??? 1 500 mm2 per metre of length (in the horizontal direction) for
            vertical air layers, and
            ??? 1 500 mm2 per square of metre of surface area for horizontal air
            layers.
        heat_flow_direction: HeatFlowDirection
            Enum-type object indicating the direction of the heat flow
            perpendicular to the surface. Can be HORIZONTAL in case of a vertical
            surface, and UPWARDS or DOWNWARDS in case of a horizontal surface.
        Tmn: Quantity
            External air temperature.
        surface_emissivity: Quantity
            The hemispherical emissivity of the surface adjacent to the
            ventilated air layer.

        Returns
        -------
        None.
        """
        # get layers between interior surface and ventilated air layer
        i = list(self.layers.keys()).index(ventilated_layer_ID) + 1
        interior_layers = list(self.layers.values())[i:]
        # sum the interior layers (returns ThermalComponent object)
        interior_part = sum(interior_layers)
        # add external surface layer for still air to the internal part
        ext_surf_layer = SurfaceLayer.create(
            ID='',
            geometry=Geometry(),
            heat_flow_direction=heat_flow_direction,
            Tmn=Tmn,
            surface_emissivity=surface_emissivity,
            internal_surface=False,
            wind_speed=Q_(0, 'm / s')
        )
        # get total thermal resistance of building component with
        # well-ventilated air layer.
        R_ve = interior_part.R + ext_surf_layer.R
        # get total thermal resistance of building component with unventilated
        # air layer
        R_nve = self._thermal_component.R
        A_ve = area_of_openings.to('mm ** 2').m
        if 500 < A_ve < 1500:
            Rtot = (1500 - A_ve) / 1000 * R_nve + (A_ve - 500) / 1000 * R_ve
            self._thermal_component.R = Rtot
        elif A_ve >= 1500:
            self._thermal_component.R = R_ve

    @property
    def R(self) -> Quantity:
        """Get thermal resistance of construction assembly per unit area."""
        return self._thermal_component.R

    @property
    def U(self) -> Quantity:
        """Get thermal transmittance of construction assembly per unit area."""
        return 1 / self._thermal_component.R

    @property
    def R_surf_ext(self) -> Quantity:
        """Get the surface resistance at the exterior side."""
        exterior_surface_layer = list(self.layers.values())[0]
        return exterior_surface_layer.R

    def __str__(self):
        _str = ''
        for comp_str in (str(layer) for layer in self.layers.values()):
            _str += comp_str
        return _str
