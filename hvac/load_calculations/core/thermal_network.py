from typing import List, Optional, Callable, Dict, Tuple
from abc import ABC, abstractmethod
import numpy as np
from hvac import Quantity
from .construction_assembly import ConstructionAssembly, ThermalComponent


Q_ = Quantity


class AbstractNode(ABC):

    def __init__(
        self,
        ID: str,
        R_list: List[Quantity],
        C: Optional[Quantity] = None,
        A: Optional[Quantity] = None,
        T_input: Optional[Callable[[float], float]] = None,
        Q_input: Optional[Callable[[float], float]] = None
    ):
        self.ID = ID
        self.R = [R.to('m ** 2 * K / W').m for R in R_list]
        self.C = C.to('J / (K * m ** 2)').m if C is not None else None
        self._A = A.to('m ** 2').m if A is not None else None
        self.T_input = T_input
        self.Q_input = Q_input
        self.dt: float = 0.0

    @abstractmethod
    def get_coefficients(self) -> Dict[str, float]:
        """Get coefficients of the LHS of the node equation."""
        ...

    @abstractmethod
    def get_input(self, k: int, T_node_prev: Tuple[float, float]) -> float:
        """
        Get RHS of the node equation.

        Parameters
        ----------
        k :
            Current time index.
        T_node_prev :
            Tuple of the two previous node temperatures. First element at time
            index k-1 and second element at k-2.
        """
        ...

    @property
    def A(self) -> Quantity:
        return Q_(self._A, 'm ** 2')

    @A.setter
    def A(self, v: Quantity) -> None:
        self._A = v.to('m ** 2').m

    def __str__(self):
        l1 = f"Node {self.ID}:\n"
        l2 = "\tR-links: " + str([round(R, 2) for R in self.R]) + "\n"
        try:
            l3 = f"\tC: {self.C:.2f}\n"
        except TypeError:
            l3 = "\tC: None\n"
        return l1 + l2 + l3


class BuildingMassNode(AbstractNode):
    """
    A building mass node is situated within the mass of the construction assembly and
    is connected with a preceding node and a next node.

    Parameter `R_list` is a list of two resistances in the given order:
    1. the conduction resistance between this node and the preceding node, and
    2. the conduction resistance between this node and the next node.
    """
    def get_coefficients(self) -> Dict[str, float]:
        return {
            'i, i-1': 1.0 / (self.R[0] * self.C),
            'i, i': -1.0 / self.C * sum([1 / R for R in self.R]) - 1.5 / self.dt,
            'i, i+1': 1.0 / (self.R[1] * self.C)
        }

    def get_input(self, k: int, T_node_prev: Tuple[float, float]) -> float:
        return -2.0 / self.dt * T_node_prev[0] + 1.0 / (2.0 * self.dt) * T_node_prev[1]


class ExteriorSurfaceNode(AbstractNode):
    """
    An exterior surface node is the first node of the 1D thermal network at the
    exterior side of the construction assembly that is connected to the sol-air
    temperature.

    Parameter `T_input` at instantiation of the node must be assigned a function
    that returns the sol-air temperature at each time moment k.

    Parameter `R_list` is a list of two resistances in the given order:
    1. The first resistance links the node to the sol-air temperature.
    2. The second resistance links the node to the next, internal node
    (which is an instance of class `BuildingMassNode`).
    """
    def get_coefficients(self) -> Dict[str, float]:
        return {
            'i, i': -1 / self.C * sum([1 / R for R in self.R]) - 1.5 / self.dt,
            'i, i+1': 1 / (self.R[1] * self.C)
        }

    def get_input(self, k: int, T_node_prev: Optional[Tuple[float, float]] = None) -> float:
        return (
            -2.0 / self.dt * T_node_prev[0] + 1.0 / (2.0 * self.dt) * T_node_prev[1]
            - self.T_input(k * self.dt) / (self.C * self.R[0])
        )


class InteriorSurfaceNode(AbstractNode):
    """
    An interior surface node is the last node of the 1D thermal network of a
    construction assembly. It is situated on the surface of the construction assembly and
    as such has no thermal capacity.

    The interior surface node can be connected to the zone air node through a
    convective resistance and/or to a thermal storage mass node within the zone
    through a radiative resistance.

    Parameter `T_input` at instantiation of the node must be assigned a function
    that returns the zone air temperature at each time moment *t* in units of
    seconds.

    Parameter `R_list` is a list of 2 or 3 resistances in the given order:
    1. the conductive resistance between the interior surface node and the
    preceding building mass node;
    2. the convective resistance between the interior surface node and the zone
    air node;
    3. the optionally radiative resistance between the interior surface node and
    the thermal storage node (if present).
    """
    def get_coefficients(self) -> Dict[str, float]:
        if len(self.R) == 3:
            # True if there is also a ThermalStorageNode present
            return {
                'i, i-1': 1.0 / self.R[0],
                'i, i': -sum([1 / R for R in self.R]),
                'i, -1': 1.0 / self.R[2]
            }
        else:
            return {
                'i, i-1': 1.0 / self.R[0],
                'i, i': -sum([1 / R for R in self.R]),
            }

    def get_input(self, k: int, T_node_prev: Tuple[float, float]) -> float:
        return -self.T_input(k * self.dt) / self.R[1]


class ThermalStorageNode(AbstractNode):
    """
    A thermal storage node represents the thermal storage mass of a zone. This
    node is connected to the zone air node and to the interior surface node of
    each construction assembly that surrounds the zone.

    Parameter `T_input` at instantiation of the node must be assigned a function
    that returns the zone air temperature at each time moment *t* in units of
    seconds.

    Heat can also be added directly to the thermal storage node (from radiative
    internal gains and transmitted solar heat gain through windows). The heat
    input to the thermal storage node at any time instance *t* (in units of
    seconds) comes from the function that was assigned to parameter `Q_input`
    at the instantiation of the `ThermalStorageNode` object.

    The parameter `R_list` must first list all the radiative resistances between
    the thermal storage mass node and the interior surface nodes. The last
    element of `R_links` must be the convective resistance between the thermal
    storage mass node and the zone air node.
    """
    def get_coefficients(self) -> Dict[str, float]:
        a = {f'i, {j}': 1.0 / (self.C * self.R[j - 1]) for j in range(1, len(self.R))}
        a['i, i'] = -1.0 / self.C * sum([1 / R for R in self.R]) - 1.5 / self.dt
        return a

    def get_input(self, k: int, T_node_prev: Tuple[float, float]) -> float:
        b = (
            -2.0 / self.dt * T_node_prev[0] + 1.0 / (2.0 * self.dt) * T_node_prev[1]
            - self.T_input(k * self.dt) / (self.C * self.R[-1])
        )
        if self.Q_input is None:
            return b
        else:
            return b - (self.Q_input(k * self.dt) / self._A) / self.C


class ThermalNetwork:

    def __init__(self, nodes: List[AbstractNode]):
        self.nodes = nodes
        self._T_ext: Callable[[float], float] | None = None
        self._T_int: Callable[[float], float] | None = None
        self._T_node_table: List[List[float]] | None = None

    @property
    def T_ext(self) -> Callable[[float], float] | None:
        return self._T_ext

    @T_ext.setter
    def T_ext(self, fun: Callable[[float], float]) -> None:
        """Set the temperature input function at the exterior side."""
        self._T_ext = fun
        self.nodes[0].T_input = self._T_ext

    @property
    def T_int(self) -> Callable[[float], float] | None:
        return self._T_int

    @T_int.setter
    def T_int(self, fun: Callable[[float], float]) -> None:
        """Set the temperature input function at the interior side."""
        self._T_int = fun
        self.nodes[-1].T_input = self._T_int

    @property
    def T_node_table(self) -> List[List[Quantity]]:
        tbl = [[Q_(T, 'degC') for T in row] for row in self._T_node_table]
        return tbl

    @T_node_table.setter
    def T_node_table(self, tbl: List[List[float]]) -> None:
        self._T_node_table = tbl

    @property
    def R_ext(self) -> Quantity:
        """Get the thermal resistance at the exterior surface."""
        R_ext = self.nodes[0].R[0]
        return Q_(R_ext, 'm ** 2 * K / W')

    @property
    def R_int(self) -> Quantity:
        """Get the thermal resistance at the interior surface."""
        R_int = self.nodes[-1].R[-1]
        return Q_(R_int, 'm ** 2 * K / W')

    def __str__(self):
        _str = ''
        for node_str in (str(node) for node in self.nodes):
            _str += node_str
        return _str


class ThermalNetworkBuilder:
    """Builds the linear thermal network of a construction assembly."""

    @classmethod
    def build(cls, construction_assembly: ConstructionAssembly) -> ThermalNetwork | None:
        """
        Get the linear thermal network of the construction assembly.

        Returns
        -------
        `ThermalNetwork` object.

        Notes
        -----
        The layers of the construction assembly must be arranged from
        the exterior surface towards the interior surface.
        """
        thermal_network = cls._compose(list(construction_assembly.layers.values()))
        reduced_thermal_network = cls._reduce(thermal_network)
        nodes = cls._transform(reduced_thermal_network)
        return ThermalNetwork(nodes)

    @staticmethod
    def _compose(layers: List[ThermalComponent]) -> List[Quantity]:
        # create a list of resistors and capacitors
        thermal_network = []
        for layer in layers:
            n = layer.slices
            R_slice = layer.R / (2 * n)
            C_slice = layer.C / n
            slices = [R_slice, C_slice, R_slice] * n
            thermal_network.extend(slices)
        return thermal_network

    @staticmethod
    def _reduce(thermal_network: List[Quantity]) -> List[Quantity]:
        # sum adjacent resistors between capacitors
        R_dummy = Q_(0, 'm ** 2 * K / W')
        reduced_thermal_network = []
        R = thermal_network[0]
        for i in range(1, len(thermal_network)):
            if R_dummy.check(thermal_network[i].dimensionality):
                # thermal_network[i] is a resistance
                R += thermal_network[i]
            else:
                # thermal_network[i] is a capacitance: only keep C
                # if it is > 0, unless for the last C (to set the interior
                # surface node, see _transform)
                if thermal_network[i].m > 0 or i == len(thermal_network) - 2:
                    reduced_thermal_network.append(R)
                    reduced_thermal_network.append(thermal_network[i])
                    R = Q_(0.0, 'm ** 2 * K / W')
        if R.m > 0:
            reduced_thermal_network.append(R)
        return reduced_thermal_network

    @staticmethod
    def _transform(reduced_thermal_network: List[Quantity]) -> List[AbstractNode] | None:
        # create list of nodes, starting at the exterior surface towards the interior surface
        if len(reduced_thermal_network) >= 5:
            i = 1
            node_index = 1
            nodes = []
            while True:
                if i == 1:
                    node = ExteriorSurfaceNode(
                        ID=f'N{node_index}',
                        R_list=[reduced_thermal_network[i - 1], reduced_thermal_network[i + 1]],
                        C=reduced_thermal_network[i]
                    )
                    nodes.append(node)
                    i += 2
                    node_index += 1
                elif i == len(reduced_thermal_network) - 2:
                    node = InteriorSurfaceNode(
                        ID=f'N{node_index}',
                        R_list=[reduced_thermal_network[i - 1], reduced_thermal_network[i + 1]]
                    )
                    nodes.append(node)
                    break
                else:
                    node = BuildingMassNode(
                        ID=f'N{node_index}',
                        R_list=[reduced_thermal_network[i - 1], reduced_thermal_network[i + 1]],
                        C=reduced_thermal_network[i]
                    )
                    nodes.append(node)
                    i += 2
                    node_index += 1
            return nodes
        return None


class ThermalNetworkSolver:
    """Solve a linear thermal network model. The solve method can be applied
    to the thermal network model of construction assembly or to the thermal network
    of a space."""
    _nodes: List[AbstractNode] = None
    _dt: float = 0.0
    _A: np.ndarray = None
    _B: np.ndarray = None
    _Tn_table: List[List[float]] = None
    _k_max: int = 0

    @classmethod
    def _init(cls, thermal_network: ThermalNetwork, dt_hr: float = 1.0) -> None:
        """
        Initialize the solver with the input data.

        Parameters
        ----------
        thermal_network: ThermalNetwork
            The thermal network model of a construction assembly or space.
        dt_hr: float, default 1.0
            The time step of the calculations expressed in hours.

        Returns
        -------
        None
        """
        cls._nodes = thermal_network.nodes
        cls._dt = dt_hr * 3600
        n = len(cls._nodes)
        cls._A = np.zeros((n, n))
        cls._B = np.zeros((n, 1))
        cls._Tn_table = [
            [0.0] * n,  # initial node temperatures at k-2
            [0.0] * n   # initial node temperatures at k-1
        ]
        cls._k_max = int(24.0 / dt_hr)

    @classmethod
    def _build_matrix_A(cls) -> None:
        """
        Build the coefficient matrix A from the thermal network model.

        Returns
        -------
        None
        """
        int_surf_node_indexes = []
        for i, node in enumerate(cls._nodes):
            node.dt = cls._dt
            a = node.get_coefficients()
            if isinstance(node, ExteriorSurfaceNode):
                cls._A[i, i] = a['i, i']
                cls._A[i, i + 1] = a['i, i+1']
            elif isinstance(node, BuildingMassNode):
                cls._A[i, i - 1] = a['i, i-1']
                cls._A[i, i] = a['i, i']
                cls._A[i, i + 1] = a['i, i+1']
            elif isinstance(node, InteriorSurfaceNode):
                cls._A[i, i - 1] = a['i, i-1']
                cls._A[i, i] = a['i, i']
                try:
                    cls._A[i, -1] = a['i, -1']
                except KeyError:
                    # the internal surface node is not connected to a 
                    # thermal storage node
                    pass
                else:
                    int_surf_node_indexes.append(i)
            elif isinstance(node, ThermalStorageNode):
                cls._A[i, i] = a['i, i']
                for p, j in enumerate(int_surf_node_indexes):
                    cls._A[i, j] = a[f'i, {p + 1}']
            else:
                pass
    
    @classmethod
    def _update_matrix_B(cls, k: int) -> None:
        """
        Update the input matrix B at time index k.
        """
        for i, node in enumerate(cls._nodes):
            cls._B[i] = node.get_input(k, (cls._Tn_table[-1][i], cls._Tn_table[-2][i]))

    @classmethod
    def _solve(cls, n_cycles: int) -> None:
        for i in range(n_cycles):
            for k in range(cls._k_max):
                cls._update_matrix_B(k)
                Tn_array = np.linalg.solve(cls._A, cls._B)
                Tn_list = np.transpose(Tn_array)[0].tolist()
                cls._Tn_table.append(Tn_list)
            if i < n_cycles - 1:
                # if not the final diurnal cycle: only keep the node temperatures
                # at the last two time steps as initial values for the next cycle.
                cls._Tn_table = cls._Tn_table[-2:]
        # remove the first two rows of the current temperature node table, which
        # are still from the previous cycle and served as initial values for the
        # last cycle.
        cls._Tn_table = cls._Tn_table[2:]

    @classmethod
    def solve(cls, thermal_network: ThermalNetwork, dt_hr: float = 1.0, n_cycles: int = 6) -> ThermalNetwork:
        """
        Solve the linear thermal network model of a construction assembly or space.

        Parameters
        ----------
        thermal_network: ThermalNetwork
            Thermal network model of construction assembly or space.
        dt_hr: float, default 1.0
            The time step of the calculations expressed in hours.
        n_cycles: int, default 6
            The number of diurnal cycles before the resulting node temperatures
            are returned.

        Returns
        -------
        A list of lists. Each list contains the node temperatures at the time
        index that corresponds with the index of the main list.
        """
        cls._init(thermal_network, dt_hr)
        cls._build_matrix_A()
        cls._solve(n_cycles)
        thermal_network.T_node_table = cls._Tn_table
        return thermal_network
