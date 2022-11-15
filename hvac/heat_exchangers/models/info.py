from typing import Dict, Union
import pandas as pd
from .air_cooler import AirCooler
from .air_condenser import AirCondenser
from .core.control_volumes import WetControlVolume


def get_heat_transfer_overview(
        heat_exchanger: Union[AirCooler, AirCondenser],
        units: Dict[str, str] | None = None
) -> pd.DataFrame:
    if units is None:
        units = {
            'A': 'm ** 2',
            'Q_dot': 'kW',
            'm_dot': 'kg / hr'
        }
    d = {
        'Ao': [],
        'Qs': [],
        'Ql': [],
        'Q': [],
        'mw': []
    }
    for cv in heat_exchanger.control_volumes:
        d['Ao'].append(cv.delta_Ao.to(units['A']).m)
        d['Qs'].append(cv.ht.delta_Qs.to(units['Q_dot']).m if cv.ht.delta_Qs is not None else float('nan'))
        d['Ql'].append(cv.ht.delta_Ql.to(units['Q_dot']).m if cv.ht.delta_Ql is not None else float('nan'))
        d['Q'].append(cv.ht.delta_Q.to(units['Q_dot']).m if cv.ht.delta_Q is not None else float('nan'))
        d['mw'].append(cv.ht.delta_mw.to(units['m_dot']).m if cv.ht.delta_mw is not None else float('nan'))
    df = pd.DataFrame(d)
    return df


def get_heat_transfer_params_overview(
    heat_exchanger: Union[AirCooler, AirCondenser],
    units: Dict[str, str] | None = None
) -> pd.DataFrame:
    if units is None:
        units = {
            'A': 'm ** 2',
            'he_dry': 'W / (m ** 2 * K)',
            'he_wet': 'kg / (m ** 2 * s)',
            'hi': 'W / (m ** 2 * K)',
            'Uo_dry': 'W / (m ** 2 * K)',
            'Uo_wet': 'kg / (m ** 2 * s)'
        }
    d = {
        'Ao': [],
        'he': [],
        'hi': [],
        'Uo': [],
    }
    for cv in heat_exchanger.control_volumes:
        d['Ao'].append(cv.delta_Ao.to(units['A']).m)
        d['hi'].append(cv.htp.hi.to(units['hi']).m)
        if isinstance(cv, WetControlVolume):
            d['he'].append(cv.htp.he.to(units['he_wet']).m)
            d['Uo'].append(cv.htp.Uo.to(units['Uo_wet']).m)
        else:
            d['he'].append(cv.htp.he.to(units['he_dry']).m)
            d['Uo'].append(cv.htp.Uo.to(units['Uo_dry']).m)
    df = pd.DataFrame(d)
    return df
