"""Physical constants in SI (CODATA 2018 exact / derived values).

Units policy:
    energy eV, length Angstrom, k in Angstrom^-1, velocity m/s,
    temperature K, 3D conductivity S/m.
"""

from __future__ import annotations

import math

# Exact SI constants (2019 SI redefinition)
HBAR_J_S: float = 1.054_571_817e-34  # J s, exact product of defined constants
E_CHARGE_C: float = 1.602_176_634e-19  # C, exact
KB_J_PER_K: float = 1.380_649e-23  # J/K, exact
M_ELECTRON_KG: float = 9.109_383_713_9e-31  # kg (CODATA 2022)

# Derived, in the canonical unit system of the package
EV_IN_J: float = E_CHARGE_C  # 1 eV in joules
KB_EV_PER_K: float = KB_J_PER_K / EV_IN_J  # Boltzmann constant, eV/K
HBAR_EV_S: float = HBAR_J_S / EV_IN_J  # hbar in eV s

ANGSTROM_M: float = 1.0e-10

# Conversion: (eV * Angstrom) / hbar -> m/s
# dH/dk [eV Ang] / hbar [eV s] * (angstrom in m) gives m/s.
EV_ANG_OVER_HBAR_TO_M_S: float = ANGSTROM_M / HBAR_EV_S

# Dimension check: C^2/(J s) = (A s)^2/(J s) = A/(V s) * m/m = S * m / m:
# A^2 s / J = (A/V) = S is wrong; A = C/s, so A^2 s^2 /(J s) = C A/(J) = A/V = S.
# Numerically: e^2/hbar = 2*pi * e^2/h ≈ 2.4341e-4 S is a *conductance* (sheet
# conductance unit). Multiplying a Berry-curvature BZ average [Ang^2] converted
# to m^2 would give S*m^2 which is NOT S/m; the 3D AHC conversion therefore
# picks up an explicit 1/length from the 3D BZ measure. This is handled in
# response/ahc.py, which centralizes all geometry factors.
E2_OVER_HBAR_SI: float = E_CHARGE_C**2 / HBAR_J_S  # sheet conductance unit, S
E2_OVER_H_SI: float = E_CHARGE_C**2 / (2.0 * math.pi * HBAR_J_S)  # conductance quantum, S
