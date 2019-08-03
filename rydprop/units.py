"""
Pint quantities for converting au to other units
"""
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.setup_matplotlib(True)
ureg.enable_contexts('sp')

ureg.define('bohr = (5.2917721092 * 10**-11) * meter = a0')
ureg.define('e_field_au = hartree/ (e*bohr) = au_e')
ureg.define('b_field_au = hbar/ (e*bohr**2) = au_b')
ureg.define('au_time = hbar/ hartree')