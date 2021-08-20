import sys

sys.path.append("/home/dewit/QMCTorch")
# sys.path.append("/home/matthijs/esc/QMCTorch")

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.utils import set_torch_double_precision

set_torch_double_precision()


class SlaterJastrowWrapperH2(SlaterJastrow):
    def __init__(self):
        mol = Molecule(atom='H 0 0 -0.3561; H 0 0 0.3561',
                       calculator='pyscf', basis='sto-3g', unit='angs')
        super().__init__(mol, kinetic='jacobi', configs='ground_state')


class SlaterJastrowWrapperCH4(SlaterJastrow):
    def __init__(self):
        mol = Molecule(atom='methane.xyz', unit='angs',
                       calculator='pyscf', basis='sto-3g', name='methane')
        super().__init__(mol, kinetic='jacobi', configs='ground_state')
