from acetn.ipeps import Ipeps
from acetn.model import Model
from acetn.model.pauli_matrix import pauli_matrices

from functools import reduce
import toml
import argparse

# Layout: 4-site unit cell, formatted like
# +++++++++++++++++++++++++ 
# + |  \| + |  \| + |  \| +
# +-0---1-+-0---1-+-0---1-+ 
# +/|   | +/|   | +/|   | +
# + |   | + |   | + |   | +
# + |   |/+ |   |/+ |   |/+
# +-2---3 +-2---3 +-2---3 +
# + |  \| + |  \| + |  \| +
# +++++++++++++++++++++++++ 
# + |  \| + |  \| + |  \| +
# +-0---1-+-0---1-+-0---1-+ 
# +/|   | +/|   | +/|   | +
# + |   | + |   | + |   | +
# + |   |/+ |   |/+ |   |/+
# +-2---3 +-2---3 +-2---3 +
# + |  \| + |  \| + |  \| +
# +++++++++++++++++++++++++ 

class ShastrySutherlandModel(Model):
    def __init__(self, config):
        super().__init__(config)
 
    # --- operator helpers on the 4-slot (d=16) local space -------------------
    def _slot(self, P, i):
        """Pauli P on slot i of 4, identity elsewhere -> d=16 operator."""
        _, _, _, I = pauli_matrices(self.dtype, self.device)
        ops = [I, I, I, I]
        ops[i] = P
        return reduce(lambda a, b: a * b, ops)

    def _dot_on(self, i, j):
        """Spin-1/2 Heisenberg dot S_i . S_j on the SAME site (slots i,j)."""
        X, Y, Z, I = pauli_matrices(self.dtype, self.device)
        terms = []
        for P in (X, Y, Z):
            ops = [I, I, I, I]
            ops[i] = P
            ops[j] = P
            terms.append(reduce(lambda a, b: a * b, ops))
        return 0.25 * (terms[0] + terms[1] + terms[2])

    def _dot_bond(self, i, j):
        """Spin-1/2 Heisenberg dot: slot i on site 1, slot j on site 2."""
        X, Y, Z, I = pauli_matrices(self.dtype, self.device)
        terms = [self._slot(P, i) * self._slot(P, j) for P in (X, Y, Z)]
        return 0.25 * (terms[0] + terms[1] + terms[2])

    def one_site_observables(self, site):
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        obs = {}
        for sl in [0, 1, 2, 3]:
            for dirn, op in zip('xyz', [X,Y,Z]):
                obs[f'mag_{dirn}_{sl}']=self._slot(op, sl)

        obs['neel_01'] = Z*Z*I*I
        obs['neel_02'] = Z*I*Z*I
        obs['neel_03'] = Z*I*I*Z

        # Plaquette order parameter: <S.S> on the four edges of the empty
        # plaquette (intra-site). Strongly negative in the plaquette phase
        # (spins bind into intra-plaquette singlets), ~0 in the dimer phase
        # (spins bind into the inter-cell J' dimers instead). The plaquette-VBS
        # symmetry-breaking order parameter is the modulation of 'plaq' across
        # the 2x2 unit cell (compare the per-site values that measure() prints).
        obs['plaq_01'] = self._dot_on(0, 1)
        obs['plaq_02'] = self._dot_on(0, 2)
        obs['plaq_13'] = self._dot_on(1, 3)
        obs['plaq_23'] = self._dot_on(2, 3)
        obs['plaq'] = 0.25 * (obs['plaq_01'] + obs['plaq_02']
                              + obs['plaq_13'] + obs['plaq_23'])
        return obs

    def two_site_observables(self, bond):
        # Singlet (dimer) order parameter: <S.S> on the J' dimer bond, which lives
        # on the inter-cell +x / +y bond. -> -3/4 (pure singlet) deep in the dimer
        # phase; ~0 in the plaquette / Neel phases. Singlet fraction = 1/4 - <S.S>.
        match self.bond_direction(bond):
            case '+x':
                return {'dimer_ss': self._dot_bond(3, 0)}
            case '-x':
                return {'dimer_ss': self._dot_bond(0, 3)}
            case '+y':
                return {'dimer_ss': self._dot_bond(2, 1)}
            case '-y':
                return {'dimer_ss': self._dot_bond(1, 2)}


    def one_site_hamiltonian(self, site):
        J = self.params.get('J')
        # four edges of the empty plaquette (all J): 0-1, 0-2, 1-3, 2-3
        return J * (self._dot_on(0, 1) + self._dot_on(0, 2)
                    + self._dot_on(1, 3) + self._dot_on(2, 3))

    def two_site_hamiltonian(self, bond):
        J = self.params.get('J')
        Jp = self.params.get('Jp')
        J4 = self.params.get('J4')
        match self.bond_direction(bond):
            case '+x':
                # + |  \| + |
                # +-0---1-+-0
                # +/|   | +/|
                # + |ME | + |
                # + |   |/+ |
                # +-2---3 +-2
                # + |  \| + |
                return J*(self._dot_bond(1, 0) + self._dot_bond(3, 2)) + Jp*self._dot_bond(3, 0) + J4*self._dot_bond(1,2)
            case '-x':
                # | + |  \|
                # 1-+-0---1-
                # | +/|   |
                # | + | ME|
                # |/+ |   |/
                # 3 +-2---3
                # | + |  \|
                return J*(self._dot_bond(0, 1) + self._dot_bond(2, 3)) + Jp*self._dot_bond(0, 3) + J4*self._dot_bond(2,1)
            case '+y':
                # +-2---3 +
                # + |\  | +
                # +++++++++
                # + |  \| +
                # +-0---1-+
                # ..| ME| ...
                return J*(self._dot_bond(0, 2) + self._dot_bond(1, 3)) + Jp*self._dot_bond(1, 2) + J4*self._dot_bond(0,3)
            case '-y':
                # + | ME|/+
                # +-2---3 +
                # + |\  | +
                # +++++++++
                # + |  \| +
                # +-0---1-+
                return J*(self._dot_bond(2, 0) + self._dot_bond(3, 1)) + Jp*self._dot_bond(2, 1) + J4*self._dot_bond(3,0)
    
    def initial_site_state(self, site):
        """Sz=0 Neel product seed |down,up,up,down> on slots (0,1,2,3).

        Slot Neel signs follow (-1)^(x+y) for the empty-plaquette layout
        0=(0,1) 1=(1,1) 2=(0,0) 3=(1,0) -> (down, up, up, down). Cells translate
        by even vectors so the sublattice is preserved: the same vector on every
        site gives a globally consistent Neel state, and it lives in the Sz=0
        sector (2 up, 2 down) so imaginary-time evolution isn't frozen in the
        polarized sector like the default |0000> seed.
        """
        bits = [1, 0, 0, 1]                 # up=|0>, down=|1>
        idx = 0
        for b in bits:
            idx = idx * 2 + b               # -> computational basis index 9
        state = [0.0] * self.dim
        state[idx] = 1.0
        return state


def main(config, params):
    ipeps = Ipeps(config)
    ipeps.set_model(ShastrySutherlandModel, params)

    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

if __name__=='__main__':
    import argparse;
    parser=argparse.ArgumentParser()
    parser.add_argument("--bond_dimension","-D", type=int, default=3)
    parser.add_argument("--chi","-X", type=int, default=18)
    parser.add_argument("Jp",type=float,
                        help="Ratio Jp / J (note that J is hardcoded to be +1 AFM)")
    parser.add_argument("--J4",type=float, default=0.,
                        help="Parasitic coupling J4 (negative = FM)")
    parser.add_argument("--config",help="toml file containing the live parameters")

    ap = parser.parse_args()

    params = {'J':1,'Jp':ap.Jp,'J4':ap.J4}

    dims = {} if ap.config is None else toml.load(ap.config)
    dims['phys'] = 16
    dims['bond'] = ap.bond_dimension
    dims['chi'] = ap.chi

    config = {
        'dtype': "float64",
        'device': "cuda",
        'TN':{
            'dims': dims,
            'nx': 2,
            'ny': 2,
        },
    }

    main(config, params)
