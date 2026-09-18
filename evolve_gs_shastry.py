from acetn.ipeps import Ipeps
from acetn.model import Model
from acetn.model.pauli_matrix import pauli_matrices

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
 
    def one_site_hamiltonian(self, site):
        J = self.params.get('J')
        Jp = self.params.get('Jp')
        X, Y, Z, I = pauli_matrices(self.dtype, self.device)

        res = J*(X*X*I*I + Y*Y*I*I + Z*Z*I*I) # 0 - 1
        res += J*(X*I*X*I + Y*I*Y*I + Z*I*Z*I) # 0 - 2
        res += J*(I*X*I*X + I*Y*I*Y + I*Z*I*Z) # 3 - 1
        res += J*(I*I*X*X + I*I*Y*Y + I*I*Z*Z) # 3 - 2

        return res*0.25


    def two_site_hamiltonian(self, bond):
        J = self.params.get('J')
        Jp = self.params.get('Jp')
        X, Y, Z, I = pauli_matrices(self.dtype, self.device)
        res=None

        match self.bond_direction(bond):
            case '+x':
                # + |  \| + |
                # +-0---1-+-0
                # +/|   | +/|
                # + |ME | + |
                # + |   |/+ |
                # +-2---3 +-2
                # + |  \| + |

                res = J*(
                          (I*X*I*I)*(X*I*I*I)
                        + (I*Y*I*I)*(Y*I*I*I)
                        + (I*Z*I*I)*(Z*I*I*I)
                        )                
                res += J*(              
                          (I*I*I*X)*(I*I*X*I)
                        + (I*I*I*Y)*(I*I*Y*I)
                        + (I*I*I*Z)*(I*I*Z*I)
                        )                
                res += Jp*(             
                          (I*I*I*X)*(X*I*I*I)
                        + (I*I*I*Y)*(Y*I*I*I)
                        + (I*I*I*Z)*(Z*I*I*I)
                        )
            case '-x':
                # | + |  \| 
                # 1-+-0---1-
                # | +/|   | 
                # | + | ME| 
                # |/+ |   |/
                # 3 +-2---3 
                # | + |  \| 
                res = J*(
                          (X*I*I*I)*(I*X*I*I)
                        + (Y*I*I*I)*(I*Y*I*I)
                        + (Z*I*I*I)*(I*Z*I*I)
                        )                    
                res += J*          (        
                          (I*I*X*I)*(I*I*I*X)
                        + (I*I*Y*I)*(I*I*I*Y)
                        + (I*I*Z*I)*(I*I*I*Z)
                        )                    
                res += Jp          *(       
                          (X*I*I*I)*(I*I*I*X)
                        + (Y*I*I*I)*(I*I*I*Y)
                        + (Z*I*I*I)*(I*I*I*Z)
                        )
            case '+y':
                # +-2---3 +
                # + |  \| +
                # +++++++++
                # + |  \| +
                # +-0---1-+
                # ..| ME| ...
                res = J*(
                          (X*I*I*I)*(I*I*X*I)
                        + (Y*I*I*I)*(I*I*Y*I)
                        + (Z*I*I*I)*(I*I*Z*I)
                        )
                res += J*(
                          (I*X*I*I)*(I*I*I*X)
                        + (I*Y*I*I)*(I*I*I*Y)
                        + (I*Z*I*I)*(I*I*I*Z)
                        )
                res += Jp*(
                          (I*X*I*I)*(I*I*X*I)
                        + (I*Y*I*I)*(I*I*Y*I)
                        + (I*Z*I*I)*(I*I*Z*I)
                        )
            case '-y':
                # + | ME|/+
                # +-2---3 +
                # + |  \| +
                # +++++++++
                # + |  \| +
                # +-0---1-+
                res = J*(
                          (I*I*X*I)*(X*I*I*I)
                        + (I*I*Y*I)*(Y*I*I*I)
                        + (I*I*Z*I)*(Z*I*I*I)
                        )           
                res += J*          (
                          (I*I*I*X)*(I*X*I*I)
                        + (I*I*I*Y)*(I*Y*I*I)
                        + (I*I*I*Z)*(I*Z*I*I)
                        )           
                res += Jp          *(
                          (I*I*X*I)*(I*X*I*I)
                        + (I*I*Y*I)*(I*Y*I*I)
                        + (I*I*Z*I)*(I*Z*I*I)
                        )

        return res*0.25 # change units to use physical spin length
 

def main(config):
    ipeps = Ipeps(config)
    ipeps.set_model(ShastrySutherlandModel, {'J':0.2,'Jp':1.0})

    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

if __name__=='__main__':
    import argparse;
    parser=argparse.ArgumentParser()
    parser.add_argument("--bond_dimension","-D", type=int, default=3)
    parser.add_argument("--chi","-X", type=int, default=18)

    ap = parser.parse_args()

    dims = {}
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

    main(config)
