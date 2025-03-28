import numpy as np

SDIG = 3
ATOL = 1e-6

def dipolar_formfactor(Q):
    A_j0, a_j0, B_j0, b_j0, C_j0, c_j0, D_j0 = [-0.3094, 0.0274, 0.3680, 17.0355, 0.6559, 6.5236, 0.2856]
    s = Q / (4*np.pi)
    ff_j0 = A_j0 * np.exp(-a_j0*s**2) + B_j0 * np.exp(-b_j0*s**2) + C_j0 * np.exp(-c_j0*s**2) + D_j0
    return ff_j0

class cr2o3_polarimetry():
    def __init__(self, SA, SB, dom="out"):
        self.gyro=1.913     # gyromagnetic ratio of the neutron
        self.r0=2.818       # classical electron radius (fm)
        self.SA=SA
        self.SB=SB
        self.dom=dom

        a=4.960     # (Angstrom)
        b=4.960
        c=13.599

        self.adir = a*np.array([1,0,0])
        self.bdir = b*np.array([-1/2,np.sqrt(3)/2,0])
        self.cdir = c*np.array([0,0,1])

        self.arec=1/(a)*np.array([1,1/np.sqrt(3),0])*(2*np.pi)
        self.brec=2/(b*np.sqrt(3))*np.array([0,1,0])*(2*np.pi)
        self.crec=1/(c)*np.array([0,0,1])*(2*np.pi)

        # structure from the Bilbao server
        # fractional coordinates and symmetry labels for magnetic ions
        cr_scattlen = 3.635     # (fm)
        o_scattlen  = 5.803
        structure_cr2o3 = [
            ["CrB",   0.000000,   0.000000,   0.152400, cr_scattlen],
            ["CrA",   0.000000,   0.000000,   0.347600, cr_scattlen],
            ["CrB",   0.000000,   0.000000,   0.652400, cr_scattlen],
            ["CrA",   0.000000,   0.000000,   0.847600, cr_scattlen],

            ["CrA",   0.333333,   0.666667,   0.014267, cr_scattlen],
            ["CrB",   0.333333,   0.666667,   0.319067, cr_scattlen],
            ["CrA",   0.333333,   0.666667,   0.514267, cr_scattlen],
            ["CrB",   0.333333,   0.666667,   0.819067, cr_scattlen],

            ["CrA",   0.666667,   0.333333,   0.180933, cr_scattlen],
            ["CrB",   0.666667,   0.333333,   0.485733, cr_scattlen],
            ["CrA",   0.666667,   0.333333,   0.680933, cr_scattlen],
            ["CrB",   0.666667,   0.333333,   0.985733, cr_scattlen],

            ["O",    0.305600,   0.000000,   0.250000, o_scattlen ],
            ["O",    0.694400,   0.694400,   0.250000, o_scattlen ],
            ["O",    0.000000,   0.305600,   0.250000, o_scattlen ],
            ["O",    0.638933,   0.666667,   0.916667, o_scattlen ],
            ["O",    0.972267,   0.333333,   0.583333, o_scattlen ],
            ["O",    0.027733,   0.361067,   0.916667, o_scattlen ],
            ["O",    0.333333,   0.972267,   0.916667, o_scattlen ],
            ["O",    0.361067,   0.027733,   0.583333, o_scattlen ],
            ["O",    0.666667,   0.638933,   0.583333, o_scattlen ],
            ["O",    0.694400,   0.000000,   0.750000, o_scattlen ],
            ["O",    0.000000,   0.694400,   0.750000, o_scattlen ],
            ["O",    0.305600,   0.305600,   0.750000, o_scattlen ],
            ["O",    0.027733,   0.666667,   0.416667, o_scattlen ],
            ["O",    0.333333,   0.361067,   0.416667, o_scattlen ],
            ["O",    0.638933,   0.972267,   0.416667, o_scattlen ],
            ["O",    0.361067,   0.333333,   0.083333, o_scattlen ],
            ["O",    0.666667,   0.027733,   0.083333, o_scattlen ],
            ["O",    0.972267,   0.638933,   0.083333, o_scattlen ],
        ]

        if dom=="in":
            spinA = np.array([0,0,-1])*self.SA
            spinB = np.array([0,0,+1])*self.SB
        elif dom=="out":
            spinA = np.array([0,0,+1])*self.SA
            spinB = np.array([0,0,-1])*self.SB

        for atom in structure_cr2o3:
            if atom[0]=="O":
                atom.append(0.0)
                atom.append(0.0)
                atom.append(0.0)
            if atom[0]=="CrA":
                atom.append(spinA[0])
                atom.append(spinA[1])
                atom.append(spinA[2])
            elif atom[0]=="CrB":
                atom.append(spinB[0])
                atom.append(spinB[1])
                atom.append(spinB[2])

        crA_atoms = []
        crB_atoms = []
        o_atoms = []
        for atom in structure_cr2o3:
            if atom[0]=="O":
                o_atoms.append(atom)
            elif atom[0]=="CrA":
                crA_atoms.append(atom)
            elif atom[0]=="CrB":
                crB_atoms.append(atom)

        self.crA_atoms = np.array(crA_atoms)
        self.crB_atoms = np.array(crB_atoms)
        self.o_atoms = np.array(o_atoms)
        return None

    def calc_NQ(self, Q):
        NQ = 0.0
        h,k,l=Q
        Qcart = self.arec*h+self.brec*k+self.crec*l
        cr2o3_structure = np.vstack([self.crA_atoms, self.crB_atoms, self.o_atoms])
        for atom in cr2o3_structure:
            xj,yj,zj = atom[1:4].astype(float)
            rjcart = self.adir*xj+self.bdir*yj+self.cdir*zj
            b = atom[4].astype(float)
            NQ += b * np.exp(1j*np.dot(Qcart,rjcart))
        return np.round(NQ, decimals=SDIG)

    def calc_MQ(self, Q):
        MQ = np.array([0,0,0], dtype=complex)
        h,k,l=Q
        Qcart = self.arec*h+self.brec*k+self.crec*l
        cr2o3_structure = np.vstack([self.crA_atoms, self.crB_atoms])
        for atom in cr2o3_structure:
            xj,yj,zj = atom[1:4].astype(float)
            rjcart = self.adir*xj+self.bdir*yj+self.cdir*zj
            Sj = atom[5:8].astype(float)
            MQ += Sj * np.exp(1j*np.dot(Qcart,rjcart)) * dipolar_formfactor(Qcart/np.sqrt(np.dot(Qcart,Qcart)))[0]
        MQ *= -self.gyro*self.r0
        return np.round(MQ, decimals=SDIG)

    def calc_MperpQ(self, Q):
        h,k,l=Q
        Qcart=self.arec*h+self.brec*k+self.crec*l
        Qhat=Qcart/np.sqrt(np.dot(Qcart,Qcart))
        MaQ,MbQ,McQ=self.calc_MQ(Q)
        MQcart=self.arec*MaQ+self.brec*MbQ+self.crec*McQ
        MperpQ=np.cross(Qhat,np.cross(MQcart,Qhat))
        return np.round(MperpQ, decimals=SDIG)

    def calc_polaxes(self, Q):
        h,k,l=Q
        Qcart=self.arec*h+self.brec*k+self.crec*l
        x=Qcart
        z=self.bdir
        y=np.cross(z,x)
        z=np.cross(x,y)
        x=x/np.sqrt(np.dot(x,x))
        y=y/np.sqrt(np.dot(y,y))
        z=z/np.sqrt(np.dot(z,z))
        return np.vstack([ x,y,z ])

    def calc_Pf(self, Q, Pi):
        Pi = np.array(Pi)
        NQ = self.calc_NQ(Q)
        MPQ = self.calc_MperpQ(Q)
        POLAXES = self.calc_polaxes(Q)
        MperpQ=POLAXES @ MPQ
        IPi =   np.conj(NQ)*NQ \
                +np.dot(np.conj(MperpQ), MperpQ)\
                +np.dot(Pi, np.conj(NQ)*MperpQ+np.conj(MperpQ)*NQ)\
                +1j*np.dot(Pi, np.cross(np.conj(MperpQ), MperpQ))
        Pf  =   Pi*np.conj(NQ)*NQ+np.conj(NQ)*MperpQ+np.conj(MperpQ)*NQ\
                +np.conj(MperpQ)*np.dot(MperpQ,Pi)+np.dot(np.conj(MperpQ),Pi)*MperpQ\
                -Pi*np.dot(np.conj(MperpQ),MperpQ)-1j*np.cross(np.conj(MperpQ),MperpQ)\
                +1j*np.conj(NQ)*np.cross(MperpQ,Pi)+1j*np.cross(Pi,np.conj(MperpQ))*NQ
        Pf = Pf/IPi
        return np.round(np.real(Pf), decimals=SDIG)

    def calc_polarimetry_matrix(self, Q):
        polmat = np.zeros((3,3))
        polmat[0,:] = self.calc_Pf(Q, np.array([1,0,0])*self.P0)
        polmat[1,:] = self.calc_Pf(Q, np.array([0,1,0])*self.P0)
        polmat[2,:] = self.calc_Pf(Q, np.array([0,0,1])*self.P0)
        return polmat

if __name__ == "__main__":

    print("\n\tCr2O3 SNP calculation")

    SA=1.3
    SB=1.3
    P0=1.0
    Q = [1,0,-2]
    cr2o3 = cr2o3_polarimetry(SA=SA,SB=SB,dom="out")

    print(f"\tSpin domain: {cr2o3.dom}")
    print(f"\tSpin magnitude of site A: {SA}")
    print(f"\tSpin magnitude of site B: {SB}")
    print(f"\tQ: {Q} r.l.u.")
    print(f"\tN(Q): {cr2o3.calc_NQ(Q):1.3f} fm")
    print(f"\tM(Q): {cr2o3.calc_MQ(Q)} fm")
    print(f"\tMperp(Q): {cr2o3.calc_MperpQ(Q)} fm")
    POLAXES=cr2o3.calc_polaxes(Q)
    MperpQ=np.round(POLAXES @ cr2o3.calc_MperpQ(Q),decimals=SDIG)
    print(f"\tMperp(Q) polarimetry frame: {MperpQ} fm")
    print()
    Pi=np.array([1,0,0])*P0
    print(f"\tPf(Q), Pi={Pi}: {cr2o3.calc_Pf(Q,Pi)}")
    Pi=np.array([0,1,0])*P0
    print(f"\tPf(Q), Pi={Pi}: {cr2o3.calc_Pf(Q,Pi)}")
    Pi=np.array([0,0,1])*P0
    print(f"\tPf(Q), Pi={Pi}: {cr2o3.calc_Pf(Q,Pi)}")
    print()