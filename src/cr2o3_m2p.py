import numpy as np

SDIG = 3
ATOL = 1e-6

class cr2o3_polarimetry():
    def __init__(self, MA, MB, domain="out"):
        self.gyro=1.913     # gyromagnetic ratio of the neutron
        self.r0=2.818       # classical electron radius (fm)
        self.MA=MA
        self.MB=MB
        self.domain=domain

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

        if domain=="in":
            momentA = np.array([0,0,-1])*self.MA
            momentB = np.array([0,0,+1])*self.MB
        elif domain=="out":
            momentA = np.array([0,0,+1])*self.MA
            momentB = np.array([0,0,-1])*self.MB

        for atom in structure_cr2o3:
            if atom[0]=="O":
                atom.append(0.0)
                atom.append(0.0)
                atom.append(0.0)
            if atom[0]=="CrA":
                atom.append(momentA[0])
                atom.append(momentA[1])
                atom.append(momentA[2])
            elif atom[0]=="CrB":
                atom.append(momentB[0])
                atom.append(momentB[1])
                atom.append(momentB[2])

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
            Mj = atom[5:8].astype(float)
            MQ += Mj * np.exp(1j*np.dot(Qcart,rjcart))
        MQ *= self.gyro*self.r0/2
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

    def calc_Pxz(self, Q):
        NQ = self.calc_NQ(Q)
        MPQ = self.calc_MperpQ(Q)
        POLAXES = self.calc_polaxes(Q)
        MperpQ=POLAXES @ MPQ
        MperpQ=np.round(MperpQ,decimals=SDIG)
        Mpx,Mpy,Mpz=MperpQ
        assert Mpx==0.0+0.0j
        Jyz=2*np.imag(np.conj(Mpy)*Mpz)
        Jny=2*np.imag(NQ*np.conj(Mpy))
        Rnz=2*np.real(NQ*np.conj(Mpz))
        Ix=np.conj(NQ)*NQ+np.conj(Mpy)*Mpy+np.conj(Mpz)*Mpz+Jyz
        Pxz=(Rnz-Jny)/Ix
        return np.real(Pxz)

    def calc_polarimetry_matrix(self, Q):
        polmat = np.zeros((3,3))
        polmat[0,:] = self.calc_Pf(Q, np.array([1,0,0])*self.P0)
        polmat[1,:] = self.calc_Pf(Q, np.array([0,1,0])*self.P0)
        polmat[2,:] = self.calc_Pf(Q, np.array([0,0,1])*self.P0)
        return polmat

if __name__=="__main__":

    print("\n\tCr2O3 SNP calculation")

    MA=1.3
    MB=1.3
    P0=1.0
    Q = [1,0,-2]
    cr2o3 = cr2o3_polarimetry(MA=MA,MB=MB,domain="out")

    print(f"\tMagnetic domain: {cr2o3.domain}")
    print(f"\tMoment magnitude of site A: {MA}")
    print(f"\tMoment magnitude of site B: {MB}")
    print(f"\tQ: {Q} r.l.u.")
    print(f"\tN(Q): {cr2o3.calc_NQ(Q):1.3f} fm")
    print(f"\tM(Q): {cr2o3.calc_MQ(Q)} fm")
    print(f"\tMperp(Q): {cr2o3.calc_MperpQ(Q)} fm")
    POLAXES=cr2o3.calc_polaxes(Q)
    MperpQ=np.round(POLAXES @ cr2o3.calc_MperpQ(Q),decimals=SDIG)
    print(f"\tMperp(Q) polarimetry frame: {MperpQ} fm")
    print(f"\tPxz matrix element: {cr2o3.calc_Pxz(Q)}")
    print()