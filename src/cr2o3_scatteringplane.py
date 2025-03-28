import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
from cr2o3_blumemaleev import cr2o3_polarimetry
mpl.rcParams["figure.dpi"]=100

SDIG = 7

def calc_scattering_plane(cr2o3_polarimetry):
    HMIN, HMAX = -4,4
    KMIN, KMAX = -4,4
    LMIN, LMAX = -8,8
    I_CALC = []
    for L in np.arange(LMIN,LMAX,1,dtype=int):
        for K in np.arange(KMIN,KMAX,1,dtype=int):
            for H in np.arange(HMIN,HMAX,1,dtype=int):
                if [H,K,L] == [0,0,0]:
                    continue
                # Q = H*astar+K*bstar+L*cstar
                NQ = cr2o3_polarimetry.calc_NQ([H,K,L])
                IQ = np.round(np.real(np.conj(NQ)*NQ)/10, decimals=SDIG)
                I_CALC.append([H,K,L,IQ])
    return np.vstack(I_CALC)

def plot_scattering_plane(cr2o3_polarimetry):
    fig = plt.figure(figsize=(11.7,8.3))
    ax = fig.add_subplot(111, projection='3d')

    # draw unit cell in reciprocal space
    ########################################################################
    a = cr2o3_polarimetry.arec
    b = cr2o3_polarimetry.brec
    c = cr2o3_polarimetry.crec
    corners = np.array([
        [0, 0, 0],    # origin
        a,            # a
        b,            # b
        c,            # c
        a + b,        # a + b
        a + c,        # a + c
        b + c,        # b + c
        a + b + c     # a + b + c
    ])

    # Define the edges by pairs of corners to connect
    # edges = [
    #     (0, 1), (0, 2), (0, 3),  # edges from origin
    #     (1, 4), (1, 5),          # edges from a
    #     (2, 4), (2, 6),          # edges from b
    #     (3, 5), (3, 6),          # edges from c
    #     (4, 7), (5, 7), (6, 7)   # edges to the far corner
    # ]

    # Draw each edge of the unit cell
    # for start, end in edges:
    #     ax.plot(
    #         [corners[start][0], corners[end][0]],
    #         [corners[start][1], corners[end][1]],
    #         [corners[start][2], corners[end][2]],
    #         'k--'
    #     )

    # Plot the basis vectors as arrows
    origin = np.array([0, 0, 0])
    ax.quiver(*origin, *a, color='r', length=1.0, arrow_length_ratio=0.1)
    ax.quiver(*origin, *b, color='g', length=1.0, arrow_length_ratio=0.1)
    ax.quiver(*origin, *c, color='b', length=1.0, arrow_length_ratio=0.1)
    ax.text(*a, "a*", color="red", fontsize=12)
    ax.text(*b, "b*", color="green", fontsize=12)
    ax.text(*c, "c*", color="blue", fontsize=12)

    # Highlight the a-c plane as a semi-transparent surface
    plane_points = np.array([
        [0, 0, 0],     # Origin
        a,             # Along a-axis
        a + c,         # Along a and c
        c              # Along c-axis
    ])
    plane = Poly3DCollection([plane_points], color='steelblue', alpha=0.3, edgecolor='k')
    ax.add_collection3d(plane)
    ax.set_axis_off()
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_zlim([-3,3])
    ax.set_box_aspect([1, 1, 1])

    title = f"Cr2O3 nuclear structure factors"#, dom: {cr2o3_polarimetry.dom}"
    print(title)
    fig.suptitle(title)

    # plot nuclear structure factors
    ########################################################################

    origin=np.array([0,0,0])
    I_CAL = calc_scattering_plane(cr2o3_polarimetry)
    SFACT = 1.0
    for IQ in I_CAL:
        H,K,L = IQ[0:3]
        I = IQ[-1]
        color="steelblue"
        if K == 0 and I > 0:
            Qcart = H*a+K*b+L*c
            if [H,K,L]==[-1,0,2] or [H,K,L]==[1,0,-2]:
                color="indianred"
            ax.scatter(Qcart[0],Qcart[1],Qcart[2],s=I*SFACT,c=color,alpha=0.5)
            ax.text(*Qcart, f"{np.array([H,K,L],dtype=int)}",fontsize=10,color=color)
    # print(I_CAL)
    # print()
    ########################################################################

    return fig



if __name__ == "__main__":

    SA=1.5/2
    SB=1.5/2
    cr2o3 = cr2o3_polarimetry(SA=SA,SB=SB,dom="in")
    I_CAL = calc_scattering_plane(cr2o3)
    # print(I_CAL[:,0:3])
    fig_scattering = plot_scattering_plane(cr2o3)
    plt.show()