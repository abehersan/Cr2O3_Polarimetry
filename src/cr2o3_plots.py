import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cr2o3_blumemaleev import cr2o3_polarimetry
mpl.rcParams["figure.dpi"]=100


def plot_magnetic_structure(cr2o3_polarimetry):
    fig = plt.figure(figsize=(11.7,8.3))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the unit cell by connecting the corners
    a = cr2o3_polarimetry.adir
    b = cr2o3_polarimetry.bdir
    c = cr2o3_polarimetry.cdir
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
    edges = [
        (0, 1), (0, 2), (0, 3),  # edges from origin
        (1, 4), (1, 5),          # edges from a
        (2, 4), (2, 6),          # edges from b
        (3, 5), (3, 6),          # edges from c
        (4, 7), (5, 7), (6, 7)   # edges to the far corner
    ]

    # Draw each edge of the unit cell
    for start, end in edges:
        ax.plot(
            [corners[start][0], corners[end][0]],
            [corners[start][1], corners[end][1]],
            [corners[start][2], corners[end][2]],
            'k--'
        )

    # Plot the basis vectors as arrows
    origin = np.array([0, 0, 0])
    ax.quiver(*origin, *a, color='r', length=1, arrow_length_ratio=0.1)
    ax.quiver(*origin, *b, color='g', length=1, arrow_length_ratio=0.1)
    ax.quiver(*origin, *c, color='b', length=1, arrow_length_ratio=0.1)
    ax.text(*a, "a", color="red", fontsize=12)
    ax.text(*b, "b", color="green", fontsize=12)
    ax.text(*c, "c", color="blue", fontsize=12)

    # Plot the atomic positions
    fractional_positions_crA = cr2o3_polarimetry.crA_atoms[:,1:4].astype(float)
    fractional_positions_crB = cr2o3_polarimetry.crB_atoms[:,1:4].astype(float)
    fractional_positions_o   = cr2o3_polarimetry.o_atoms[:,1:4].astype(float)
    cartesian_positions_crA = np.array([pos[0] * a + pos[1] * b + pos[2] * c for pos in fractional_positions_crA])
    cartesian_positions_crB = np.array([pos[0] * a + pos[1] * b + pos[2] * c for pos in fractional_positions_crB])
    cartesian_positions_o   = np.array([pos[0] * a + pos[1] * b + pos[2] * c for pos in fractional_positions_o])

    ax.scatter(cartesian_positions_crA[:, 0], cartesian_positions_crA[:, 1], cartesian_positions_crA[:, 2], color='red', s=100)
    ax.scatter(cartesian_positions_crB[:, 0], cartesian_positions_crB[:, 1], cartesian_positions_crB[:, 2], color='blue', s=100)
    ax.scatter(cartesian_positions_o[:, 0],   cartesian_positions_o[:, 1],   cartesian_positions_o[:, 2],   color='black', s=10)

    # Plot the spins as arrows originating from each atomic position
    spins_crA = cr2o3_polarimetry.crA_atoms[:,5:8].astype(float)
    for pos, spin in zip(cartesian_positions_crA, spins_crA):
        ax.quiver(
            pos[0], pos[1], pos[2],     # Start point (the atom position)
            spin[0], spin[1], spin[2],  # Vector spin components
            color='red', length=0.5, arrow_length_ratio=0.3
        )
    spins_crB = cr2o3_polarimetry.crB_atoms[:,5:8].astype(float)
    for pos, spin in zip(cartesian_positions_crB, spins_crB):
        ax.quiver(
            pos[0], pos[1], pos[2],         # Start point (the atom position)
            spin[0], spin[1], spin[2],      # Vector spin components
            color='blue', length=0.5, arrow_length_ratio=0.3
        )

    # Highlight the a-c plane as a semi-transparent surface
    plane_points = np.array([
        [0, 0, 0],     # Origin
        a,             # Along a-axis
        a + c,         # Along a and c
        c              # Along c-axis
    ])
    plane = Poly3DCollection([plane_points], color='steelblue', alpha=0.3, edgecolor='k')
    ax.add_collection3d(plane)

    # Set plot labels and limits
    ax.set_axis_off()
    ax.set_xlim([-7,7])
    ax.set_ylim([-7,7])
    ax.set_zlim([0,14])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    return fig

def plot_polarimetry_axes(cr2o3_polarimetry, Q, Pi):
    POLAXES = cr2o3_polarimetry.calc_polaxes(Q)
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
    edges = [
        (0, 1), (0, 2), (0, 3),  # edges from origin
        (1, 4), (1, 5),          # edges from a
        (2, 4), (2, 6),          # edges from b
        (3, 5), (3, 6),          # edges from c
        (4, 7), (5, 7), (6, 7)   # edges to the far corner
    ]

    # Draw each edge of the unit cell
    for start, end in edges:
        ax.plot(
            [corners[start][0], corners[end][0]],
            [corners[start][1], corners[end][1]],
            [corners[start][2], corners[end][2]],
            'k--'
        )

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
    ########################################################################


    # define polarimetry axes x||Q, z vertical and y = z cross x and plot them
    ########################################################################
    POL_AX_LEN = 0.5 # scale of the polarimetry axes for the plot only!!!

    # QQ = cr2o3_polarimetry.Mrec @ Q# / (2*np.pi)
    QQ=cr2o3_polarimetry.arec*Q[0]+cr2o3_polarimetry.brec*Q[1]+cr2o3_polarimetry.crec*Q[2]
    ax.quiver(*origin, *QQ, color='purple', length=1.0, arrow_length_ratio=0.1)
    ax.text(*QQ, f"Q={Q}", color="purple", fontsize=12)

    x = POLAXES[0,:]
    ax.quiver(*QQ, *x, color='red', length=POL_AX_LEN, arrow_length_ratio=0.1)
    ax.text(*QQ+x*POL_AX_LEN, f"x", color="red", fontsize=12)
    y = POLAXES[1,:]
    ax.quiver(*QQ, *y, color='green', length=POL_AX_LEN, arrow_length_ratio=0.1)
    ax.text(*QQ+y*POL_AX_LEN, f"y", color="green", fontsize=12)
    z = POLAXES[2,:]
    ax.quiver(*QQ, *z, color='blue', length=POL_AX_LEN, arrow_length_ratio=0.1)
    ax.text(*QQ+z*POL_AX_LEN, f"z", color="blue", fontsize=12)
    ########################################################################


    # Blume-Maleev equations to calculate Pf given Pi
    ########################################################################
    MperpQhat = np.imag(cr2o3_polarimetry.calc_MperpQ(Q))
    MperpQhat = MperpQhat/np.sqrt(np.dot(MperpQhat,MperpQhat))
    ax.quiver(*QQ, *MperpQhat, color='grey', length=1.0, arrow_length_ratio=0.1)
    ax.text(*QQ+MperpQhat, f"Mi", color="grey", fontsize=12)

    Picart = np.linalg.inv(POLAXES) @ Pi
    ax.quiver(*QQ, *Picart, color='grey', length=np.linalg.norm(Picart), arrow_length_ratio=0.1)
    ax.text(*QQ+Picart, f"Pi", color="grey", fontsize=12)

    Pf = cr2o3_polarimetry.calc_Pf(Q, Pi)
    Pfcart = np.linalg.inv(POLAXES) @ Pf
    ax.quiver(*QQ, *Pfcart, color='grey', length=np.linalg.norm(Pfcart), arrow_length_ratio=0.1)
    ax.text(*QQ+Pfcart, f"Pf", color="grey", fontsize=12)
    ########################################################################

    title = f"Cr2O3, dom: {cr2o3_polarimetry.dom}, \nQ: {Q}, Pi: {Pi}, Pf: {Pf}"
    fig.suptitle(title)

    ax.set_axis_off()
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_zlim([-3,3])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    return fig

if __name__ == "__main__":

    print("\n\tCr2O3 SNP calculation")

    SA=1.3
    SB=1.3
    P0=1.0
    Q = [1,0,-2]
    cr2o3 = cr2o3_polarimetry(SA=SA,SB=SB,dom="out")

    Pi=np.array([1,0,0])*P0
    fig_structure = plot_magnetic_structure(cr2o3)
    fig_polarimetry = plot_polarimetry_axes(cr2o3, Q, Pi)

    print(f"\tSpin domain: {cr2o3.dom}")
    Pi=np.array([1,0,0])*P0
    print(f"\tPf(Q), Pi={Pi}: {cr2o3.calc_Pf(Q,Pi)}")
    Pi=np.array([0,1,0])*P0
    print(f"\tPf(Q), Pi={Pi}: {cr2o3.calc_Pf(Q,Pi)}")
    Pi=np.array([0,0,1])*P0
    print(f"\tPf(Q), Pi={Pi}: {cr2o3.calc_Pf(Q,Pi)}")
    print()
    plt.show()