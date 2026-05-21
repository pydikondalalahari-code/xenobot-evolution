# src/representation.py

import numpy as np #storing voxel arrays
from scipy.ndimage import label #It assigns labels for chunks in genome to see whether they are connected or not
import xml.etree.ElementTree as ET # creating xml files to feed simulator
from pathlib import Path

EMPTY    = 0
PASSIVE  = 1
ACTIVE_P = 2   # phase-offset active  (expands first)
ACTIVE_N = 3   # anti-phase  active   (contracts first)
N_MATERIALS = 4

# values used to compute motion by the simulator from the 2020 paper.
MATERIAL_MAP = {
    PASSIVE:  {'r': 0.01, 'E': 1e6,  'rho': 1e3, 'nu': 0.35,
               'cilia': False, 'phase': 0.0},
    ACTIVE_P: {'r': 0.01, 'E': 5e5,  'rho': 1e3, 'nu': 0.35,
               'cilia': True,  'phase': 0.0},
    ACTIVE_N: {'r': 0.01, 'E': 5e5,  'rho': 1e3, 'nu': 0.35,
               'cilia': True,  'phase': 3.14159},
}


# This function purpose is to create a random robot body.the parameters are default grid size and density(helps in telling empty space/occupied)
def random_genome(grid_size=(8, 8, 8), density=0.4) -> np.ndarray:
    genome = np.random.choice(
        [EMPTY, PASSIVE, ACTIVE_P, ACTIVE_N],#choice list
        size=grid_size,
        p=[1 - density, density / 3, density / 3, density / 3], #controls probabilities
    ).astype(np.int32)
    return genome # entire 3d body with all voxel points



# return genome with the largest connected component of non-empty voxels.
def largest_connected_component(genome: np.ndarray) -> np.ndarray:
    filled = (genome != EMPTY).astype(int) #checks for occupancies and turns into 1.(negelects material type)
    labeled, n_features = label(filled) #label the connected voxel group and label it n_features means number of connected components found.
    if n_features == 0: #robot is empty, so return same genome
        return genome.copy()
    sizes = [(labeled == i).sum() for i in range(1, n_features + 1)] #size of each connected components
    dominant = int(np.argmax(sizes)) + 1 #selects the largest one
    result = genome.copy()
    result[labeled != dominant] = EMPTY # apart from the selected one, all others become empty
    return result

#Return True if all non-empty voxels form a single connected component.
def is_connected(genome: np.ndarray) -> bool:
    
    filled = (genome != EMPTY).astype(int)
    _, n_features = label(filled)
    return n_features <= 1


def count_voxels(genome: np.ndarray) -> dict:
    return {
        'total':    int((genome != EMPTY).sum()),
        'passive':  int((genome == PASSIVE).sum()),
        'active_p': int((genome == ACTIVE_P).sum()),
        'active_n': int((genome == ACTIVE_N).sum()),
    }


# genome_to_vxa() takes the 8×8×8 voxel robot and writes a .vxa XML file that VoxCraft can simulate.
#parameters genome array, path to save .vxa file, simulation time: How many seconds does this test take?

def genome_to_vxa(genome: np.ndarray, output_path: str, sim_time: float = 1.0):
    
    root = ET.Element("VXA", Version="1.1") #creating xml structure <VXA Version="1.1">

    # Simulator
    #in root-->simulator-->integration-->dtfrac value [parent,child]
    #At EACH timestep:simulator calculates:new forces, acceleration,velocity,position
    #So motion is produced by repeated timestep updates
    #physics simulation is:many tiny updates,chained together, that creates apparent continuous movement.
    
    sim = ET.SubElement(root, "Simulator")
    integ = ET.SubElement(sim, "Integration")
    ET.SubElement(integ, "Integrator").text = "0"
    ET.SubElement(integ, "DtFrac").text = "0.9"

    stop = ET.SubElement(sim, "StopCondition")
    ET.SubElement(stop, "StopConditionType").text = "2"      # 2 = time-based
    ET.SubElement(stop, "StopConditionValue").text = str(sim_time)

    rec = ET.SubElement(sim, "RecordHistory")
    ET.SubElement(rec, "RecordStepSize").text = "100"
    ET.SubElement(rec, "RecordVoxel").text = "1"
    ET.SubElement(rec, "RecordLink").text = "0"
    ET.SubElement(rec, "RecordFixedVoxels").text = "0"

    # environment
    
    env = ET.SubElement(root, "Environment")
    grav = ET.SubElement(env, "Gravity")
    ET.SubElement(grav, "GravEnabled").text = "1"
    ET.SubElement(grav, "GravAcc").text = "-9.81"
    ET.SubElement(grav, "FloorEnabled").text = "1"

    thermal = ET.SubElement(env, "Thermal")
    ET.SubElement(thermal, "TempEnabled").text = "1"
    ET.SubElement(thermal, "TempAmp").text = "39"
    ET.SubElement(thermal, "TempPeriod").text = "0.025"
    ET.SubElement(thermal, "VaryTempEnabled").text = "0"
    ET.SubElement(thermal, "TempBase").text = "25"

    vxc = ET.SubElement(root, "VXC", Version="0.94")
    palette = ET.SubElement(vxc, "Palette")

    #writes material id and remaining physics parameter into file
    for mat_id, props in MATERIAL_MAP.items():
        mat = ET.SubElement(palette, "Material", ID=str(mat_id))
        ET.SubElement(mat, "MatModel").text = "0"
        mech = ET.SubElement(mat, "Mechanical")
        ET.SubElement(mech, "MatDensity").text = str(props['rho'])
        ET.SubElement(mech, "ElasticMod").text = str(props['E'])
        ET.SubElement(mech, "PoissonsRatio").text = str(props['nu'])
        ET.SubElement(mech, "CTE").text = str(props['r'])
        #if it is active then add movement timing.
        if props['cilia']:
            ET.SubElement(mat, "Cilia").text = "1"
            ET.SubElement(mat, "PhaseOffset").text = str(props['phase'])


    structure = ET.SubElement(vxc, "Structure",
                              replace="replace", Compression="ASCII_READABLE")
    X, Y, Z = genome.shape
    ET.SubElement(structure, "X_Voxels").text = str(X)
    ET.SubElement(structure, "Y_Voxels").text = str(Y)
    ET.SubElement(structure, "Z_Voxels").text = str(Z)

    data_el = ET.SubElement(structure, "Data")
    for z in range(Z):
        layer = ET.SubElement(data_el, "Layer")
        row = "".join(str(genome[x, y, z])
                      for y in range(Y) for x in range(X))
        layer.text = row   
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="unicode", xml_declaration=True)
