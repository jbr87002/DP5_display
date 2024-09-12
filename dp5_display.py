import pickle
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
import os

def show_results(calc_folder_path):
    data_dic_path = f'{calc_folder_path}/dp5/data_dic.p'
    with open(data_dic_path, 'rb') as f:
        data_dic = pickle.load(f)

    for labels, atom_p, mol_path in zip(data_dic['Clabels'], data_dic['CDP5_atom_probs'], data_dic['mols']):
        structure_name = mol_path.split('.')[0]
        full_mol_path = f'{calc_folder_path}/{mol_path}'
        m = Chem.MolFromMolFile(full_mol_path, removeHs=True)

        m = add_dp5_to_mol(m, atom_p, labels)

        drawer = draw_dp5_mol(m, format='svg')
        img_dir = f'{calc_folder_path}/imgs'
        os.makedirs(img_dir, exist_ok=True)
        with open(f'{img_dir}/{structure_name}.svg', 'w') as f:
            f.write(drawer.data)

def add_dp5_to_mol(mol, dp5_probs, c_labels):
    mol_with_props = Chem.Mol(mol)
    carbons = [int(l[1:])-1 for l in c_labels]
    for score, index in zip(dp5_probs, carbons):
        atom = mol_with_props.GetAtomWithIdx(index)
        atom.SetDoubleProp('DP5', score)
    return mol_with_props

def draw_dp5_mol(mol, format='png'):
    cmap = plt.colormaps['RdYlGn']
 
    highlights = {}
    dopts = rdMolDraw2D.MolDrawOptions()
    dopts.fillHighlights = False
    dopts.continuousHighlight = False
    dopts.useBWAtomPalette()
 
    for atom in mol.GetAtoms():
        if atom.HasProp('DP5'):
            score = atom.GetDoubleProp('DP5')
            atom.SetProp('atomNote', str(f'{score:.2f}'))
            *col, alpha = cmap(score)
            highlights[atom.GetIdx()] = tuple(col)
    rdDepictor.Compute2DCoords(mol)
    d = Draw.MolsToGridImage([mol], highlightAtomLists=[highlights.keys()],
                   highlightAtomColors=[highlights], highlightBondLists=None, subImgSize=(400,400), useSVG=(format=='svg'), returnPNG=(format=='png'), molsPerRow=1, drawOptions=dopts)
    return d