import endf
from NeSST.constants import *
from NeSST.cross_sections import NeSST_SDX, NeSST_DDX
import numpy as np
import json
from dataclasses import dataclass

# Some defs
# See https://t2.lanl.gov/nis/endf/mts.html

MF_XSECTION = 3
MF_ENERGYANGLEDISTS = 4
MF_PRODENERGYANGLEDISTS = 6

MT_TOTXSECTION = 1
MT_ELXSECTION = 2
MT_N2NXSECTION = 16
MT_INELXSECTION_START = 51

@dataclass
class ENDFManifest:
    total : bool
    elastic : bool
    inelastic : bool
    n2n : bool

def retrieve_ENDF_data(json_file):
    mainfest = convert_json_to_manifest(data_dir+json_file)
    ENDF_data = retrieve_ENDF_data_from_manifest(mainfest)
    return ENDF_data

def convert_json_to_manifest(json_file):
    with open(json_file, 'r') as js:
        json_dict = json.load(js)
    manifest_dict = {}
    for filename, file_manifest in json_dict.items():
        manifest_dict[filename] = ENDFManifest(total=file_manifest['total'],
                                               elastic=file_manifest['elastic'],
                                               inelastic=file_manifest['inelastic'],
                                               n2n=file_manifest['n2n'])
    return manifest_dict

def retrieve_ENDF_data_from_manifest(manifest):
    return_dict = {'interactions' : ENDFManifest(total=False,
                                                 elastic=False,
                                                 inelastic=False,
                                                 n2n=False)}

    for filename, file_manifest in manifest.items():

        mat = endf.Material(ENDF_dir+filename)

        if(file_manifest.total == True):
            if(return_dict['interactions'].total):
                print(f'{filename}: Warning, loading total cross section data from multiple files...')
            else:
                return_dict['interactions'].total = True

            total_dataset = mat.section_data[MF_XSECTION,MT_TOTXSECTION]
            return_dict['A'] = total_dataset['AWR']
            total_table = total_dataset['sigma']
            total_E, total_sigma = total_table.x, total_table.y
            return_dict['total_xsec'] = {'E' : total_E, 'sig' : total_sigma}

        if(file_manifest.elastic == True):
            if(return_dict['interactions'].elastic):
                print(f'{filename}: Warning, loading elastic cross section data from multiple files...')
            else:
                return_dict['interactions'].elastic = True

            elastic_table = mat.section_data[MF_XSECTION,MT_ELXSECTION]['sigma']
            elastic_E, elastic_sigma = elastic_table.x, elastic_table.y
            return_dict['elastic_xsec'] = {'E' : elastic_E, 'sig' : elastic_sigma}

            LTT = mat.section_data[MF_ENERGYANGLEDISTS,MT_ELXSECTION]['LTT']

            if(LTT == 0): # Isotropic
                return_dict['elastic_dxsec'] = {'legendre' : True, 'E' : np.array([elastic_E[0],elastic_E[-1]]), 'a_l' : np.zeros((2,1)), 'N_l' : 1}

            elif(LTT == 1 or LTT == 3): # Legendre
                if(LTT == 3):
                    print(f'{filename}: Warning, LTT ({LTT}) for elastic scattering, using Legendre only')
                elastic_table = mat.section_data[MF_ENERGYANGLEDISTS,MT_ELXSECTION]['legendre']
                elastic_E, elastic_al = elastic_table['E'], convert_to_NeSST_legendre_format(elastic_table['a_l'])
                return_dict['elastic_dxsec'] = {'legendre' : True, 'E' : elastic_E, 'a_l' : elastic_al, 'N_l' : elastic_al.shape[1]}

            elif(LTT == 2): # Tabulated
                elastic_table = mat.section_data[MF_ENERGYANGLEDISTS,MT_ELXSECTION]['tabulated']
                elastic_E = elastic_table['E']
                NE = elastic_E.shape[0]
                elastic_Ein, elastic_mu, elastic_f = [] , [] , []
                for iE in range(NE):
                    elastic_mu.append(elastic_table['mu'][iE].x)
                    elastic_f.append(elastic_table['mu'][iE].y)
                    Nmu = elastic_table['mu'][iE].x.shape[0]
                    elastic_Ein.append(np.repeat(elastic_E[iE],Nmu))

                points = np.column_stack((np.concatenate(elastic_Ein),np.concatenate(elastic_mu)))
                values = np.concatenate(elastic_f)
                SDX = NeSST_SDX(Ein=elastic_E,points=points,values=values)
                return_dict['elastic_dxsec'] = {'legendre' : False, 'SDX' : SDX}

            else:
                print(f'{filename}: Warning, LTT ({LTT}) for elastic scattering not recognised...')

        if(file_manifest.inelastic == True):
            if(return_dict['interactions'].inelastic):
                print(f'{filename}: Warning, loading inelastic cross section data from multiple files...')
            else:
                return_dict['interactions'].inelastic = True

            i_inelastic = 1
            while True:
                MT = MT_INELXSECTION_START+(i_inelastic-1)
                try:
                    inelastic_table = mat.section_data[MF_XSECTION,MT]['sigma']
                    inelastic_E, inelastic_sigma = inelastic_table.x, inelastic_table.y
                    return_dict[f'inelastic_xsec_n{i_inelastic}'] = {'E' : inelastic_E, 'sig' : inelastic_sigma}
                    inelastic_Q = mat.section_data[MF_XSECTION,MT]['QI']
                except:
                    break

                LTT = mat.section_data[MF_ENERGYANGLEDISTS,MT]['LTT']

                if(LTT == 0): # Isotropic
                    return_dict[f'inelastic_dxsec_n{i_inelastic}'] = {'legendre' : True, 'E' : np.array([inelastic_E[0],inelastic_E[-1]]), 'a_l' : np.zeros((2,1)), 'N_l' : 1, 'Q' : inelastic_Q}

                elif(LTT == 1 or LTT == 3): # Legendre
                    if(LTT == 3):
                        print(f'{filename}: Warning, LTT ({LTT}) for inelastic scattering, level {i_inelastic}, using Legendre only')
                    inelastic_table = mat.section_data[MF_ENERGYANGLEDISTS,MT]['legendre']
                    inelastic_E, inelastic_al = inelastic_table['E'], convert_to_NeSST_legendre_format(inelastic_table['a_l'])
                    return_dict[f'inelastic_dxsec_n{i_inelastic}'] = {'legendre' : True, 'E' : inelastic_E, 'a_l' : inelastic_al, 'N_l' : inelastic_al.shape[1], 'Q' : inelastic_Q}

                elif(LTT == 2): # Tabulated
                    inelastic_table = mat.section_data[MF_ENERGYANGLEDISTS,MT]['tabulated']
                    inelastic_E = inelastic_table['E']
                    NE = inelastic_E.shape[0]
                    inelastic_mu, Nmu, inelastic_f = [] , [] , []
                    for iE in range(NE):
                        inelastic_mu.append(inelastic_table['mu'][iE].x)
                        inelastic_f.append(inelastic_table['mu'][iE].y)
                        Nmu.append(inelastic_table['mu'][iE].x.shape[0])
                    SDX = NeSST_SDX(NEin=NE,Ein=inelastic_E,Ncos=Nmu,cos=inelastic_mu,f=inelastic_f)
                    return_dict[f'inelastic_dxsec_n{i_inelastic}'] = {'legendre' : False, 'SDX' : SDX, 'Q' : inelastic_Q}
                else:
                    print(f'{filename}: Warning, LTT ({LTT}) for inelastic scattering, level {i_inelastic}, not recognised...')

                i_inelastic += 1

            return_dict['n_inelastic'] = i_inelastic-1

        if(file_manifest.n2n == True):
            if(return_dict['interactions'].n2n):
                print(f'{filename}: Warning, loading n,2n cross section data from multiple files...')
            else:
                return_dict['interactions'].n2n = True

            n2n_table = mat.section_data[MF_XSECTION,MT_N2NXSECTION]['sigma']
            n2n_E, n2n_sigma = n2n_table.x, n2n_table.y
            return_dict['n2n_xsec'] = {'E' : n2n_E, 'sig' : n2n_sigma}

            n2n_Q = mat.section_data[MF_XSECTION,MT_N2NXSECTION]['QI']
            n2n_table = mat.section_data[MF_PRODENERGYANGLEDISTS,MT_N2NXSECTION]
            n2n_products = n2n_table['products']
            for n2n_product in n2n_products:
                if(n2n_product['AWP'] == 1.0): # is neutron?
                    if(n2n_product['LAW'] == 6):
                        return_dict['n2n_dxsec'] = {'LAW' : 6, 
                                                    'A_i' : n2n_product['AWP'],
                                                    'A_e' : n2n_product['AWP'],
                                                    'A_t' : n2n_table['AWR'],
                                                    'A_p' : n2n_product['AWP'],
                                                    'A_tot' : n2n_product['distribution']['APSX'],
                                                    'Q_react' : n2n_Q}
                    elif(n2n_product['LAW'] == 7):
                        DDX = convert_to_NeSST_LAW7_format(n2n_product['distribution'])
                        return_dict['n2n_dxsec'] = {'LAW' : 7, 
                                                    'DDX' : DDX,
                                                    'Q_react' : n2n_Q}
                    else:
                        print(f"{filename}: n2n data LAW = {n2n_product['LAW']}, NeSST cannot use...")

    return return_dict

def retrieve_total_cross_section_from_ENDF_file(filename):

    mat = endf.Material(ENDF_dir+filename)

    total_dataset = mat.section_data[MF_XSECTION,MT_TOTXSECTION]
    total_table = total_dataset['sigma']
    total_E, total_sigma = total_table.x, total_table.y

    return total_E, total_sigma 

def convert_to_NeSST_legendre_format(ENDF_al):
    max_Nl = max([len(a) for a in ENDF_al])
    uniform_len_al = [np.pad(a,(0,max_Nl-len(a))) for a in ENDF_al]
    return np.array(uniform_len_al)

def convert_to_NeSST_LAW7_format(ENDF_dist):
    NE = ENDF_dist['NE']
    Ein = []
    Ncos = []
    cos = []
    NEout = {}
    Eout = {}
    f = {}
    Emax = {}
    for iE in range(NE):
        dist_Ein = ENDF_dist['distribution'][iE]
        Ein.append(dist_Ein['E'])
        NMU = dist_Ein['NMU']
        Ncos.append(NMU)
        cos_arr = np.zeros(NMU)
        for iM in range(NMU):
            dist_mu = dist_Ein['mu'][iM]
            cos_arr[iM] = dist_mu['mu']
            NEout[(iE,iM)] = dist_mu['f'].x.shape[0]
            Eout[(iE,iM)] = dist_mu['f'].x
            f[(iE,iM)] = dist_mu['f'].y
            Emax[(iE,iM)] = np.amax(dist_mu['f'].x)
        cos.append(cos_arr)

    DDX = NeSST_DDX(
        NEin = NE,
        Ein = Ein,
        Ncos = Ncos,
        cos = cos,
        NEout = NEout,
        Eout = Eout,
        f = f,
        Emax = Emax
    )

    return DDX