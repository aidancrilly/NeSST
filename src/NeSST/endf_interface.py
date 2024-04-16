import endf
from NeSST.constants import *
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

def convert_to_NeSST_legendre_format(ENDF_al):
    max_Nl = max([len(a) for a in ENDF_al])
    uniform_len_al = [np.pad(a,(0,max_Nl-len(a))) for a in ENDF_al]
    return np.array(uniform_len_al)

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

            try:
                elastic_table = mat.section_data[MF_ENERGYANGLEDISTS,MT_ELXSECTION]['legendre']
                elastic_E, elastic_al = elastic_table['E'], convert_to_NeSST_legendre_format(elastic_table['a_l'])
                return_dict['elastic_dxsec'] = {'E' : elastic_E, 'a_l' : elastic_al, 'N_l' : elastic_al.shape[1]}
            except:
                print(f'{filename}: Legendre data missing for elastic...')

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
                except:
                    break

                try:
                    inelastic_table = mat.section_data[MF_ENERGYANGLEDISTS,MT]['legendre']
                    inelastic_E, inelastic_al = inelastic_table['E'], convert_to_NeSST_legendre_format(inelastic_table['a_l'])
                    return_dict['inelastic_dxsec_n{i_inelastic}'] = {'E' : inelastic_E, 'a_l' : inelastic_al}
                except:
                    print(f'{filename}: Legendre data missing for inelastic (n,n{i_inelastic})...')

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
                    if(n2n_product['LAW'] == 7):
                        DDX = convert_to_NeSST_LAW7_format(n2n_product['distribution'])
                        return_dict['n2n_dxsec'] = {'LAW' : 7, 
                                                    'NE' : n2n_product['distribution']['NE'],
                                                    'DDX' : DDX,
                                                    'Q_react' : n2n_Q}
                    else:
                        print(f"{filename}: n2n data LAW = {n2n_product['LAW']}, NeSST cannot use...")

    return return_dict

retrieve_ENDF_data('H2.json')