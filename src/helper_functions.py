import os 
import torch
import itertools
import pandas as pd


def check_protein_vec(protein_dict):
    """
    A function to check whether the protein vectors are properly embedded by esm
    """
    rm_list = []
    for k, v in protein_dict.items():
        if torch.isnan(v).any():
            rm_list.append(k)
    
    for i in rm_list:
        del protein_dict[i]
    print(f"Remove {len(rm_list)} proteins")
    print(f"==> {rm_list}")
    return protein_dict


def load_get_target_protein(protein_path): 
    esm2_embedded =  torch.load(protein_path, map_location=torch.device("cpu"))
    esm2_embedded = check_protein_vec(esm2_embedded)
    return esm2_embedded

def rm_empty_dict(d):
    l_rm = []
    for k, v in d.items():
        if len(v) == 0:
            l_rm.append(k)
    for i in l_rm:
        del d[i]
    return d



def load_chemical_vectors(base_path):
    """
    base_path: The path to the directory where chemical vectors are stored
    It is necessary to have the following vector files stored in this directory.
    - morganfingerprint.pt
    - chemical_checker.pt
    - chemberta.pt
    """
    chem_vec = {}

    vec_morganfinger = torch.load(os.path.join(base_path, "morgan_fingerprint.pt"), torch.device("cpu"))
    chem_vec["mf"] = rm_empty_dict(vec_morganfinger)
    print(f"# moragn: {len(vec_morganfinger.keys())}")


    vec_cc = torch.load(os.path.join(base_path, "chemical_checker.pt"), torch.device("cpu"))
    vec_cc = rm_empty_dict(vec_cc)
    print(f"# cc: {len(vec_cc.keys())}")
    chem_vec["cc"] = vec_cc


    vec_chembert = torch.load(os.path.join(base_path, "chemberta-2.pt"), torch.device("cpu"))
    vec_chembert = rm_empty_dict(vec_chembert)
    print(f"# bert: {len(vec_chembert.keys())}")
    chem_vec["bert"] = vec_chembert

    return chem_vec


def logging_score(model_output, target, output_path):
    protein_names = list(target.keys())
    output_list = []
    for protein_name in protein_names:

        scores_tmp = [{k: v} for k, v in model_output.items() if protein_name in k]
        id_pair = []
        score = []
        for i in scores_tmp:
            id_pair.append(list(i.keys())[0][1])
            score.append(list(i.values()))
        score = list(itertools.chain.from_iterable(score))

        df = pd.DataFrame(zip(id_pair, score), columns=["Compound_ID", "CPI_score"])
        df = df.sort_values("CPI_score", ascending=False)

        file_name = protein_name + "_screening_score.csv"
        cwd = os.getcwd()
        save_path = os.path.join(output_path, file_name)

        os.makedirs(output_path, exist_ok=True)
        df.to_csv(save_path)
        output_list.append(save_path)
    return output_list


