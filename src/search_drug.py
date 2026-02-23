import random

import torch # type: ignore
import numpy as np

from helper_functions import load_get_target_protein, load_chemical_vectors, logging_score
from model import create_dataloader_search_drugs, create_BaCNet, search_drug

def fix_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Drug Screening Inference")
    parser.add_argument('--model', type=str, default='../models/checkpoint.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--protein', type=str, required=True,
                        help='Path to the target protein embedding')
    parser.add_argument('--chemical', type=str, required=True,
                        help='Base path to the chemical vector files')
    parser.add_argument('--output', type=str, default='inference_results.csv',
                        help='Path to save the inference results')
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    fix_seeds()
    
    args = parse_args()
    model_path = args.model
    protein_path = args.protein
    chem_base_path = args.chemical
    output_path = args.output
    
    
    # load target protein (esm_embedded)
    target = load_get_target_protein(protein_path) 
    # load chemical library
    library = load_chemical_vectors(chem_base_path) 

    # prepare inference
    data_loader = create_dataloader_search_drugs(target=target,
                                                 library=library
                                                 )

    print("len(dataset) =", len(data_loader.dataset))
    print("batch_size   =", data_loader.batch_size)
    print("drop_last    =", data_loader.drop_last)
    print("len(dataloader) =", len(data_loader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #construct model
    model = create_BaCNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device=device)

    print(f"checkpoint: {model_path}")
    print(f"Target protein: {protein_path}")
    print(f"Chemical library base path: {chem_base_path}")
    print("Start screening")
    # Start time measurement
    import time
    start = time.time()
    
    scores = search_drug(model, device,data_loader)
    
    # end time measurement
    elapsed_time = time.time() - start
    print(f"Elapsed time: {elapsed_time} [sec]")
    print("End screening")


    logging_score(scores, target, output_path)