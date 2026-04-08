import argparse
import os 

from evaluation.models.base_model import convert_fsdp_checkpoints_to_hfmodels
from transformers import AutoTokenizer, AutoConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('local_dir', type=str, help='Path to the local directory containing the FSDP checkpoints.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory where the converted HF models will be saved.')

    
    
    args = parser.parse_args()
    huggingface_dir = os.path.join(args.local_dir, "huggingface")
    convert_fsdp_checkpoints_to_hfmodels(args.local_dir, args.output_dir, huggingface_dir)
    tokenizer = AutoTokenizer.from_pretrained(huggingface_dir)
    tokenizer.save_pretrained(args.output_dir)
    