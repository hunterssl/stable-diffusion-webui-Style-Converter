import launch
import os,shutil
from pathlib import Path
import socket

# TODO: add pip dependency if need extra module only on extension
if not launch.is_installed("tqdm"):
    launch.run_pip("install tqdm", "requirements for Style-Converter")
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

checkpoints_dir = root_dir+"/models/Stable-diffusion"
checkpoints = ["toonyou_beta3.safetensors", 
            "CounterfeitV30_30.safetensors", 
            "pikasAnimatedMix_10Pruned.safetensors", 
            "aniflatmixAnimeFlatColorStyle_v20.safetensors", 
            "majicmixLux_v10.safetensors"]

loras_dir = root_dir+"/models/Lora"
loras = ["Retromanga_V1.safetensors"]

embeddings_dir = root_dir+"/embeddings"
embeddings = ["https://huggingface.co/embed/EasyNegative/blob/main/EasyNegative.safetensors",
              ]

def download_progress(source_file, destination_file, chunk_size=4096):
    """
    Function to download a file from a URL and display a progress bar.
    """
    with open(destination_file, "wb") as output_file:
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=source_file.split("/")[-1]) as pbar:
            for chunk in source_file.iter_content(chunk_size=chunk_size):
                output_file.write(chunk)
                pbar.update(len(chunk))
        pbar.close()
        print(f"[Style-Converter] {source_file} Downloaded.")

# for iter in [(checkpoints_dir,checkpoints), (loras_dir,loras), (embeddings_dir,embeddings)]:
#     if os.path.exists(iter[0]):
#         path = Path(iter[0])
#         for i in iter[1]:
#             files = [file for file in path.rglob(i.split("/")[-1])]
#             if len(files):
#                 pass
#             else:
#                 print(f"[Style-Converter] {i} not Found,try to download.")
#                 download_progress(i, files[0]+"/"+i)
#     else:
#         print(f"[Style-Converter] Error: Dir {iter[0]} not Found.")