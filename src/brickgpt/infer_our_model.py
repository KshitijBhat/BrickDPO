import os
import time

import transformers
from transformers import HfArgumentParser

from brickgpt.models import BrickGPT, BrickGPTConfig
from brickgpt.render_bricks import render_bricks

import os

def prepare_output_paths(model_version: str, filename: str):
    """
    Given a model version (1, 2, or 3) and a filename provided by the user,
    return the txt, ldr, and png output paths within the correct model-specific directory.

    model_version:
        '1' → 2048 model          → outputs_2048/
        '2' → 2500 model          → outputs_2500/
        '3' → 2500 partial model  → outputs_2500_partial/
    """

    # -------------------------------
    # 1. Model version → model path + base output directory
    # -------------------------------

    if model_version == '0':
        model_name = None
        base_parent_dir = "outputs_baseline"

    elif model_version == '1':
        model_name = "kshitij-hf/brick-dpo-base-2048"
        base_parent_dir = "outputs_2048"

    elif model_version == '2':
        raise ValueError("model not trained yet")
        model_name = "kshitij-hf/brick-dpo-base-2500"
        base_parent_dir = "outputs_2500"

    elif model_version == '3':
        model_name = "kshitij-hf/brickgpt_partial"
        base_parent_dir = "outputs_2500_partial"

    else:
        raise ValueError("model_version must be one of: '1', '2', '3'")

    # -------------------------------
    # 2. Build filename (strip extension)
    # -------------------------------

    if not filename or filename.strip() == "":
        filename = "output.png"

    print("ff", filename)
    base_name = filename[:-4] if filename.lower().endswith('.png') else filename
    print("bb", base_name)

    # model suffix
    if model_version == '0':
        suffix = "_baseline"
    elif model_version == '1':
        suffix = "_model2048"
    elif model_version == '2':
        suffix = "_model2500"
    else:
        suffix = "_model2500_partial"

    base_name = base_name + suffix

    # -------------------------------
    # 3. Construct full paths
    # -------------------------------

    os.makedirs(base_parent_dir, exist_ok=True)

    txt_filename = base_parent_dir + "/" + base_name + ".txt"
    ldr_filename = base_parent_dir + "/" + base_name + ".ldr"
    img_filename = base_parent_dir + "/" + base_name + ".png"

    # -------------------------------
    # 4. Return model path + output paths
    # -------------------------------

    model_name_short = base_parent_dir[8:]
    return {
        "model_name_or_path": model_name,
        "txt_filename": txt_filename,
        "ldr_filename": ldr_filename,
        "img_filename": img_filename,
        "model_name": model_name_short,
    }


def main():
    parser = HfArgumentParser(BrickGPTConfig)
    (cfg,) = parser.parse_args_into_dataclasses()

    brickgpt = BrickGPT(cfg)
    brickgpt.max_regenerations = 70
    prompt = input('Enter a prompt, or <Return> to exit: ')

    while True:
        if not prompt:
            break

        # Model version
        model_version = input('Options: 0 = baseline model, 1 = 2048 model, 2 = 2500 model, 3 = 2500 + partial. Enter 0, 1, 2, or 3: ')
        filename = input('Enter a filename to save the output image (default=output.png): ')
        
        result = prepare_output_paths(model_version, filename)
        model_path   = result["model_name_or_path"]
        txt_filename = result["txt_filename"]
        ldr_filename = result["ldr_filename"]
        img_filename = result["img_filename"]
        model_name   = result["model_name"]

        print(model_path, txt_filename, ldr_filename, img_filename, model_name)
        if model_path != None:
            brickgpt.model_name_or_path = model_path
            print(f"Setting model path to {model_path}")
        else:
            print("Using baseline model")

        # Seed
        seed = input('Enter a generation seed (default=42): ')
        seed = int(seed) if seed else 42
        transformers.set_seed(seed)

        # Generate bricks
        print(f'Generating with model version {model_name}...')
        start_time = time.time()
        output = brickgpt(prompt)
        end_time = time.time()

        # Save results
        with open(txt_filename, 'w') as f:
            f.write(output['bricks'].to_txt())
        with open(ldr_filename, 'w') as f:
            f.write(output['bricks'].to_ldr())
        render_bricks(ldr_filename, img_filename)

        # Print results
        print('--------------------')
        print(f'Finished generating in {end_time - start_time:.2f}s.')
        print('Total # bricks:', len(output['bricks']))
        print('Total # brick rejections:', output['rejection_reasons'].total())
        print('Brick rejection reasons:', dict(output['rejection_reasons']))
        print('Total # regenerations:', output['n_regenerations'])
        print(f'Saved results to {txt_filename}, {ldr_filename}, and {img_filename}')
        print('--------------------')

        prompt = input('Enter another prompt, or <Return> to exit: ')


if __name__ == '__main__':
    main()
