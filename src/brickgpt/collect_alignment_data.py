import os
import time

import transformers

from brickgpt.models import BrickGPT, BrickGPTConfig
from brickgpt.render_bricks import render_bricks




def data_for_a_prompt(model, prompt, seed=42):
    if not prompt:
        return

    transformers.set_seed(seed)

    # Generate bricks
    print('Generating...')
    start_time = time.time()
    output = model(prompt)
    end_time = time.time()

    # Print results
    print('--------------------')
    print(f'Finished generating in {end_time - start_time:.2f}s.')
    print('Total # bricks:', len(output['bricks']))
    print('Total # brick rejections:', output['rejection_reasons'].total())
    print('Brick rejection reasons:', dict(output['rejection_reasons']))
    print('Total # regenerations:', output['n_regenerations'])
    print('--------------------')



    

if __name__ == '__main__':
    ######
    # Configs

    cfg = BrickGPTConfig()
    cfg.world_dim = 20
    cfg.max_bricks = 2000
    cfg.max_brick_rejections = 500
    cfg.use_logit_masking = True
    cfg.max_regenerations = 100
    cfg.use_gurobi = True
    cfg.temperature = 0.6
    cfg.temperature_increase = 0.01
    cfg.max_temperature = 2.0
    cfg.top_k = 20
    cfg.top_p = 1.0

    brickgpt = BrickGPT(cfg)

    prompts = [
        "Curved-back chair with a sleek, minimalist form.",
        "Chair featuring a curved backrest and angular legs.",
        "Chair with a distinct curved backrest, supported by angular base pieces.",
        "The chair has a smoothly curved backrest and splayed, angular supports.",
        "A chair with an elegantly curved backrest, the seat gently inclines backward. The angular supports splay outward, enhancing stability. The backrest transitions into the seat, creating a seamless flow from the top of the chair down to the base."
    ]
    for prompt in prompts[0:1]:
        data_for_a_prompt(brickgpt, prompt)