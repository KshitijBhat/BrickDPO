import os
import time

import transformers
from transformers import HfArgumentParser

from brickgpt.models import BrickGPT, BrickGPTConfig
from brickgpt.render_bricks import render_bricks

from brickgpt.data import max_brick_dimension, BrickStructure, Brick


def main():
    base_name = "mid_partial2"
    img_filename = os.path.abspath(base_name + '.png')
    ldr_filename = os.path.abspath(base_name + '.ldr')


    BrickStructure_obj = BrickStructure.from_txt("1x2 (7,15,0)\n1x1 (7,13,0)\n1x2 (7,5,0)\n2x2 (6,7,0)\n1x1 (6,5,0)\n2x6 (5,11,0)\n2x2 (5,3,0)\n2x6 (3,11,0)\n2x6 (3,3,0)\n2x6 (1,11,0)\n6x2 (1,9,0)\n2x2 (1,3,0)\n1x2 (0,15,0)\n1x1 (0,13,0)\n2x2 (0,7,0)\n1x2 (0,5,0)\n1x1 (7,18,1)\n1x2 (7,16,1)\n2x2 (6,14,1)\n2x6 (6,8,1)\n2x4 (6,4,1)\n2x1 (6,3,1)\n1x2 (5,10,1)\n2x6 (4,2,1)\n2x2 (3,10,1)\n1x2 (2,10,1)\n4x2 (2,8,1)\n2x6 (2,2,1)\n6x1 (1,19,1)\n6x2 (1,17,1)\n6x1 (1,16,1)\n1x1 (1,3,1)\n1x2 (0,16,1)\n6x2 (0,14,1)\n2x6 (0,8,1)\n2x4 (0,4,1)\n1x2 (7,13,2)\n1x2 (7,9,2)\n2x4 (6,15,2)\n2x4 (6,5,2)\n1x1 (5,16,2)\n1x1 (5,5,2)\n2x1 (3,5,2)\n4x1 (2,19,2)\n1x1 (2,16,2)\n1x1 (2,5,2)\n6x2 (2,3,2)\n1x2 (1,18,2)\n1x1 (1,17,2)\n1x4 (1,3,2)\n2x4 (0,13,2)\n8x1 (0,12,2)\n2x2 (0,9,2)\n2x1 (0,7,2)\n1x1 (0,6,2)\n1x2 (0,4,2)\n1x1 (7,18,3)\n1x2 (7,16,3)\n1x1 (7,14,3)\n1x1 (7,10,3)\n1x2 (7,6,3)\n2x2 (6,11,3)\n2x1 (6,9,3)\n1x1 (4,12,3)\n1x1 (4,3,3)\n4x1 (2,19,3)\n2x1 (2,12,3)\n6x2 (2,4,3)\n6x2 (1,17,3)\n6x2 (1,15,3)\n6x2 (1,6,3)\n1x2 (1,4,3)\n")
    ldr = BrickStructure_obj.to_ldr()
    with open(ldr_filename, 'w') as f:
        f.write(ldr)


    render_bricks(ldr_filename, img_filename)


if __name__ == '__main__':
    main()