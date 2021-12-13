Requirements
============

gcc & g++ ≥ 5.4

    pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


.. _Model: https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#-model-zoo-and-baselines
.. _Config: https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs

Download model from Model_ and related config file from Config_.::

    e.g.:
        model: R_50_FPN_s1x (file model_final_162be9.pkl)
        config: densepose_rcnn_R_50_FPN_s1x.yaml


::

    0      = Background
    1, 2   = Torso
    3      = Right Hand
    4      = Left Hand
    5      = Right Foot
    6      = Left Foot
    7, 9   = Upper Leg Right
    8, 10  = Upper Leg Left
    11, 13 = Lower Leg Right
    12, 14 = Lower Leg Left
    15, 17 = Upper Arm Left
    16, 18 = Upper Arm Right
    19, 21 = Lower Arm Left
    20, 22 = Lower Arm Right
    23, 24 = Head
