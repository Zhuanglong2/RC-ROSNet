# RC-ROSNet
### **RC-ROSNet: Fusing 3D Radar Range-Angle Heat Maps and Camera Images for Radar Object Segmentation**

**by Long Zhuang, Taihong Yang, and Yiqing Yao**

## Installation

0. Clone the repo:

```bash
$ git clone https://github.com/Zhuanglong2/RC-ROSNet.git
```

1. Create a conda environment using:

```bash
cd $ROOT/RCROSNet
conda env create -f env.yml
conda activate RCROSNet
pip install -e .
```

Due to certain discrepancies with scikit library, you might need to do:

```bash
pip install scikit-image
pip install scikit-learn
```

NOTE: We also provide `requirements.txt` file for venv enthusiasts.

2. Dataset:

The CARRADA dataset is available on Valeo.ai's github: [https://github.com/valeoai/carrada_dataset](https://github.com/valeoai/carrada_dataset).

## Running the code:

You must specify the path at which you store the logs and load the data from, this is done through:

```bash
cd $ROOT/RC-ROSNet-main/mvrss/utils/
python set_paths.py --carrada -dir containing the Carrada file- --logs -dir_to_output-
```

## Training

```bash
cd $ROOT/RC-ROSNet-main/mvrss/ 
python -u train.py --cfg ./config_files/RC-RODNet.json --cp_store -dir_to_checkpoint_store-
```

## Testing

```bash
$ cd $ROOT/TransRadar/mvrss/ 
$ python -u test.py --cfg $ROOT/RC-ROSNet-main/mvrss/logs/carrada/RC-RODNet/RC-RODNet_3/config.json
```

## Important note:

The pre-trained weights provided were obtained using an NVIDIA GTX 3060Ti GPU. Due to the significant variance in radar data observed across different hardware platforms, we recommend that researchers perform re-evaluation on their own systems for fairness.

## Acknowledgements

This repository heavily borrows from [TransRadar](https://openaccess.thecvf.com/content/WACV2024/papers/Dalbah_TransRadar_Adaptive-Directional_Transformer_for_Real-Time_Multi-View_Radar_Semantic_Segmentation_WACV_2024_paper.pdf)


