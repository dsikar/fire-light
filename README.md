# FireNet-LightWeight-Network-for-Fire-Detection
A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications  
(Preprint of the research paper on this work is available at https://arxiv.org/abs/1905.11922v2. Please consider citing if you happen to use the codes or dataset.
  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab_jp_nb_link] 
[colab_jp_nb_link]: https://colab.research.google.com/github/dsikar/fire-light/blob/master/Codes/ColabTrain.ipynb
  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiji14/deepbedmap/]
  
[![Multiple Top-Level VIs Demo Link](https://img.shields.io/badge/Details-Demo_Link-green.svg)][MultipleTopLevelVIsDemoLink]
[MultipleTopLevelVIsDemoLink]: https://ni.github.io/webvi-examples/MultipleTopLevelVIs/Builds/Web%20Server/Configuration1/MultipleTopLevelVIs/
### Citation
```
@article{jadon2019firenet,
  title={Firenet: A specialized lightweight fire \& smoke detection model for real-time iot applications},
  author={Jadon, Arpit and Omama, Mohd and Varshney, Akshay and Ansari, Mohammad Samar and Sharma, Rishabh},
  journal={arXiv preprint arXiv:1905.11922},
  year={2019}
}
```

In our paper we showed results on two datasets:
- A self created diverse dataset with images randomly sampled from our self-shot fire and non-fire
videos.
- Foggia's dataset (used for testing), which is available here (https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/).

It needs to be mentioned that the data used for training is only our custom created dataset, and not Foggia's full dataset. Testing was performed on previously held-out samples from our dataset as well as on Foggia's dataset. 

The following is the link to our dataset used in the FireNet paper:
- https://drive.google.com/drive/folders/1HznoBFEd6yjaLFlSmkUGARwCUzzG4whq?usp=sharing
