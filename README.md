# SimGRL
Official Code for SimGRL in NeurIPS 2024.

# Setup
```
conda env create -f setup/conda.yaml
conda activate dmcgb
sh setup/install_envs.sh
```

# Datasets

The [Places](http://places2.csail.mit.edu/download.html) dataset for data augmentation.

```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

The [DAVIS](https://davischallenge.org/davis2017/code.html) dataset for video backgrounds of Distracting Control Suite.

```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```

# Training
```
python src/train.py \
  --algorithm simgrl \
```

# Citation
If you find this code useful for your research, please use the following BibTeX entry.
```
@inproceedings{songsimple,
  title={A Simple Framework for Generalization in Visual RL under Dynamic Scene Perturbations},
  author={Song, Wonil and Choi, Hyesong and Sohn, Kwanghoon and Min, Dongbo},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

# Acknowledgement
This project is based on [SVEA](https://github.com/nicklashansen/dmcontrol-generalization-benchmark), we thank the original authors for their excellent work.
