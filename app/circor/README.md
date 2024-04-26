# Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection

![EMBC](https://embc.embs.org/2024/wp-content/uploads/sites/102/2023/05/ieee-embc-2024-logo2x.png)

This sub-repository provides codes for evaluating the performance of pre-trained models intended to reproduce the results in our [IEEE EMBC 2024](https://embc.embs.org/2024/) paper.
Our contents include:

- Data downloading and formatting notebook. It also covers code setup.
- Training/testing codes and utility batch scripts for reproducing our experiments.
- The command lines used for the paper.
- The notebook used to summarize and format results for the paper.

Please refer to the following paper (arXiv link) for the details.

```bibtex
@article{niizumi2024,
    title   = {{Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection}},
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Kunio Kashino},
    journal = {to appear at IEEE EMBC},
    year    = {2024},
    url     = {TBD}
}
```

## 1. Setup

[0-Prepare.ipynb](0-Prepare.ipynb) provides complete setup steps, including:
- Code setup (training/test program and external evaluation code)
- Downloading dataset from `physionet.org`
- Format the code for our experiments
- Integrity check for the data

### 1-1. Folders after the setup

You will find the following folders after the setup.

    bat      -- Batch scripts for automating experiments
    evar     -- Experiment runs under this folder
      /work  -- The data used during the training
    heart-murmur-detection  -- Copy of the repository of the previous study from Walker et al.
      /data  -- The data used for the final test
    m2d_vit_base-80x608p16x16-221006-mr7_enconly  -- The pre-trained M2D weight
    physionet.org  -- The copy of the dataset
    scores   -- The results of our paper

## 2. Run Experiments

We provide two example notebooks for running experiments.

- [1-Run-M2D.ipynb](1-Run-M2D.ipynb) provides an example of a complete command line. You can train a model using an M2D model, and you should obtain a result close to the paper. You can also check the details of fine-tuning parameters.
- [2-Run-BYOL-A.ipynb](2-Run-BYOL-A.ipynb) provides an example of the experiment using a batch file. This is exactly what we performed for the paper.

Please find the complete command line in [Command lines used for the paper](#command-lines-used-for-the-paper).

## 3. Summarizing the results

[9-Summarize-results-CirCor.ipynb](9-Summarize-results-CirCor.ipynb) provides complete steps to summarize the results using our result files in the `scores` folder.

## Files

This sub-repository contains the following files:

- 0-Prepare.ipynb -- A notebook for preparing the experiment
- 1-Run-M2D.ipynb -- A notebook for the M2D experiment
- 2-Run-BYOL-A.ipynb -- A notebook for the BYOL-A experiment
- 9-Summarize-results-CirCor.ipynb -- A notebook for summarizing results
- circor_eval.py -- The main program for the experiment
- bat/*.sh -- Scripts for automating experiments for each pre-trained audio representation
- diff-evar.patch -- A patch file for EVAR
- diff-heart-murmur-detection.patch -- A patch file for heart-murmur-detection

## Acknowledgements

We appreciate the previous studies that shared their codes.
Our code uses [Benjamin-Walker/heart-murmur-detection](https://github.com/Benjamin-Walker/heart-murmur-detection) from the paper:

```bibtex
@article{walker2022DBResNet,
    title={Dual Bayesian ResNet: A Deep Learning Approach to Heart Murmur Detection},
    author={Benjamin Walker and Felix Krones and Ivan Kiskin and Guy Parsons and Terry Lyons and Adam Mahdi},
    journal={Computing in Cardiology},
    volume={49},
    year={2022}
}
```

## Command lines used for the paper

We used the following command lines.

Please note that the following contains `m2d_vit_base-80x608p16x16-220930-mr7` pre-trained weights, which we actually used while we are providing `m2d_vit_base-80x608p16x16-221006-mr7_enconly` here.
The latter provides better results, so we replace the weight on the repository.

```sh
cd evar
bash ../bat/m2d_ftcircor.sh ../m2d_vit_base-80x608p16x16-220930-mr7 1 5 7 300
bash ../bat/m2d_ftcircor.sh ../m2d_vit_base-80x608p16x16-220930-mr7 2 5 7 300
bash ../bat/m2d_ftcircor.sh ../m2d_vit_base-80x608p16x16-220930-mr7 3 5 7 300

bash ../bat/ast_ftcircor.sh 1 5 42
bash ../bat/ast_ftcircor.sh 2 5 42
bash ../bat/ast_ftcircor.sh 3 5 42

bash ../bat/byola_ftcircor.sh 1 5 42
bash ../bat/byola_ftcircor.sh 2 5 42
bash ../bat/byola_ftcircor.sh 3 5 42

bash ../bat/cnn14_ftcircor.sh 1 5 42
bash ../bat/cnn14_ftcircor.sh 2 5 42
bash ../bat/cnn14_ftcircor.sh 3 5 42

bash ../bat/m2d_ftcircor_rand.sh m2d_vit_base-80x608p16x16 1 5 7
bash ../bat/m2d_ftcircor_rand.sh m2d_vit_base-80x608p16x16 2 5 7
bash ../bat/m2d_ftcircor_rand.sh m2d_vit_base-80x608p16x16 3 5 7

bash ../bat/ast_ftcircor_noaug.sh 1 5 42
bash ../bat/ast_ftcircor_noaug.sh 2 5 42
bash ../bat/ast_ftcircor_noaug.sh 3 5 42

bash ../bat/byola_ftcircor_noaug.sh 1 5 42
bash ../bat/byola_ftcircor_noaug.sh 2 5 42
bash ../bat/byola_ftcircor_noaug.sh 3 5 42

bash ../bat/cnn14_ftcircor_noaug.sh 1 5 42
bash ../bat/cnn14_ftcircor_noaug.sh 2 5 42
bash ../bat/cnn14_ftcircor_noaug.sh 3 5 42
```