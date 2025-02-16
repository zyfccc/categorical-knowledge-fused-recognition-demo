## Intro

In the paper titled "Categorical Knowledge Fused Recognition: Fusing Hierarchical Knowledge with Image Classification through Aligning and Deep Metric Learning", a novel deep metric learning based CKFR method has been proposed to effectively fuse prior knowledge about image categories with mainstream backbone image classification models through an aligning process in an end-to-end manner. The CKFR method enhances the reasoning aspect of image recognition. This is the repo that contains demonstration code of the CKFR method.


## Run the code

### Prepare the dataset

The mini-Imagenet dataset is used as a demo. The dataset is prepared with the following structure:

```
  -| mini-imagenet
    -- train.csv
    -- val.csv
    -- test.csv
    -| images
      -- *.jpg
```


The project file structure for running the code is as follows:

```
-| this-repo
  -| libs
    -- ImageDataset_train.py
    -- ImageDataset_val.py
  -| model
    -- pretrained.py
  -- ckfr_train_demo.py
  -- dis_imagenet.npy   # distance cache file

-| mini-imagenet
```


### Prepare environment

 - Python 3
 - Pytorch 2
 - Cuda 12


### Run the script

Below is an example command to run the CKFR training demo:

```
python ckfr_train_demo.py --alpha 20 --ell 1.2 --epochs 30 --lr 1e-5
```


## Demo results

Original sample images (first row) and Class Activation Maps (GradCAMs) produced from baseline (second row) and CKFR method (third rows):

<img src="https://private-user-images.githubusercontent.com/5927085/413603888-160ded0a-71d3-471b-aa16-a17a32d53b19.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzk2OTg3MTgsIm5iZiI6MTczOTY5ODQxOCwicGF0aCI6Ii81OTI3MDg1LzQxMzYwMzg4OC0xNjBkZWQwYS03MWQzLTQ3MWItYWExNi1hMTdhMzJkNTNiMTkucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDIxNiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTAyMTZUMDkzMzM4WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NmQzZGI0YWNlZmVhZTRiNzk2NjVhNDE4NGM3Y2VlZWM5MjkxOTg2MGYzMGMxZTdmZDE2YTU1OTcyNjNlMjYxMyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.U_rxPYJKuMLNhmkxyBQqFkXTndJKKO9sGafBp3VI25s">

Compared to the baseline in the CAM plots, CKFR enhanced object identification and localization capacities as well as the reasoning aspect of the recognition.


## Pre-print

Here is the [link](https://arxiv.org/abs/2407.20600) to the pre-print paper.
