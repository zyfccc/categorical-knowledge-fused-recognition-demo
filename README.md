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

<img src="https://private-user-images.githubusercontent.com/5927085/418439074-9b2d8adc-5ea4-4d75-ae08-354c8b72b42d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDU1NjAwMzQsIm5iZiI6MTc0NTU1OTczNCwicGF0aCI6Ii81OTI3MDg1LzQxODQzOTA3NC05YjJkOGFkYy01ZWE0LTRkNzUtYWUwOC0zNTRjOGI3MmI0MmQucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDQyNSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA0MjVUMDU0MjE0WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9MTY2ZGY0MjdkN2Y0N2ZmY2M1NmQwYjFjYjJjYzMyY2ExNDA5MDliNGJjODBjY2IwZWE2OWVhMTE4YmNmNjE3NCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.Im2sJIfZMtX2T8d1sIEOuFS1OgEuhFmwAXdOp7kMrdE">

Compared to the baseline in the CAM plots, CKFR enhanced object identification and localization capacities as well as the reasoning aspect of the recognition.


## Paper

Here is the [link](https://ieeexplore.ieee.org/document/10959394) to the paper.

And the paper can be cited as follow:

```
@inproceedings{ckfr2024,
  author={Zhao, Yunfeng and Zhou, Huiyu and Wu, Fei and Wu, Xifeng},
  booktitle={2024 IEEE 8th International Conference on Vision, Image and Signal Processing (ICVISP)}, 
  title={Categorical Knowledge Fused Recognition: Fusing Hierarchical Knowledge with Image Classification Through Aligning and Deep Metric Learning}, 
  year={2024},
  pages={1-8}
}
```
