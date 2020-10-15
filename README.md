nbv_3d_prob_cnn
===============

The nbv_3d_cnn package was used to test multiple CNN-based and traditional NBV methods for research.

Methods:

- Random
- CNNDirectional (requires CNN model "" (empty string))
- CNNFlat (requires CNN model `flat`)
- CNNQuat (requires CNN model `quat`)
- InformationGain
- InformationGainProb
- AutocompleteIGain (requires CNN model `autocomplete`)
- OmniscientGain

Build
-----

This is a standard ROS (Robot Operating System) package, which may be compiled with `catkin build` or `catkin_make`.
Tested on Ubuntu 18.04 with ROS melodic.

System dependencies:

- OpenCV
- Eigen3
- OctoMap
- OpenCL
- Tensorflow

Usage
-----

### 2D tests

- Place the 2D scene dataset, composed of `.tif` files, into `data/inria_dataset/AerialImageDataset/train/gt/`. Files in the right format may be downloaded from the ground truth of the [INRIA dataset](https://project.inria.fr/aerialimagelabeling/). The dataset is very large, but only the ground truth is needed here.
- Launch `generate_test_dataset.launch` to generate the ground truth for all methods. Ground truth is placed into `data/inria_environments`.<br />
  **Note**: create the directory if it does not exists.
- Launch `train_2d.launch`. Edit parameter `model_type` to select the network to be trained. Output files are written into the `data/output` folder. Tensorboard data are written into the `data/tensorboard` folder.
- Copy the checkpoint file from `data/output` to `data/trained_models`.
- Launch `simulate_nbv_cycle.launch`. Change parameter `nbv_algorithm` to select the method. Change parameter `image_file_name` to select the environment image. Statistics are saved to `data/logs`. Debug output is saved to `data/simulate_nbv_cycle`.

### 3D tests

- Place the 3D scene dataset, composed of OctoMap `.bt` files, into `data/scenes`. Sample files generated randomly from the [YCB benchmark](https://www.ycbbenchmarks.com/) may be downloaded from [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_generated_scenes_3d.zip).
- Launch `generate_test_dataset_3d.launch` to generate the ground truth for all methods. Ground truth is placed into `data/environments_3d`.
- Data augmentation for `flat` and `quat` 3D CNNs should be done offline, to speedup training. Launch `augment_test_dataset_3d.launch` for data augmentation.<br />**Warning**: this may increase the size of the `data/environments_3d` folder by one or two hundred GB.
- Launch `train_3d.launch`. Edit parameter `model_type` to select the network to be trained. Output files are written into the `data/output` folder. Tensorboard data are written into the `data/tensorboard` folder.
- Copy the checkpoint file from `data/output` to `data/trained_models`.
- Launch `simulate_nbv_cycle_3d.launch`. Change parameter `nbv_algorithm` to select the method. Change parameter `image_file_name` to select the environment image. Statistics are saved to `data/logs_3d`. Debug output is saved to `data/simulate_nbv_cycle_3d`.
