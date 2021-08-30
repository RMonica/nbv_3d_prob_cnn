nbv_3d_cnn
==========

The nbv_3d_cnn package was used to compare multiple CNN-based and traditional NBV methods for research.

**Methods**:

- **Random**: random view pose selection.
- **CNNDirectional**: direct gain prediction for every viewpoint and ray, on a grid (requires CNN model `""` (empty string))
- **CNNFlat**: direct gain prediction for every viewpoint and every direction, flat (requires CNN model `flat`)
- **CNNQuat**: prediction of the best direction for each viewpoint, as quaternion (requires CNN model `quat`)
- **InformationGain**: number of visible unknown voxels
- **InformationGainProb**: probabilistic number of visible unknown voxels
- **AutocompleteIGain**: probabilistic number of visible unknown voxels, occupancy probability is predicted using CNN (requires CNN model `autocomplete`)
- **AutocompleteFixedNumberIGain**: as AutocompleteIGain, but only a random subset of viewpoints are evaluated (requires CNN model `autocomplete`)
- **OmniscientGain**: cheating method, using the perfect knowledge of the environment from the ground truth

Results have been published in the article:

- R. Monica, J. Aleotti, _A Probabilistic Next Best View Planner for Depth Cameras based on Deep Learning_, IEEE Robotics and Automation Letters, 2021

The methods above correspond to the methods presented in the article as follows:

- **Random**: _Random_.
- **CNNDirectional**: _D-City-CNN_.
- **CNNFlat**: _Flat City-CNN_.
- **CNNQuat**: _B-City-CNN_.
- **InformationGain**: _Information Gain_.
- **InformationGainProb**: _Probabilistic Information Gain_.
- **AutocompleteIGain**: _CNN Probabilistic_ (proposed approach).
- **AutocompleteFixedNumberIGain**: _CNN Probabilistic Downsampled_.
- **OmniscientGain**: _Omniscient (oracle)_.

Build
-----

This is a standard ROS (Robot Operating System) package, which may be compiled with `catkin build` or `catkin_make`.
Tested on Ubuntu 18.04 with ROS Melodic.

System dependencies:

- OpenCV
- Eigen3
- OctoMap
- Point Cloud Library
- OpenCL
- Tensorflow

**Note**: by default, ROS compiles without optimizations and produces a very slow executable. If you haven't already, please activate optimizations. Example commands:

```
  catkin_make -DCMAKE_BUILD_TYPE=RelWithDebInfo
  catkin build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Usage
-----

Tests were performed on three datasets, a 2D dataset, a 3D unstructured dataset (3DU) and a 3D tabletop dataset (3DT).

**Note**: as the directory structure cannot be committed into Git, please create any non-existing directory.

### Pre-trained models

Pre-trained models may be downloaded from here: <http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_trained_models.zip>.

Models should be placed into folder `data/trained_models`.

### Training

**2D**

- Place the 2D scene dataset, composed of `.tif` files, into `data/inria_dataset/AerialImageDataset/train/gt/`. Files in the right format may be downloaded from the ground truth of the [INRIA dataset](https://project.inria.fr/aerialimagelabeling/). The dataset is very large, but only the ground truth (the black and white images) is needed in this case.
- Launch `generate_test_dataset.launch` to generate the ground truth for all methods. Ground truth is placed into `data/inria_environments`.
- Launch `train_2d.launch`. Edit parameter `model_type` to select the network to be trained. Checkpoint files are written into the `data/output` folder. Tensorboard data is written into the `data/tensorboard` folder.
- When done, copy the checkpoint file from `data/output` to `data/trained_models`.

**3DU**

- Place the 3DU dataset, composed of OctoMap `.bt` files, into `data/scenes`. Sample files generated randomly from the [YCB benchmark](https://www.ycbbenchmarks.com/) may be downloaded from [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_generated_scenes_3d.zip).
- Launch `generate_test_dataset_3d.launch` to generate the ground truth for all methods. Ground truth is placed into `data/environments_3d`.
- Data augmentation for `flat` and `quat` 3D CNNs should be done in advance, to speedup training. Launch `augment_test_dataset_3d.launch` for data augmentation.<br />**Warning**: this may increase the size of the `data/environments_3d` folder by one or two hundred GB.
- Launch `train_3d.launch`. Edit parameter `model_type` to select the network to be trained. Checkpoint files are written into the `data/output` folder. Tensorboard data is written into the `data/tensorboard` folder.
- Copy the checkpoint file from `data/output` to `data/trained_models`.

**3DT**

- Place the 3DT tabletop dataset, composed of OctoMap `.bt` files, into `data/scenes_realistic`. Sample files generated randomly from the [RD dataset](http://rimlab.ce.unipr.it/%7ermonica/grasping_rd_dataset.zip) can be downloaded from [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_scenes_realistic.zip).
- Launch `generate_test_dataset_3d_realistic.launch` to generate the ground truth for all methods. Ground truth is placed into `data/environments_3d_realistic`.
- Launch `augment_test_dataset_3d_realistic.launch` to perform data augmentation.
- Launch `train_3d_realistic.launch` for training. Edit parameter `model_type` to select the network to be trained. Checkpoint files are written into the `data/output` folder. Tensorboard data is written into the `data/tensorboard` folder.
- Copy the checkpoint file from `data/output` to `data/trained_models/realistic`.

### Evaluation

**Common parameters**

These parameters may be set for the `simulate_nbv_cycle` node into `simulate_nbv_cycle.launch`, `simulate_nbv_cycle_3d.launch` and `simulate_nbv_cycle_3d_realistic.launch`.

- `nbv_algorithm` (string): select the method string (see Methods above).
- `log_file` (string): output log file name
- `image_file_name` (string): test environment file name (image for 2D tests, OctoMap `.bt` for 3D tests)
- `sample_fixed_number_of_views` (int): number of view samples for the AutocompleteFixedNumberIGain method
- `max_iterations` (int): number of NBV iterations to be performed
- `save_images` (bool): whether to save debug images
- `random_seed` (int): random seed (for repeatability)
- `a_priori_occupied_prob` (double): a-priori occupancy probability for InformationGainProb
- `sensor_resolution_x`, `sensor_resolution_y`, `sensor_focal_length`: camera intrinsics

**2D**

- Launch `simulate_nbv_cycle.launch`. Statistics are saved to `data/logs`. Debug output is saved to `data/simulate_nbv_cycle`.

**3DU**

- Launch `simulate_nbv_cycle_3d.launch`. Statistics are saved to `data/logs_3d`. Debug output is saved to `data/simulate_nbv_cycle_3d`.

**3DT**

- Launch `simulate_nbv_cycle_3d_realistic.launch`. Statistics are saved to `data/logs_3d_realistic`. Debug output is saved to `data/simulate_nbv_cycle_3d_realistic`.

2021-08-30