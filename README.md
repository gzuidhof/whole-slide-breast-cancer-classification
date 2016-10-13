# msc

Repository for a Master's thesis project. Involves the classification of very high resolution whole slide images of breast cells. The goal is to best classify patches of these images into either benign vs cancer, or benign vs DCIS vs IDC.

## Reproducing the results

To repeat the experiments you can follow these steps. A lot of random factors are involved (cudnn not being deterministic, no fixed weight initialization and randomness in data sampling), so results may differ. Running these commands will train the same models as in the article, and generate the same plots and tables.


### Requirements

#### Software requirements
* Python 2.7
* CUDA (8.0 used for experiments, but >=6 should work)
* CUDNN (5.1 used for experiments, but >=3 should work)
* Python packages `numpy theano lasagne scipy tqdm scikit-learn scikit-image pandas matplotlib` 

Optional software requirements
* OpenCV 2 or 3 with Python bindings (will fallback to scikit-image otherwise)

*Note about running on Windows:* Multiprocessing may not behave as intended on Windows platforms, although this was kept in mind, it was not tested. All multiprocessing code also supports multithreading instead with simple changes to the config files.

#### Hardware requirements
* CUDA-enabled graphics card with 12GB of memory (eg. Nvidia Titan X). *Note you can do with less, but a smaller minibatch size must be used which may influence results*.
* A minimum of 32GB system memory. This allows for aggressive caching, which speeds up training.


#### Datasets
Two datasets consisting of whole-slide images are advised.

* **CroppedAnnotations_Revisited3classL1**
* **Level1_LargePatch_Aug17**

Only the second one is actually required, but using the first is a lot faster for the 224x224 patch networks. Both the `.tiff` and `.mrxs` multiresolution file format, as well as ordinary `.jpg` files are supported.


## Experiments

#### Configuration files
The settings for the different CNNs we will train are stored in declarative config files. These can be found in  the `/config/` folder. To use these config files one can invoke a script as following:

```shell
python resnet_trainer.py my_config
```
These are composable, multiple config files can be supplied as arguments. The later ones overwrite the earlier ones. **Use this to set the correct path to the data on your machine**. Instead of just the name, a full path to the file can also be used.


### 224x224 patch networks
1. Create a folder anywhere on the computer to store cached data for the sampler.
2. Create your own config file in the config folder, name it `my_data_location.ini` for example. See `config/beryllium.ini` for an example. Or simply change the existing configs to point towards this folder.
3. Prepare the sampler masks as such:
```
cd src/sampler
python wsi_prepare_masks.py 2class_vggnet my_data_location
python wsi_prepare_masks.py 3class_vggnet my_data_location
```
This will take between 10 and 30 minutes, but only has to be done once unique combination of patch size and amount of classes.


4. Train the networks. Of course you can train multiple in parallel:
```
cd src
python vgg_trainer.py 2_class_vggnet
python vgg_trainer.py 3_class_vggnet
python resnet_trainer.py 2_class_wide_resnet
python resnet_trainer.py 3_class_wide_resnet
```
To speed things up slightly cnmem can be used.

### Stacked networks
The stacked networks automatically select the network to stack in: the last trained one which completed 120 epochs.


1. Create a folder anywhere on the computer to store cached data for the sampler.
2. Create your own config file in the config folder, name it `my_data_location.ini` for example. See `config/beryllium.ini` for an example. Or simply change the existing configs to point towards this folder.
3. Prepare the sampler masks as such:
```
cd src/sampler
python wsi_prepare_masks.py stack_on_3class_512
python wsi_prepare_masks.py stack_on_2class_768
python wsi_prepare_masks.py stack_on_3class_768
python wsi_prepare_masks.py stack_on_3class_1024
```


4. Train the networks. Of course you can train multiple in parallel:
```
cd src
python resnet_trainer_stacked.py stack_on_3class_512
python resnet_trainer_stacked.py stack_on_2class_768
python resnet_trainer_stacked.py stack_on_3class_768
python resnet_trainer_stacked.py stack_on_3class_1024
```

### Generate plots and figures
Generating the figures and plots is fully automated. 
Just run

```
cd figures
mkdir output
python gen_graphs.py
```
To include them in the tex source, copy the output over to `tex/figures` and `tex/tables` manually.

### Dense prediction
To predict the whole full dataset, create a folder `wsi_predictions` and put a slides listing in `wsi_predictions/slides.txt`. See the file in the root of this project for an example.

Edit `src/sampler/wsi_predict.py` with your favorite editor. At the top of the file are capitalized fields which you should point to where your files are. Also, change the path to the model you wish to use for prediction. 

Then, on as many machines as you want (it automatically handles parallelization of predictions) run
```
cd src/sampler
python wsi_predict.py 36
```

The `36` argument is the batch size to use on this machine. This defaults to 42, which fits on a graphics card with 12GB of memory. This does not influence prediction results. Again, use cnmem for a slight speedup.

The results can be found in `wsi_predictions/`, they are also saved as images, so feel free to inspect them!

### Whole slide models (heatmap to slide label)
```
cd src
python feature_extraction.py
python slide_model.py
```















