# Introduction
This is my submission for the "Create Your Own Image Classifier" project as part of the Udacity nanodegree [AI Programming with Python](https://learn.udacity.com/nanodegrees/nd089).  The point of this project is to demonstrate understanding of deep learning concepts and the `pytorch` library through a toy classification task.  The task is to classify images of flowers into one of 102 possible categories.  This is accomplished using transfer learning starting with a neural network trained on ImageNet data (options: `AlexNet`, `DenseNet121`, `VGG13`, `VGG16`, or `RESNET18`).  I freeze the pretrained model layers and replace the classification layer(s) with a new neural network, which is then trained for the flowers classification task.

The original notebook that was used to work through the entire process is [Image Classifier Project.ipynb](<aipnd-project/Image Classifier Project.ipynb>).  This was then refactored into two command line applications: one for training ([train.py](aipnd-project/train.py)) and one for classification predictions given an image ([predict.py](aipnd-project/predict.py)).

# Setup
## Python virtual environment
Set up the `pyenv` virtual environment.

```
pyenv virtualenv 3.11.7 udacity-flower-classifier
pyenv local udacity-flower-classifier
python3 -m pip install --upgrade pip

git clone git@github.com:mrperkett/udacity-project-create-image-classifier.git
cd udacity-project-create-image-classifier/
python3 -m pip install -r requirements.txt

# register IPython kernel
python3 -m ipykernel install --user --name udacity-flower-classifier
```

## Jupyter
For this project, I used the VSCode [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) (v2024.4.0) to run the Jupyter notebook.  There are other options for running Jupyter notebooks.  For another option using `pyenv` and the `jupyter lab` via the command line, see [this repo](https://github.com/mrperkett/udacity-project-finding-donors).


# Running
## `Image Classifier Project.ipynb` Notebook
Can be run in the usual way; see setup section for details.  See [Image Classifier Project.html](<aipnd-project/Image Classifier Project.html>) for the HTML export of the notebook.

## Command line scripts

### `train.py`
```
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch {alexnet,densenet121,resnet18,vgg13,vgg16}] [--learning_rate LEARNING_RATE]
                [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS] [--dropout DROPOUT] [--gpu] [--category_names CATEGORY_NAMES]
                data_directory

positional arguments:
  data_directory        Path to the directory containing the 'train', 'valid', and 'test' folders containing the jpeg images

options:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Path to the directory in which to save the model checkpoint file when training completes.
  --arch {alexnet,densenet121,resnet18,vgg13,vgg16}
                        Pretrained model architecture from which to start training
  --learning_rate LEARNING_RATE
                        Learning rate during model training
  --hidden_units HIDDEN_UNITS
                        Number of units in the hidden layer
  --epochs EPOCHS       Number of epochs to train
  --dropout DROPOUT     Dropout rate for hidden layer during training (in range (0.0, 1.0))
  --gpu                 Train using GPU
  --category_names CATEGORY_NAMES
                        JSON input file specifying the category label to class label (e.g. '1': 'pink primrose')
```

#### Example run AlexNet
```
$ python3 ./train.py "/mnt/c/Large Files/flower_data" --save_dir "." --arch alexnet --learning_rate 0.001 --hidden_units 512 --epochs 1 --gpu --category_names "cat_to_name.json" --dropout 0.2

Creating training/validation flower data loaders
        training_data_loader
Loading alexnet base model and modifying classifier layer
Training model for 1 epoch(s).
epoch 0
        batch_num: 30 / 103
        batch_num: 60 / 103
        batch_num: 90 / 103
        batch_num: 103 / 103
        training run time:     65.4
        validation run time:   7.9
        training avg loss:     2.6946
        training avg accuracy: 0.3840
        validation avg loss:   1.0448
        validation accuracy:   0.7127
run_time: 73.2
Saving checkpoint file: ./checkpoint.pt
Done
```

#### Example run DenseNet121
```
$ python3 ./train.py "/mnt/c/Large Files/flower_data" --save_dir "." --arch densenet121 --learning_rate 0.001 --hidden_units 512 --epochs 1 --gpu --category_names "cat_to_name.json" --dropout 0.2

Creating training/validation flower data loaders
        training_data_loader
Loading densenet121 base model and modifying classifier layer
Training model for 1 epoch(s).
epoch 0
        batch_num: 30 / 103
        batch_num: 60 / 103
        batch_num: 90 / 103
        batch_num: 103 / 103
        training run time:     83.7
        validation run time:   9.4
        training avg loss:     3.0806
        training avg accuracy: 0.3472
        validation avg loss:   1.3175
        validation accuracy:   0.7323
run_time: 93.1
Saving checkpoint file: ./checkpoint.pt
Done
```

#### Example run default parameters
```
# Equivalent to
# python3 ./train.py "/mnt/c/Large Files/flower_data" --save_dir "." --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 1
python3 ./train.py "/mnt/c/Large Files/flower_data"

Creating training/validation flower data loaders
        training_data_loader
Loading vgg13 base model and modifying classifier layer
Training model for 1 epoch(s).
epoch 0
        batch_num: 30 / 103
        batch_num: 60 / 103
        batch_num: 90 / 103
        batch_num: 103 / 103
        training run time:     489.8
        validation run time:   61.8
        training avg loss:     7.5153
        training avg accuracy: 0.0846
        validation avg loss:   4.0451
        validation accuracy:   0.1308
run_time: 551.6
Saving checkpoint file: /mnt/c/Users/mattp/OneDrive/Documents/Training/Udacity - AI Programming with Python/06 Project - Create Your Own Image Classifier/udacity-project-create-image-classifier/aipnd-project/checkpoint.pt
Done
```



### `predict.py`
```
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu] image_file_path model_file_path

positional arguments:
  image_file_path       File path for the jpeg image to classify
  model_file_path       File path for the saved model

options:
  -h, --help            show this help message and exit
  --top_k TOP_K         Display the top <top_k> predicted classes
  --category_names CATEGORY_NAMES
                        JSON input file specifying the category label to class label (e.g. '1': 'pink primrose')
  --gpu                 Predict using GPU
```

#### Example run AlexNet
```
python3 ./predict.py "test_images/cautleya_spicata.jpg" "checkpoint-nhidden_512-nepochs_1-alexnet.pt" --top_k 5 --category_names "cat_to_name.json" --gpu

idx     label     prob       name
60      '61'      0.9144     cautleya spicata
85      '84'      0.0313     columbine
33      '37'      0.0100     cape flower
4       '102'     0.0088     blackberry lily
55      '57'      0.0049     gaura
```

#### Example run DenseNet121
```
python3 ./predict.py "test_images/cautleya_spicata.jpg" "checkpoint-nhidden_512-nepochs_1-densenet121.pt" --top_k 5 --category_names "cat_to_name.json" --gpu

idx     label     prob       name
60      '61'      0.2977     cautleya spicata
55      '57'      0.0937     gaura
57      '59'      0.0866     orange dahlia
33      '37'      0.0714     cape flower
94      '92'      0.0703     bee balm
```

#### Example run default parameters

Warning: this runs on the CPU, which is slower.

```
python3 ./predict.py "test_images/cautleya_spicata.jpg" "checkpoint.pt"

idx     label     prob       name
90      '89'      0.0165
```


### `run_all.sh`
Run batch training and predictions (see [run_all.sh](aipnd-project/run_all.sh)).

```
./run_all.sh > output.txt
```

See [output-nhidden_512_nepochs_1.txt](output/output-nhidden_512_nepochs_1.txt) and [output-nhidden_512_nepochs_5.txt](output/output-nhidden_512_nepochs_5.txt) for example output.

# Results

See [Image Classifier Project.ipynb](<aipnd-project/Image Classifier Project.ipynb>) ([HTML](<aipnd-project/Image Classifier Project.html>)).

## Training and prediction summary
See [run_all.sh](aipnd-project/run_all.sh) for script to generate this data.

### `num_epochs = 1`
| | training run time | validation testing run time | validation accuracy |
|-|----------------|------------------------|------------|
| AlexNet | 73.6 | 9.1 | 0.742 |
| DenseNet121 | 91.7 | 12.0 | 0.719 |
| RESNET18 | 91.7 | 9.3 | 0.694 |
| VGG13 | 102.5 | 13.6 | 0.796 |
| VGG16 | 140.3 | 18.3 | 0.814 |

### `num_epochs = 5`

| | total run time | validation accuracy |
|-|-------------------|---------------------|
| AlexNet | 422 | 0.846 |
| DenseNet121 | 701 | 0.910 |
| RESNET18 | 486 | 0.908 |
| VGG13 | 746 | 0.899 |
| VGG16 | 795 | 0.881 |

# Base Models
For reference, I've put information on the classifier layers present in each of the allowed base models trained on ImageNet data.

```
>>> from torchvision import models
>>> from torchvision.models.densenet import DenseNet121_Weights
>>> from torchvision.models.resnet import ResNet18_Weights
>>> from torchvision.models.alexnet import AlexNet_Weights
>>> from torchvision.models.vgg import VGG16_Weights, VGG13_Weights

>>> models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).classifier
Sequential(
  (0): Dropout(p=0.5, inplace=False)
  (1): Linear(in_features=9216, out_features=4096, bias=True)
  (2): ReLU(inplace=True)
  (3): Dropout(p=0.5, inplace=False)
  (4): Linear(in_features=4096, out_features=4096, bias=True)
  (5): ReLU(inplace=True)
  (6): Linear(in_features=4096, out_features=1000, bias=True)
)

>>> models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).classifier
Linear(in_features=1024, out_features=1000, bias=True)

>>> models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).fc
Linear(in_features=512, out_features=1000, bias=True)

>>> models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1).classifier
Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU(inplace=True)
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace=True)
  (5): Dropout(p=0.5, inplace=False)
  (6): Linear(in_features=4096, out_features=1000, bias=True)
)

>>> models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).classifier
Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU(inplace=True)
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace=True)
  (5): Dropout(p=0.5, inplace=False)
  (6): Linear(in_features=4096, out_features=1000, bias=True)
)
```

# Data
The data used in this exercise originates from the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  Udacity provides the images presorted into directories `train/` `test/` `valid/`.  

I plan to write scripts allowing you to reproduce the data organization in the Udacity format in the future.  For now, I will put some general notes below.

**Udacity-provided data**
- train: 6552
- test: 819
- valid: 818
- total: 8189 

**Downloaded data**
- train: 1020
- test: 6149
- valid: 1020
- total: 8189 

It looks like the same images were used by Udacity, but they have divided up the sets differently.

You can download the files you need here:

- [dataset images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
- [image labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)
- [data splits](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat)

You can use `scipy.io` to load the `.mat` files.

```
>>> import scipy.io
>>> mat = scipy.io.loadmat("../setid.mat")
>>> mat["tstid"][0].shape
(6149,)
>>> mat["trnid"][0].shape
(1020,)
>>> mat["valid"][0].shape
(1020,)
```