# doggy
An AI model to classify dog's actions

## Setup
Follow this guide to setup the trainning and evaluation environment
### Amazon DLAMI
If you are using Amazon DLAMI, all the AI frameworks should be already installed, follow the start up screen to activate the training environment

### Your Own Environment
If you are using your own environment (the recommend way is virtualenv), follow this guide to setup Tensorflow https://www.linode.com/docs/applications/remote-desktop/install-vnc-on-ubuntu-16-04

### Libraries
We use Keras in this project, follow this guide to install it https://keras.io/#installation

We also use `pillow`, run the following command to install it
```
pip install pillow
```
## How to train
### Clone the repo
Clone this repo and download the images file

### Activate the environment
Activate the TensorFlow, depends on the environment you are in, e.g. for your local own environment, run this command `source ~/tensorflow/bin/activate`, for DLAMI, run the command for TensorFlow on the welcome screen, usually `source activate tensorflow_p36`

### Start trainning
Depends on where the images are saved, you might need to change the paths a bit inside `train.py`, change the following as required, by default the `dog_dataset` folder is the same as the `train.py`
```
# data dir
train_data_dir = 'dog_dataset/train'
validation_data_dir = 'dog_dataset/test'
```
Once the repo and images are downloaded, run this command in the TensorFlow environment `python train.py`

