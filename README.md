# Learning-to-See-in-the-Dark
[Project Website](http://web.engr.illinois.edu/~cchen156/SID.html)<br/>
This is a Tensorflow implementation of Learning to See in the Dark, in CVPR 2018.  

![teaser](images/fig1.png "Sample inpainting results on held-out images")

## Demo Video
https://youtu.be/3PcxO-mvzPs

<iframe width="480" height="270" src="https://www.youtube.com/embed/3PcxO-mvzPs" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
## Setup

### Requirement
Required python (version 2.7) libraries: Tensorflow (>=1.1) + Scipy + Numpy + Rawpy.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes but not tested.

### Dateset
We provide the dataset by Sony and Fuji cameras. To download the data, you can run
```Shell
python download_dataset.py
```
or you can download it directly from Google drive for the [Sony](https://drive.google.com/open?id=1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx)  and [Fuji](https://drive.google.com/open?id=1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH) sets. 

The file lists are provided. In each row, there are a short-exposed image path, the corresponding long-exposed image path, camera ISO and F number. Note that multiple short-exposed images may correspond to the same long-exposed image. 

The file name contains the image information. For example, in "10019_00_0.033s.RAF", the first digit "1" means it is from the test set ("0" for training set and "2" for valiadation set); "0019" is the image ID; the following "00" is the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.  


### Testing
1. Clone this repository.
2. Download the pretrained models by running
```Shell
python download_models.py
```
3. Run "python test_Sony.py". This will generate results on the Sony test set.
4. Run "python test_Fuji.py". This will generate results on the Fuji test set.

By default, the code takes the data in the "./dataset/Sony/" folder and "./dataset/Fuji/". If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code. 

### Training
1. To train the Sony model, run "python train_Sony.py".
2. To train the Fuji model, run "python train_Fuji.py".

By default, the code takes the data in the "./dataset/Sony/" folder and "./dataset/Fuji/". If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code.

The training and testing code has been cleaned to be as simple as possible. Loading the raw data and proccesing by Rawpy takes significant more time than the backpropagation. By default, the code will load all the training data into memory. This requires at least 64GB RAM for training the Sony model and 128GB RAM for the Fuji model. We will provide RAM friendly code and processed images soon.

## Questions
If you have questions about the code and data, please email to cchen156@illinois.edu.

## Citation
If you use our code and dataset for research, please cite our paper:

Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.

### License
MIT License.
