# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN) on Kitti and Citiscapes datasets.

### Architecture
#### Kitti
![alt text](image.png)

#### Citiscapes
``` mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'fontSize': '25px'}, 'config': {'width': 1200}}}%%
graph LR
    %% ========== INPUT ==========
    A[("INPUT IMAGE\n(256px × 512px × 3 channels)")]:::input
    --> B[["VGG-16 Backbone"]]:::vgg
    
    %% ========== VGG LAYERS ==========
    subgraph VGG-16 Feature Extraction.
    B --> C["layer3_out\n(32×64×256)"]:::vgglayer
    B --> D["layer4_out\n(16×32×512)"]:::vgglayer
    B --> E["layer7_out\n(8×16×4096)"]:::vgglayer
    end
    
    %% ========== DECODER PATH ==========
    E --> F["1×1 Convolution\n↓\n8×16×3"]:::conv
    F --> G["Transposed Conv\n4×4 kernel\nstride=2\n↑ to 16×32×3"]:::conv
    
    D --> H["1×1 Convolution\n↓\n16×32×3"]:::conv
    G --> I{{"Add"}}:::add
    H --> I
    
    I --> J["Transposed Conv\n4×4 kernel\nstride=2\n↑ to 32×64×3"]:::conv
    
    C --> K["1×1 Convolution\n↓\n32×64×3"]:::conv
    J --> L{{"Add"}}:::add
    K --> L
    
    L --> M["Final Transposed Conv\n16×16 kernel\nstride=8\n↑ to 256×512×3"]:::conv
    M --> N[("OUTPUT MASK\n(256×512×3 classes)")]:::output

    %% ========== STYLING ==========
    classDef input fill:#FFF2CC,stroke:#D6B656,stroke-width:3px,font-size:18px
    classDef output fill:#D5E8D4,stroke:#82B366,stroke-width:3px,font-size:18px
    classDef vgg fill:#E1D5E7,stroke:#9673A6,stroke-width:2px
    classDef vgglayer fill:#F5F5F5,stroke:#999,stroke-dasharray:5
    classDef conv fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,font-size:16px
    classDef add fill:#FFF,stroke:#1ABC9C,stroke-width:3px,color:#000
    
    %% ========== LINK STYLES ==========
    linkStyle default stroke:#666,stroke-width:3px
    linkStyle 0,7,13 stroke-width:4px,stroke:#333
```

#### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Start
##### Run
Run the following command to run the project:
```
python main.py
```

`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Results
#### Kitti (50 epochs)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)

All road segmented images for Kitti are located [here](https://github.com/sachink20aug/CarND-Semantic-Segmentation/tree/master/runs/1542165778.83792)


#### Citiscapes
[Link](https://github.com/sachink20aug/CarND-Semantic-Segmentation/blob/master/cityscapes_result.gif)

### Setup

##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Kitti Road Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Citiscapes Road Dataset
A subset of citiscape [dataset](https://www.cityscapes-dataset.com/) has been chosen. Unfortunately, according to the Cityscapes dataset licence, I can not publish all produced images.
