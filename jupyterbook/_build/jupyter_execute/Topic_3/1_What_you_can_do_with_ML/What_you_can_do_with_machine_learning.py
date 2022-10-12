#!/usr/bin/env python
# coding: utf-8

# # What You Can Do With Machine Learning

# ## Computer Vision and Graphics

# ### Image Classifiers

# The most common machine learning task is to take an image and classify if a single object is in an image

# ```{admonition} Benchmark Datasets
# [MNIST](http://yann.lecun.com/exdb/mnist/) - The MNIST database of handwritten digits. 
# 
# [CAL 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) - Caltech-101 consists of pictures of objects belonging to 101 classes, plus one background clutter class. Each image is labelled with a single object. Each class contains roughly 40 to 800 images, totalling around 9k images. Images are of variable sizes, with typical edge lengths of 200-300 pixels. This version contains image-level labels only. The original dataset also contains bounding boxes.
# 
# [ImageNet](https://www.image-net.org) -ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer vision and deep learning research. The data is available for free to researchers for non-commercial use.
# 
# [CFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html) - The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
# ```

# ![](./figs/Classification.png)
# 
# from Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NeuroIPS, 2012.

# * Microsoft (Deep Residual Learning) [[Paper](http://arxiv.org/pdf/1512.03385v1.pdf)][[Slide](http://image-net.org/challenges/talks/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)]
#   * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition, arXiv:1512.03385.
# * Microsoft (PReLu/Weight Initialization) [[Paper]](http://arxiv.org/pdf/1502.01852)
#   * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, arXiv:1502.01852.
# * Batch Normalization [[Paper]](http://arxiv.org/pdf/1502.03167)
#   * Sergey Ioffe, Christian Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, arXiv:1502.03167.
# * GoogLeNet [[Paper]](http://arxiv.org/pdf/1409.4842)
#   * Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, CVPR, 2015.
# * VGG-Net [[Web]](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [[Paper]](http://arxiv.org/pdf/1409.1556)
#   * Karen Simonyan and Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Visual Recognition, ICLR, 2015.
# * AlexNet [[Paper]](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012)
#   * Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NIPS, 2012.

# ### Classification Model Benchmarks

# ![](./figs/imagenet_benchmarks.jpg)
# [ImageNet Leaderboard](https://paperswithcode.com/sota/image-classification-on-imagenet)

# ### Object Detection
# Models that try to find and detect the location of multiple objects in an image

# ![](./figs/object_detection.png)
# 
# from Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv:1506.01497

# ```{admonition} Benchmark Datasets
# [COCO](http://cocodataset.org/#home) - COCO is a large-scale object detection, segmentation, and captioning dataset.
#  
# 
# [open_images_v4](https://storage.googleapis.com/openimages/web/index.html) - Open Images is a dataset of ~9M images that have been annotated with image-level labels and object bounding boxes.
# 
# The training set of V4 contains 14.6M bounding boxes for 600 object classes on 1.74M images, making it the largest existing dataset with object location annotations. The boxes have been largely manually drawn by professional annotators to ensure accuracy and consistency. The images are very diverse and often contain complex scenes with several objects (8.4 per image on average). Moreover, the dataset is annotated with image-level labels spanning thousands of classes.
# ```

# * PVANET [[Paper]](https://arxiv.org/pdf/1608.08021) [[Code]](https://github.com/sanghoon/pva-faster-rcnn)
#   * Kye-Hyeon Kim, Sanghoon Hong, Byungseok Roh, Yeongjae Cheon, Minje Park, PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection, arXiv:1608.08021
# * OverFeat, NYU [[Paper]](http://arxiv.org/pdf/1312.6229.pdf)
#   * OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks, ICLR, 2014.
# * R-CNN, UC Berkeley [[Paper-CVPR14]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) [[Paper-arXiv14]](http://arxiv.org/pdf/1311.2524)
#   * Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik, Rich feature hierarchies for accurate object detection and semantic segmentation, CVPR, 2014.
# * SPP, Microsoft Research [[Paper]](http://arxiv.org/pdf/1406.4729)
#   * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, ECCV, 2014.
# * Fast R-CNN, Microsoft Research [[Paper]](http://arxiv.org/pdf/1504.08083)
#   * Ross Girshick, Fast R-CNN, arXiv:1504.08083.
# * Faster R-CNN, Microsoft Research [[Paper]](http://arxiv.org/pdf/1506.01497)
#   * Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv:1506.01497.
# * R-CNN minus R, Oxford [[Paper]](http://arxiv.org/pdf/1506.06981)
#   * Karel Lenc, Andrea Vedaldi, R-CNN minus R, arXiv:1506.06981.
# * End-to-end people detection in crowded scenes [[Paper]](http://arxiv.org/abs/1506.04878)
#   * Russell Stewart, Mykhaylo Andriluka, End-to-end people detection in crowded scenes, arXiv:1506.04878.
# * You Only Look Once: Unified, Real-Time Object Detection [[Paper]](http://arxiv.org/abs/1506.02640), [[Paper Version 2]](https://arxiv.org/abs/1612.08242), [[C Code]](https://github.com/pjreddie/darknet), [[Tensorflow Code]](https://github.com/thtrieu/darkflow)
#   * Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, You Only Look Once: Unified, Real-Time Object Detection, arXiv:1506.02640
#   * Joseph Redmon, Ali Farhadi (Version 2)
# * Inside-Outside Net [[Paper]](http://arxiv.org/abs/1512.04143)
#   * Sean Bell, C. Lawrence Zitnick, Kavita Bala, Ross Girshick, Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
# * Deep Residual Network (Current State-of-the-Art) [[Paper]](http://arxiv.org/abs/1512.03385)
#   * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition
# * Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning [[Paper](http://arxiv.org/pdf/1503.00949.pdf)]
# * R-FCN [[Paper]](https://arxiv.org/abs/1605.06409) [[Code]](https://github.com/daijifeng001/R-FCN)
#   * Jifeng Dai, Yi Li, Kaiming He, Jian Sun, R-FCN: Object Detection via Region-based Fully Convolutional Networks
# * SSD [[Paper]](https://arxiv.org/pdf/1512.02325v2.pdf) [[Code]](https://github.com/weiliu89/caffe/tree/ssd)
#   * Wei Liu1, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, SSD: Single Shot MultiBox Detector, arXiv:1512.02325
# * Speed/accuracy trade-offs for modern convolutional object detectors [[Paper]](https://arxiv.org/pdf/1611.10012v1.pdf)
#   * Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin Murphy, Google Research, arXiv:1611.10012

# ### Semantic Segmentation
# The task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category

# ![](./figs/semantic_segmentation.png)
# 
# from Jifeng Dai, Kaiming He, Jian Sun, BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation, arXiv:1503.01640.

# ```{admonition} Benchmark Datasets
# 
# [PASCAL VOC (PASCAL Visual Object Classes Challenge)](http://host.robots.ox.ac.uk/pascal/VOC/) - The PASCAL Visual Object Classes (VOC) 2012 dataset contains 20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person. Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations. This dataset has been widely used as a benchmark for object detection, semantic segmentation, and classification tasks. The PASCAL VOC dataset is split into three subsets: 1,464 images for training, 1,449 images for validation and a private testing set.
# 
# ```

# * SEC: Seed, Expand and Constrain
#   *  Alexander Kolesnikov, Christoph Lampert, Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation, ECCV, 2016. [[Paper]](http://pub.ist.ac.at/~akolesnikov/files/ECCV2016/main.pdf) [[Code]](https://github.com/kolesman/SEC)
# * Adelaide
#   * Guosheng Lin, Chunhua Shen, Ian Reid, Anton van dan Hengel, Efficient piecewise training of deep structured models for semantic segmentation, arXiv:1504.01013. [[Paper]](http://arxiv.org/pdf/1504.01013) (1st ranked in VOC2012)
#   * Guosheng Lin, Chunhua Shen, Ian Reid, Anton van den Hengel, Deeply Learning the Messages in Message Passing Inference, arXiv:1508.02108. [[Paper]](http://arxiv.org/pdf/1506.02108) (4th ranked in VOC2012)
# * Deep Parsing Network (DPN)
#   * Ziwei Liu, Xiaoxiao Li, Ping Luo, Chen Change Loy, Xiaoou Tang, Semantic Image Segmentation via Deep Parsing Network, arXiv:1509.02634 / ICCV 2015 [[Paper]](http://arxiv.org/pdf/1509.02634.pdf) (2nd ranked in VOC 2012)
# * CentraleSuperBoundaries, INRIA [[Paper]](http://arxiv.org/pdf/1511.07386)
#   * Iasonas Kokkinos, Surpassing Humans in Boundary Detection using Deep Learning, arXiv:1411.07386 (4th ranked in VOC 2012)
# * BoxSup [[Paper]](http://arxiv.org/pdf/1503.01640)
#   * Jifeng Dai, Kaiming He, Jian Sun, BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation, arXiv:1503.01640. (6th ranked in VOC2012)
# * POSTECH
#   * Hyeonwoo Noh, Seunghoon Hong, Bohyung Han, Learning Deconvolution Network for Semantic Segmentation, arXiv:1505.04366. [[Paper]](http://arxiv.org/pdf/1505.04366) (7th ranked in VOC2012)
#   * Seunghoon Hong, Hyeonwoo Noh, Bohyung Han, Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation, arXiv:1506.04924. [[Paper]](http://arxiv.org/pdf/1506.04924)
#   * Seunghoon Hong,Junhyuk Oh,	Bohyung Han, and	Honglak Lee, Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network, arXiv:1512.07928 [[Paper](http://arxiv.org/pdf/1512.07928.pdf)] [[Project Page](http://cvlab.postech.ac.kr/research/transfernet/)]
# * Conditional Random Fields as Recurrent Neural Networks [[Paper]](http://arxiv.org/pdf/1502.03240)
#   * Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, Philip H. S. Torr, Conditional Random Fields as Recurrent Neural Networks, arXiv:1502.03240. (8th ranked in VOC2012)
# * DeepLab
#   * Liang-Chieh Chen, George Papandreou, Kevin Murphy, Alan L. Yuille, Weakly-and semi-supervised learning of a DCNN for semantic image segmentation, arXiv:1502.02734. [[Paper]](http://arxiv.org/pdf/1502.02734) (9th ranked in VOC2012)
# * Zoom-out [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
#   * Mohammadreza Mostajabi, Payman Yadollahpour, Gregory Shakhnarovich, Feedforward Semantic Segmentation With Zoom-Out Features, CVPR, 2015
# * Joint Calibration [[Paper]](http://arxiv.org/pdf/1507.01581)
#   * Holger Caesar, Jasper Uijlings, Vittorio Ferrari, Joint Calibration for Semantic Segmentation, arXiv:1507.01581.
# * Fully Convolutional Networks for Semantic Segmentation [[Paper-CVPR15]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1411.4038)
#   * Jonathan Long, Evan Shelhamer, Trevor Darrell, Fully Convolutional Networks for Semantic Segmentation, CVPR, 2015.
# * Hypercolumn [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hariharan_Hypercolumns_for_Object_2015_CVPR_paper.pdf)
#   * Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik, Hypercolumns for Object Segmentation and Fine-Grained Localization, CVPR, 2015.
# * Deep Hierarchical Parsing
#   * Abhishek Sharma, Oncel Tuzel, David W. Jacobs, Deep Hierarchical Parsing for Semantic Segmentation, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sharma_Deep_Hierarchical_Parsing_2015_CVPR_paper.pdf)
# * Learning Hierarchical Features for Scene Labeling [[Paper-ICML12]](http://yann.lecun.com/exdb/publis/pdf/farabet-icml-12.pdf) [[Paper-PAMI13]](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
#   * Clement Farabet, Camille Couprie, Laurent Najman, Yann LeCun, Scene Parsing with Multiscale Feature Learning, Purity Trees, and Optimal Covers, ICML, 2012.
#   * Clement Farabet, Camille Couprie, Laurent Najman, Yann LeCun, Learning Hierarchical Features for Scene Labeling, PAMI, 2013.
# * University of Cambridge [[Web]](http://mi.eng.cam.ac.uk/projects/segnet/)
#   * Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015. [[Paper]](http://arxiv.org/abs/1511.00561)
# * Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015. [[Paper]](http://arxiv.org/abs/1511.00561)
# * Princeton
#   * Fisher Yu, Vladlen Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.07122v2.pdf)]
# * Univ. of Washington, Allen AI
#   * Hamid Izadinia, Fereshteh Sadeghi, Santosh Kumar Divvala, Yejin Choi, Ali Farhadi, "Segment-Phrase Table for Semantic Segmentation, Visual Entailment and Paraphrasing", ICCV, 2015, [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Izadinia_Segment-Phrase_Table_for_ICCV_2015_paper.pdf)]
# * INRIA
#   * Iasonas Kokkinos, "Pusing the Boundaries of Boundary Detection Using deep Learning", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.07386v2.pdf)]
# * UCSB
#   * Niloufar Pourian, S. Karthikeyan, and B.S. Manjunath, "Weakly supervised graph based semantic segmentation by learning communities of image-parts", ICCV, 2015, [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Pourian_Weakly_Supervised_Graph_ICCV_2015_paper.pdf)]

# ### Edge Detection

# Taking an image and identifying the edges

# ![](./figs/edge_detection_example.png)
# 
# from Gedas Bertasius, Jianbo Shi, Lorenzo Torresani, DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection, CVPR, 2015.

# ```{admonition} Benchmark Datasets
# 
# [The multi-cue boundary detection dataset](https://serre-lab.clps.brown.edu/resource/multicue/) - We collected two sets of hand-annotations for the last video frame of the left image for every scene: one for object boundaries, and one for “lower-level” edges. Hand-segmentation was performed by paid undergraduate students at Brown University (Providence, RI). We wrote custom custom Java software to enable manual annotations within a web browser. Annotators were not limited in the amount of time they had available to complete the task. The segmentation involved annotating contours that defined the boundary of each object’s visible surface regions. We gave all annotators the same basic instructions as done in Martin, Fowlkes, and Malik (2004): “You will be presented a photographic image. Divide the image into some number of segments, where the segments represent things or parts of things in the scene. The number of segments is up to you, as it depends on the image. Something between 2 and 30 is likely to be appropriate. It is important that all of the segments have approximately equal importance.” 
# 
# ```

# * Holistically-Nested Edge Detection [[Paper]](http://arxiv.org/pdf/1504.06375) [[Code]](https://github.com/s9xie/hed)
#   * Saining Xie, Zhuowen Tu, Holistically-Nested Edge Detection, arXiv:1504.06375.
# * DeepEdge [[Paper]](http://arxiv.org/pdf/1412.1123)
#   * Gedas Bertasius, Jianbo Shi, Lorenzo Torresani, DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection, CVPR, 2015.
# * DeepContour [[Paper]](http://mc.eistar.net/UpLoadFiles/Papers/DeepContour_cvpr15.pdf)
#   * Wei Shen, Xinggang Wang, Yan Wang, Xiang Bai, Zhijiang Zhang, DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection, CVPR, 2015.
# 

# ### Colorization
# Taking an image or strokes of color and converting it into a colored image

# ![](./figs/Colorization.png)

# ```{admonition} More Information
# 
# [Awesome Image Colorization](https://github.com/MarkMoHR/Awesome-Image-Colorization)
# 
# ```

# ### Super-resolution
# Taking a low-quality image and enhancing its resolution

# ![](./figs/Image_superresolution.png)
# 
# {cite:p}`Dong2016-ad`

# ```{admonition} Benchmark Datasets
# 
# [BSD (Berkeley Segmentation Dataset)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) - BSD is a dataset used frequently for image denoising and super-resolution. Of the subdatasets, BSD100 is aclassical image dataset having 100 test images proposed by Martin et al.. The dataset is composed of a large variety of images ranging from natural images to object-specific such as plants, people, food etc. BSD100 is the testing set of the Berkeley segmentation dataset BSD300.
# 
# ```

# ### Image and Video Denoising
# Techniques to remove noise from images and videos

# ![](./figs/Denoising.jpg)
# 
# {cite}`Xu2020-ny`

# ```{admonition} Benchmark Datasets
# 
# [The Darmstadt Noise Dataset](https://noise.visinf.tu-darmstadt.de) - Lacking realistic ground truth data, image denoising techniques are traditionally evaluated on images corrupted by synthesized i. i. d. Gaussian noise.  This is quite problematic, since noise in real photographs is not i. i. d. Gaussian and even seemingly minor details of the synthetic noise process, such as whether the noisy values are rounded to integers, can have a significant effect on the relative performance of methods.
# 
# Hence, we present a novel denoising benchmark, the Darmstadt Noise Dataset (DND). It consists of 50 pairs of real noisy images and corresponding ground truth images that were captured with consumer grade cameras of differing sensor sizes. For each pair, a reference image is taken with the base ISO level while the noisy image is taken with higher ISO and appropriately adjusted exposure time. The reference image undergoes a careful post-processing entailing small camera shift adjustment, linear intensity scaling and removal of low-frequency bias. The post-processed image serves as ground truth for our denoising benchmark.
# 
# ```

# ```{admonition} More Information
# 
# [Awesome Denoise](https://github.com/oneTaken/Awesome-Denoise)
# 
# ```

# ### Optical flows

# ![](./figs/optical_flows.jpg)

# {cite:p}`Lagemann2021-oq`

# ```{admonition} Benchmark Datasets
# [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) -Virtual KITTI is a photo-realistic synthetic video dataset designed to learn and evaluate computer vision models for several video understanding tasks: object detection and multi-object tracking, scene-level and instance-level semantic segmentation, optical flow, and depth estimation.
# 
# Virtual KITTI contains 50 high-resolution monocular videos (21,260 frames) generated from five different virtual worlds in urban settings under different imaging and weather conditions. These worlds were created using the Unity game engine and a novel real-to-virtual cloning method. These photo-realistic synthetic videos are automatically, exactly, and fully annotated for 2D and 3D multi-object tracking and at the pixel level with category, instance, flow, and depth labels
#  ```

# ### Human Pose Estimation
# Predicting the pose of a human in an image 

# ![](./figs/pose_estimation_figure.png)

# ```{admonition} Benchmark Datasets
# [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#) - MPII Human Pose dataset is a state of the art benchmark for evaluation of articulated human pose estimation. The dataset includes around 25K images containing over 40K people with annotated body joints. The images were systematically collected using an established taxonomy of every day human activities. Overall the dataset covers 410 human activities and each image is provided with an activity label. Each image was extracted from a YouTube video and provided with preceding and following un-annotated frames. In addition, for the test set we obtained richer annotations including body part occlusions and 3D torso and head orientations.
# 
# Following the best practices for the performance evaluation benchmarks in the literature we withhold the test annotations to prevent overfitting and tuning on the test set. We are working on an automatic evaluation server and performance analysis tools based on rich test set annotations.
#  ```

# - [Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation](https://arxiv.org/abs/1911.10529) -[[CODE]](https://github.com/jialee93/Improved-Body-Parts) - Jia Li, Wen Su, Zengfu Wang (AAAI2020)
# - [An End-to-End Framework for Unsupervised Pose Estimation of Occluded Pedestrians](https://arxiv.org/abs/2002.06429)   - Sudip Das, Perla Sai Raj Kishore, Ujjwal Bhattacharya (Arxiv 2020)
# - [Transferring Dense Pose to Proximal Animal Classes](https://arxiv.org/abs/2003.00080) - [[CODE]](https://asanakoy.github.io/densepose-evolution/)  - Artsiom Sanakoyeu, Vasil Khalidov, Maureen S. McCarthy, Andrea Vedaldi, Natalia Neverova (CVPR 2020)
# - [Peeking into occluded joints: A novel framework for crowd pose estimation](https://arxiv.org/abs/2003.10506)   - ingteng Qiu, Xuanye Zhang, Yanran Li, Guanbin Li, Xiaojun Wu, Zixiang Xiong, Xiaoguang Han, Shuguang Cui (Arxiv 2020)
# - [Motion-supervised Co-Part Segmentation](https://arxiv.org/abs/2004.03234)   - Aliaksandr Siarohin*, Subhankar Roy*, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, Nicu Sebe (Arxiv 2020)
# - [Detailed 2D-3D Joint Representation for Human-Object Interaction](https://arxiv.org/abs/2004.08154) - [[CODE]](https://github.com/DirtyHarryLYL/DJ-RN)  - Yong-Lu Li, Xinpeng Liu, Han Lu, Shiyi Wang, Junqi Liu, Jiefeng Li, Cewu Lu (CVPR 2020)
# - [Distribution Aware Coordinate Representation for Human Pose Estimation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.pdf) - [[CODE]](https://github.com/ilovepose/DarkPose)  - Feng Zhang, Xiatian Zhu, Hanbin Dai, Mao Ye, Ce Zhu (CVPR 2020)
# - [Yoga-82: A New Dataset for Fine-grained Classification of Human Poses](https://arxiv.org/abs/2004.10362) - [[Data]](https://sites.google.com/view/yoga-82/home)  - Manisha Verma, Sudhakar Kumawat, Yuta Nakashima, Shanmuganathan Raman (CVPRW 2020)
# - [Self-supervised Keypoint Correspondences for Multi-Person Pose Estimation and Tracking in Videos
# ](https://arxiv.org/abs/2004.12652)   - Rafi Umer, Andreas Doering, Bastian Leibe, Juergen Gall (Arxiv 2020)
# - [Making DensePose fast and light](https://arxiv.org/abs/2006.15190)   (Arxiv 2020)
# - [Differentiable Hierarchical Graph Grouping for Multi-Person Pose Estimation](https://arxiv.org/pdf/2007.11864) - Sheng Jin, Wentao Liu, Enze Xie, Wenhai Wang, Chen Qian, Wanli Ouyang, Ping Luo (ECCV 2020)
# - [Whole-Body Human Pose Estimation in the Wild](https://arxiv.org/abs/2007.11858) - [[Data]](https://github.com/jin-s13/COCO-WholeBody)  - Sheng Jin, Lumin Xu, Jin Xu, Can Wang, Wentao Liu, Chen Qian, Wanli Ouyang, Ping Luo (ECCV 2020)

# ### 6D Object Pose Estimation
# Models to determine the location and orientation of objects from an image important for robotics

# ![](./figs/centersnap_reconstruction_6d)

# {cite:p}`Irshad2022-hz`

# ```{admonition} Benchmark Datasets
# [ICCV2015 Occluded Object Challenge](https://hci.iwr.uni-heidelberg.de/vislearn/iccv2015-occlusion-challenge/#Dataset) - The purpose of this challenge is to compare different methods for object pose estimation in a realistic setting featuring heavy occlusion. Our dataset includes eight objects in a cluttered scene. Given a RGB-D image the method has to estimate the position and orientation (a total of six degrees of freedom) of each object. You can participate by applying your method to our data and submitting your results. We will evaluate submitted results according to multiple metrics and display the scores for comparison.
#  ```

# ## Natural Language Processing
# 
# a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them.
# 
# *Source Wikipedia*

# ### Extracting Knowledge from Text Corpus

# ![](./figs/Materials_Science_NLP.webp)
# 
# {cite}`Tshitoyan2019-zt`

# ```{admonition} Benchmark Datasets
# [GLUE (General Language Understanding Evaluation benchmark)](https://gluebenchmark.com/) - General Language Understanding Evaluation (GLUE) benchmark is a collection of nine natural language understanding tasks, including single-sentence tasks CoLA and SST-2, similarity and paraphrasing tasks MRPC, STS-B and QQP, and natural language inference tasks MNLI, QNLI, RTE and WNLI.
# 
# [SQuAD (Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/) - The Stanford Question Answering Dataset (SQuAD) is a collection of question-answer pairs derived from Wikipedia articles. In SQuAD, the correct answers of questions can be any sequence of tokens in the given text. Because the questions and answers are produced by humans through crowdsourcing, it is more diverse than some other question-answering datasets. SQuAD 1.1 contains 107,785 question-answer pairs on 536 articles. SQuAD2.0 (open-domain SQuAD, SQuAD-Open), the latest version, combines the 100,000 questions in SQuAD1.1 with over 50,000 un-answerable questions written adversarially by crowdworkers in forms that are similar to the answerable ones.
# 
# [SST (Stanford Sentiment Treebank)](https://nlp.stanford.edu/sentiment) - The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.
# 
# Each phrase is labelled as either negative, somewhat negative, neutral, somewhat positive or positive. The corpus with all 5 labels is referred to as SST-5 or SST fine-grained. Binary classification experiments on full sentences (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) refer to the dataset as SST-2 or SST binary.
# ```

# ### Speech Recognition
# Learning context representations and information from audio files

# ![](./figs/wav2vec2.png)
# 
# {cite}`Baevski_undated-gm`

# ```{admonition} Benchmark Datasets
# [LibriSpeech](http://www.openslr.org/12) - The LibriSpeech corpus is a collection of approximately 1,000 hours of audiobooks that are a part of the LibriVox project. Most of the audiobooks come from the Project Gutenberg. The training data is split into 3 partitions of 100hr, 360hr, and 500hr sets while the dev and test data are split into the ’clean’ and ’other’ categories, respectively, depending upon how well or challening Automatic Speech Recognition systems would perform against. Each of the dev and test sets is around 5hr in audio length. This corpus also provides the n-gram language models and the corresponding texts excerpted from the Project Gutenberg books, which contain 803M tokens and 977K unique words.
# 
# [AudioSet](https://research.google.com/audioset/index.html) - Audioset is an audio event dataset, which consists of over 2M human-annotated 10-second video clips. These clips are collected from YouTube, therefore many of which are in poor-quality and contain multiple sound-sources. A hierarchical ontology of 632 event classes is employed to annotate these data, which means that the same sound could be annotated as different labels. For example, the sound of barking is annotated as Animal, Pets, and Dog. All the videos are split into Evaluation/Balanced-Train/Unbalanced-Train set.
# 
# ```

# ## Generative Models
# Generative models generate new examples from a distribution

# ![](./figs/generative_v_discriminative.png)

# In deep learning these models are called Generative Adversarial Networks (GANs) $\rightarrow$ there are a ton of cool applications

# ### Generating Anime Characters
# 
# ![](./figs/Generated_anime.jpg)
# 
# {cite:p}`Jin2017-gy`

# ### Generating Images from Text
# 
# ![](./figs/text_to_image.jpg)
# 
# {cite}`Dash2017-kw`

# In[1]:


from IPython.display import IFrame

IFrame("http://gaugan.org/gaugan2/", width=2200, height=1200)


# ### Image-to-Image Translation

# ![](./figs/cyclegans.jpeg)
# 
# {cite:p}`Zhu2017-zi`

# ### DeepFakes

# Synthetic media where a person is replaced by the likeness (image and voice of another person)

# In[2]:


from IPython.display import HTML

HTML(
    '<iframe width="800" height="450" src="https://www.youtube.com/embed/cQ54GDm1eL0" title="You Won’t Believe What Obama Says In This Video! 😉" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
) 


# ![](./figs/Faceswap.jpeg)
# 
# from [https://github.com/shaoanlu/faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN)

# ## Physics Informed and Physics Constrained Machine Learning

# ### Physics-Informed Neural Networks (PINNs)

# ![](./figs/PINNS.webp)
# 
# {cite}`Cai2022-ex`

# ### Accelerated Fitting Using Physics-Constrained Neural Networks
# 
# As long as the empirical functions are differentiable you can train a model to predict the parameters using the empirical function as a decoder. 

# #### Model Architecture
# 
# ![](./figs/AE_image.svg)

# #### Fit Results
# 
# ![](./figs/Fit_results.svg)

# ### Learning Underlying Governing Equations

# There are ways to take raw data and candidate functions and learn underlying governing equations using sparse identification

# ![](./figs/lorenz.jpeg)
# 
# {cite:p}`Brunton2016-ej`

# ## Reinforcement Learning

# ### Playing Mario

# In[3]:


from IPython.display import HTML

HTML(
    '<iframe width="800" height="450" src="https://www.youtube.com/embed/qv6UVOQ0F44" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
)


# ### Much More Complex Games

# In[4]:


from IPython.display import HTML

HTML(
    '<iframe width="800" height="450" src="https://www.youtube.com/embed/UuhECwm31dM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
)


# ### Physical Object Manipulation or Fine Motor Skills

# In[5]:


from IPython.display import HTML

HTML(
    '<iframe width="800" height="450" src="https://www.youtube.com/embed/x4O8pojMF0w" title="Solving Rubik’s Cube with a Robot Hand" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
)


# ## Take Away Messages:

# * Data analysis and machine learning can be used in a variety of complex tasks

# * There are a variety of different implementations and methods for in machine learning

# * We have only just scratched the surface in how machine learning can be applied, this is a very exciting time

# * In this class we will prepare you to be a machine learning practitioner, and use machine learning for applications in science and manufacturing

# * Machine learning is a rapidly growing field. You could take a full course in each of these areas

# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
