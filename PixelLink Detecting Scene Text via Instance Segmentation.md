# PixelLink: Detecting Scene Text via Instance Segmentation

acquisition	:	취득, 획득, 인수

indispensable	:	없어서는 안될

requiring	:	요구

iterations	:	반복

novel 	:	새로운, 신기한

 benchmarks	:	기준, 기준점, 수준 기표

## Abstract

##### Most state-of-the-art scene text detection algorithms are deep learning based methods that depend on bounding box regression and perform at least two kinds of predictions: text/nontext classification and location regression. 

 state-of-the-art text detection algorithms은 text/nontext classification and location regression을 prediction 하는 bounding box regression 기반 딥러닝 algorithm이다. 

##### Regression plays a key role in the acquisition of bounding boxes in these methods, but it is not indispensable because text/non-text prediction can also be considered as a kind of semantic segmentation that contains full location information in itself. 

 Regression은 위와 같은 방법에서 bounding boxes를 획득하는데 중요한 역할을 한다. 그러나 이것은 꼭 필요한 것은 아니다. 왜냐하면 text/non-text prediction은 글자에 대한 모든 위치정보를 가지고 있는 semantic segmentation으로 생각해볼 수 있기 때문이다.

##### However, text instances in scene images often lie very close to each other, making them very difficult to separate via semantic segmentation. 

 그러나 이미지에서 text instances는 종종 매우 가까이 놓여져있어, semantic segmentation을 통해 그들을 분리하는 것은 매우 어렵다.

##### Therefore, instance segmentation is needed to address this problem. 

그러므로 instance segmentation이 이러한 문제를 해결하기 위해 필요하다.

##### In this paper, PixelLink, a novel scene text detection algorithm based on instance segmentation, is proposed. 

 이 논문에선 instatnce segmentation기반의 novel scene text detection algorithm인 PixelLink가 제안된다.

##### Text instances are first segmented out by linking pixels within the same instance together. 

Text instances는 처음, linking pixels에 의해 same instance끼리 세분화된다.

##### Text bounding boxes are then extracted directly from the segmentation result without location regression. 

바로 다음, segmentation result로부터 location regression없이  Text bounding boxes가 추출된다.

##### Experiments show that, compared with regression-based methods, PixelLink can achieve better or comparable performance on several benchmarks, while requiring many fewer training iterations and less training data.

실험은 regression-based methods들과 비교하여, PixelLink는 더 적은 데이터와 반복으로도 더 좋은 성과를 이루는 것을 보여준다.



robust	:	건장한, 억센, 독한

efficient	:	실력있는, 효과가 있는, 능률적인

confidence	:	자신, 신임, 비밀, 털어놓음

offsets	:	오프셋, 갈라짐, 출발

linkage	:	결합

fomulation	:	공식화

pixel-wise	:	픽셀 단위

scratch	:	할퀴다, 긁다, 긁어 모으다

## Introduction

##### Reading text in the wild, or robust reading has drawn great interest for a long time (Ye and Doermann 2015). It is usually divided into two steps or sub-tasks: text detection and text recognition. 

##### The detection task, also called localization, takes an image as input and outputs the locations of text within it. 

localization이라고 불리는 detection task는 image를 input으로, 글자의 위치를 output으로 가진다.

##### Along with the advances in deep learning and general object detection, more and more accurate as well as efficient scene text detection algorithms have been proposed, e.g., CTPN (Tian et al. 2016), TextBoxes (Liao et al. 2017), SegLink (Shi, Bai, and Belongie 2017) and EAST (Zhou et al. 2017).

#####  Most of these state-of-the-art methods are built on Fully Convolutional Networks (Long, Shelhamer, and Darrell 2015), and perform at least two kinds of predictions:

##### 1. Text/non-text classification. Such predictions can be taken as probabilities of pixels being within text bounding boxes (Zhang et al. 2016). But they are more frequently used as confidences on regression results (e.g., TextBoxes, SegLink, EAST).

##### 2. Location regression. Locations of text instances, or their segments/slices, are predicted as offsets from reference boxes (e.g., TextBoxes, SegLink, CTPN), or absolute locations of bounding boxes (e.g., EAST).

대부분의 state-of-the-art 방법은 Fully Convolutional Networks를 가지고 있고 Text/non-text classification, Location regression의 2가지 prediction을 수행한다. Text/non-text classification은 text bounding boxes안에 있는 pixels들의 확률이다. Location regression은 absolute locations of bounding boxes을 predict 한다.

#####  In methods like SegLink, linkages between segments are also predicted. After these predictions, post-processing that mainly includes joining segments together (e.g., SegLink, CTPN) or Non-Maximum Suppression (e.g., TextBoxes, EAST), is applied to obtain bounding boxes as the final output. 

predictions 후에 non-maximum suppression과 같은 post-processing이 적용된 후, final output으로 bounding-boxes가 결과물로 나온다.

#####  Location regression has long been used in object detection, as well as in text detection, and has proven to be effective. It plays a key role in the formulation of text bounding boxes in state-of-the-art methods.

#####   However, as mentioned above, text/non-text predictions can not only be used as the confidences on regression results, but also as a segmentation score map, which contains location information in itself and can be used to obtain bounding boxes directly. Therefore, regression is not indispensable. 

text/non-text predictions는 confidences on regression results로 쓰일 뿐 아니라, location information in itself 에 대한 정보를 포함하는 segmentation score map으로도 쓰일 수 있다. 이러한 사실은 regression이 없이도 bounding boxes를 직접적으로 구할 수 있다.

#####  However, as shown in Fig. 1, text instances in scene images usually lie very close to each other. In such cases, they are very difficult, and are sometimes even impossible to separate via semantic segmentation (i.e., text/non-text prediction) only; therefore, segmentation at the instance level is further required. To solve this problem, a novel scene text detection algorithm, PixelLink, is proposed in this paper. It extracts text locations directly from an instance segmentation result, instead of from bounding box regression. 

semantic segmentation으로는 붙어있는 글자들을 분리하는게 힘들다. 그래서 instance segmentation 방식이 필요하다.

#####  In PixelLink, a Deep Neural Network (DNN) is trained to do two kinds of pixel-wise predictions, text/non-text prediction, and link prediction. 

PixelLink에서 Deep Neural Network는 pixel단위의 text/non-text prediction과 link prediction을 하기 위해 학습된다.

#####  Pixels within text instances are labeled as positive (i.e. text pixels), and otherwise are labeled as negative (i.e., nontext pixels). 

text instances안에 있는 pixels는 positive(text pixels)로, 다른 것은은 negative(nontext pixels)로 라벨링된다.

#####  The concept of link here is inspired by the link design in SegLink, but with significant difference. Every pixel has 8 neighbors. For a given pixel and one of its neighbors, if they lie within the same instance, the link between them is labeled as positive, and otherwise negative. 

모든 pixel은 8개의 이웃이 있다. 만약 이웃한 pixel이 같은 instance라면, pixel들 사이의 link는 positive로, 아니라면 negative로 라벨링된다. 

#####  Predicted positive pixels are joined together into Connected Components (CC) by predicted positive links. 

predicted positive links에 의해,  Predicted positive pixels는 함께 Connected Components (CC)에 들어간다.

#####  Instance segmentation is achieved in this way, with each CC representing a detected text. Methods like minAreaRect in OpenCV (Its 2014) can be applied to obtain the bounding boxes of CCs as the final detection result. 

OpenCV에  minAreaRect를 통해 bounding boxes를 얻을 수 있다. 

#####  Our experiments demonstrate the advantages of PixelLink over state-of-the-art methods based on regression. Specifically, trained from scratch, PixelLink models can achieve comparable or better performance on several benchmarks while requiring fewer training iterations and less training data.

