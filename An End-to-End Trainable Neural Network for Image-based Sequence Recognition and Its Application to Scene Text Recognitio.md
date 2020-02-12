# An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition



integrate	:	평균값을 나타내다, 종교적 차별을 폐지하다, 전체로 합치다, 통합하다

properties	:	재산, 성질, 소유권, 소유지, 유산, 소품, 물질 고유의 성질

possess	:	소유하다, 소지하다, 보유하다, (기질, 특징을) 지니다, 갖추다

arbitrary	:	독단적인, 임의의, 멋대로인

naturally	:	당연히, 자연히, 본래, 있는 그대로, 실물 그대로

confine	:	국한시키다, 넣다, 얽매이다.

lexicon	:	어휘, 어휘 목록, 사전

## Abstract

##### Image-based sequence recognition has been a longstanding research topic in computer vision. In this paper, we investigate the problem of scene text recognition, which is among the most important and challenging tasks in image-based sequence recognition.

#####  A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed.

feature extraction, sequence modeling and transcription을 하나의 framework에 통합한 A novel neural network architecture이 제안된다.

#####  Compared with previous systems for scene text recognition, the proposed architecture possesses four distinctive properties: 

scene text recognition을 목표로하는 기존의 system과 비교하여, 제안된 architecture은 4가지 고유의 성질을 가지고 있다.

##### (1) It is `end-to-end trainable`, in contrast to most of the existing algorithms whose components are separately trained and tuned. 

대부분의 알고리즘의 구성은 개별적으로 학습되고 tuned되는 것과 반대로 제안된 알고리즘은 end-to-end trainable하다.

##### (2) It naturally handles sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. 

character segmentation이나 horizontal scale normalization을 포함하지 않고, 임의의 길이의 sequences를 자연스럽게 다룬다.

##### (3) It is not confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. 

미리 정의된 어휘에 국한되지 않고, 어휘로부터 자유로운 scene text recognition, lexicon-based scene text recognition에서 놀라운 성능을 달성했다.

##### (4) It generates an effective yet much smaller model, which is more practical for real-world application scenarios. 

더 작은 모델로 더 큰 효과를 달성했다. 이것은 real-world application scenarios에 더 실용적이다.

##### The experiments on standard benchmarks, including the IIIT-5K, Street View Text and ICDAR datasets, demonstrate the superiority of the proposed algorithm over the prior arts. Moreover, the proposed algorithm performs well in the task of image-based music score recognition, which evidently verifies the generality of it.







## 1. Introduction

##### Recently, the community has seen a strong revival of neural networks, which is mainly stimulated by the great success of deep neural network models, specifically Deep Convolutional Neural Networks (DCNN), in various vision tasks. 

##### However, majority of the recent works related to deep neural networks have devoted to detection or classification of object categories [12, 25].

#####  In this paper, we are concerned with a classic problem in computer vision: imagebased sequence recognition. 

##### In real world, a stable of visual objects, such as scene text, handwriting and musical score, tend to occur in the form of sequence, not in isolation. 

##### Unlike general object recognition, recognizing such sequence-like objects often requires the system to predict a series of object labels, instead of a single label. 

##### Therefore, recognition of such objects can be naturally cast as a sequence recognition problem.

#####  Another unique property of sequence-like objects is that their lengths may vary drastically. 

##### For instance, English words can either consist of 2 characters such as “OK” or 15 characters such as “congratulations”. Consequently, the most popular deep models like DCNN [25, 26] cannot be directly applied to sequence prediction, since DCNN models often operate on inputs and outputs with fixed dimensions, and thus are incapable of producing a variable-length label sequence.

#####  Some attempts have been made to address this problem for a specific sequence-like object (e.g. scene text). 

##### For example, the algorithms in [35, 8] firstly detect individual characters and then recognize these detected characters with DCNN models, which are trained using labeled character images.

#####  Such methods often require training a strong character detector for accurately detecting and cropping each character out from the original word image. 

##### Some other approaches (such as [22]) treat scene text recognition as an image classification problem, and assign a class label to each English word (90K words in total). 

##### It turns out a large trained model with a huge number of classes, which is difficult to be generalized to other types of sequence like objects, such as Chinese texts, musical scores, etc., because the numbers of basic combinations of such kind of sequences can be greater than 1 million.

#####  In summary, current systems based on DCNN can not be directly used for image-based sequence recognition. 

##### Recurrent neural networks (RNN) models, another important branch of the deep neural networks family, were mainly designed for handling sequences. 

##### One of the advantages of RNN is that it does not need the position of each element in a sequence object image in both training and testing. 

##### However, a preprocessing step that converts 1 an input object image into a sequence of image features, is usually essential. 

##### For example, Graves et al. [16] extract a set of geometrical or image features from handwritten texts, while Su and Lu [33] convert word images into sequential HOG features. 

##### The preprocessing step is independent of the subsequent components in the pipeline, thus the existing systems based on RNN can not be trained and optimized in an end-to-end fashion. 

##### Several conventional scene text recognition methods that are not based on neural networks also brought insightful ideas and novel representations into this field. 

##### For example, Almazan` et al. [5] and Rodriguez-Serrano et al. [30] proposed to embed word images and text strings in a common vectorial subspace, and word recognition is converted into a retrieval problem. 

##### Yao et al. [36] and Gordo et al. [14] used mid-level features for scene text recognition. 

##### Though achieved promising performance on standard benchmarks, these methods are generally outperformed by previous algorithms based on neural networks [8, 22], as well as the approach proposed in this paper. 

//

##### The main contribution of this paper is a novel neural network model, whose network architecture is specifically designed for recognizing sequence-like objects in images. 

##### The proposed neural network model is named as Convolutional Recurrent Neural Network (CRNN), since it is a combination of DCNN and RNN. 

##### For sequence-like objects, CRNN possesses several distinctive advantages over conventional neural network models: 

##### 1) It can be directly learned from sequence labels (for instance, words), requiring no detailed annotations (for instance, characters); 

##### 2) It has the same property of DCNN on learning informative representations directly from image data, requiring neither hand-craft features nor preprocessing steps, including binarization/segmentation, component localization, etc.; 

##### 3) It has the same property of RNN, being able to produce a sequence of labels; 

##### 4) It is unconstrained to the lengths of sequence-like objects, requiring only height normalization in both training and testing phases; 

##### 5) It achieves better or highly competitive performance on scene texts (word recognition) than the prior arts [23, 8];

##### 6) It contains much less parameters than a standard DCNN model, consuming less storage space.



## 2. The Proposed Network Architecture

##### The network architecture of CRNN, as shown in Fig. 1, consists of three components, including the convolutional layers, the recurrent layers, and a transcription layer, from bottom to top. 

##### At the bottom of CRNN, the convolutional layers automatically extract a feature sequence from each input image. 

##### On top of the convolutional network, a recurrent network is built for making prediction for each frame of the feature sequence, outputted by the convolutional layers. 

##### The transcription layer at the top of CRNN is adopted to translate the per-frame predictions by the recurrent layers into a label sequence. 

##### Though CRNN is composed of different kinds of network architectures (eg. CNN and RNN), it can  be jointly trained with one loss function.

![1573618772684](C:\Users\multicampus\AppData\Roaming\Typora\typora-user-images\1573618772684.png)

##### Figure 1. The network architecture. The architecture consists of three parts: 

##### 1) convolutional layers, which extract a feature sequence from the input image; 

##### 2) recurrent layers, which predict a label distribution for each frame; 

##### 3) transcription layer, which translates the per-frame predictions into the final label sequence.



invariant	:	

holistic	:	



## 2.1. Feature Sequence Extraction

##### In CRNN model, the component of convolutional layers is constructed by taking the convolutional and max-pooling layers from a standard CNN model (fully-connected layers are removed). 

##### Such component is used to extract a sequential feature representation from an input image. 

##### Before being fed into the network, all the images need to be scaled to the same height. 

##### Then a sequence of feature vectors is extracted from the feature maps produced by the component of convolutional layers, which is the input for the recurrent layers. 

##### Specifically, each feature vector of a feature sequence is generated from left to right on the feature maps by column. 

##### This means the i-th feature vector is the concatenation of the i-th columns of all the maps. 

##### The width of each column in our settings is fixed to single pixel. 

##### As the layers of convolution, max-pooling, and elementwise activation function operate on local regions, they are translation invariant. 

##### Therefore, each column of the feature maps corresponds to a rectangle region of the original im-age (termed the receptive field), and such rectangle regions are in the same order to their corresponding columns on the feature maps from left to right. 

##### As illustrated in Fig. 2, each vector in the feature sequence is associated with a receptive field, and can be considered as the image descriptor for that region.

![1573618849973](C:\Users\multicampus\AppData\Roaming\Typora\typora-user-images\1573618849973.png)

##### Figure 2. The receptive field. 

##### Each vector in the extracted feature sequence is associated with a receptive field on the input image, and can be considered as the feature vector of that field.

##### Being robust, rich and trainable, deep convolutional features have been widely adopted for different kinds of visual recognition tasks [25, 12]. 

##### Some previous approaches have employed CNN to learn a robust representation for sequence-like objects such as scene text [22]. 

##### However, these approaches usually extract holistic representation of the whole image by CNN, then the local deep features are collected for recognizing each component of a sequence like object. 

##### Since CNN requires the input images to be scaled to a fixed size in order to satisfy with its fixed input dimension, it is not appropriate for sequence-like objects due to their large length variation. 

##### In CRNN, we convey deep features into sequential representations in order to be invariant to the length variation of sequence-like objects.



contextual	:	

cues	:	

successive	:	

contrasting	:	

differentials	:	

Stacking	:	

arbitrary	:	

traversing	:	

burden	:	

Conceptually	:	

complementary	:	

abstractions	:	

## 2.2. Sequence Labeling

##### A deep bidirectional Recurrent Neural Network is built on the top of the convolutional layers, as the recurrent layers. 

##### The recurrent layers predict a label distribution yt for each frame xt in the feature sequence x = x1, . . . , xT . 

##### The advantages of the recurrent layers are three-fold. 

##### Firstly, RNN has a strong capability of capturing contextual information within a sequence. 

##### Using contextual cues for image-based sequence recognition is more stable and helpful than treating each symbol independently. 

##### Taking scene text recognition as an example, wide characters may require several successive frames to fully describe (refer to Fig. 2). 

##### Besides, some ambiguous characters are easier to distinguish when observing their contexts, e.g. it is easier to recognize “il” by contrasting the character heights than by recognizing each of them separately. 

##### Secondly, RNN can back-propagates error differentials to its input, i.e. the convolutional layer, allowing us to jointly train the recurrent layers and the convolutional layers in a unified network. 

##### Thirdly, RNN is able to operate on sequences of arbitrary lengths, traversing from starts to ends. 

##### A traditional RNN unit has a self-connected hidden layer between its input and output layers. 

##### Each time it receives a frame xt in the sequence, it updates its internal state ht with a non-linear function that takes both current input xt and past state ht−1 as its inputs: ht = g(xt, ht−1). 

##### Then the prediction yt is made based on ht.

##### In this way, past contexts {xt' }t' < t are captured and utilized for prediction. 

##### Traditional RNN unit, however, suffers from the vanishing gradient problem [7], which limits the range of context it can store, and adds burden to the training process. 

##### Long-Short Term Memory [18, 11] (LSTM) is a type of RNN unit that is specially designed to address this problem. 

##### An LSTM (illustrated in Fig. 3) consists of a memory cell and three multiplicative gates, namely the input, output and forget gates. 

##### Conceptually, the memory cell stores the past contexts, and the input and output gates allow the cell to store contexts for a long period of time. 

##### Meanwhile, the memory in the cell can be cleared by the forget gate. 

##### The special design of LSTM allows it to capture long-range dependencies, which often occur in image-based sequences.

##### LSTM is directional, it only uses past contexts. 

##### However, in image based sequences, contexts from both directions are useful and complementary to each other. 

##### Therefore, we follow [17] and combine two LSTMs, one forward and one backward, into a bidirectional LSTM. 

##### Furthermore, multiple bidirectional LSTMs can be stacked, resulting in a deep bidirectional LSTM as illustrated in Fig. 3.b. 

##### The deep structure allows higher level of abstractions than a shallow one, and has achieved significant performance improvements in the task of speech recognition [17]. 

##### In recurrent layers, error differentials are propagated in the opposite directions of the arrows shown in Fig. 3.b, i.e. Back-Propagation Through Time (BPTT). 

##### At the bottom of the recurrent layers, the sequence of propagated differentials are concatenated into maps, inverting the operation of converting feature maps into feature sequences, and fed back to the convolutional layers. 

##### In practice, we create a custom network layer, called “Map-to-Sequence”, as the bridge between convolutional layers and recurrent layers.

![1573618987125](C:\Users\multicampus\AppData\Roaming\Typora\typora-user-images\1573618987125.png)

##### Figure 3. (a) The structure of a basic LSTM unit.

##### An LSTM consists of a cell module and three gates, namely the input gate, the output gate and the forget gate. 

##### (b) The structure of deep bidirectional LSTM we use in our paper. Combining a forward (left to right) and a backward (right to left) LSTMs results in a bidirectional LSTM. 

##### Stacking multiple bidirectional LSTM results in a deep bidirectional LSTM.



## 2.3. Transcription

##### Transcription is the process of converting the per-frame predictions made by RNN into a label sequence. Mathematically, transcription is to find the label sequence with the highest probability conditioned on the per-frame predictions. In practice, there exists two modes of transcription, namely the lexicon-free and lexicon based transcriptions. A lexicon is a set of label sequences that prediction is constraint to, e.g. a spell checking dictionary. In lexiconfree mode, predictions are made without any lexicon. In lexicon-based mode, predictions are made by choosing the label sequence that has the highest probability


## 2.3.1 Probability of label sequence

##### We adopt the conditional probability defined in the Connectionist Temporal Classification (CTC) layer proposed
by Graves et al. [15]. The probability is defined for label sequence l conditioned on the per-frame predictions
y = y1, . . . , yT , and it ignores the position where each label in l is located. Consequently, when we use the negative
log-likelihood of this probability as the objective to train the
network, we only need images and their corresponding label sequences, avoiding the labor of labeling positions of
individual characters.
The formulation of the conditional probability is briefly
described as follows: The input is a sequence y =
y1, . . . , yT where T is the sequence length. Here, each
yt ∈ <|L0
|
is a probability distribution over the set L
0 =
L ∪ , where L contains all labels in the task (e.g. all English characters), as well as a ’blank’ label denoted by . A
sequence-to-sequence mapping function B is defined on sequence π ∈ L0T
, where T is the length. B maps π onto l
by firstly removing the repeated labels, then removing the
’blank’s. For example, B maps “--hh-e-l-ll-oo--”
(’-’ represents ’blank’) onto “hello”. Then, the conditional probability is defined as the sum of probabilities of
all π that are mapped by B onto l:

![1573622509879](C:\Users\multicampus\AppData\Roaming\Typora\typora-user-images\1573622509879.png)

where the probability of π is defined as p(π|y) =
QT
t=1 y
t
πt
, y
t
πt
is the probability of having label πt at time
stamp t. Directly computing Eq. 1 would be computationally infeasible due to the exponentially large number
of summation items. However, Eq. 1 can be efficiently
computed using the forward-backward algorithm described
in [15].