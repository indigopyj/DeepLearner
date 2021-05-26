# DeepLearner
### :OpenCV-based Driver Drowsiness Detection System using 3D Facial Movement Estimation
### 딥러닝(CNN) 기반 운전자 머리 및 시선 추적을 통한 졸음운전 방지
Notion [link](http://bit.ly/ewhadeeplearner), Github [link](https://github.com/indigopyj/DeepLearner)
### Description
#### This program detects the drowsiness of a driver in real time with using OpenCV Python.

### Flow Charts
<div>
  <img width= 500 src="https://user-images.githubusercontent.com/17904547/70920798-79c61880-2066-11ea-8216-44690f13286f.png">
  <h5>[pic 1] Flow chart of optimization<br></h5>
  <img width = 300 src="https://user-images.githubusercontent.com/17904547/70920824-877b9e00-2066-11ea-9a1a-268e99d7fbfb.png">
  <h5>[pic 2] Flow chart of detection of drawsiness<br></h5>
  </div>

### Prerequisite : list to install
- dlib [link](http://dlib.com)
 : a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems.
- cmake [link](https://cmake.org/download/)
  : open-source software tool for managing the build process of software using a compiler-independent method. It supports directory hierarchies and applications that depend on multiple libraries.
##### After that, type these codes for complete install.
~~~
python setup.py build
python setup.py install
~~~
#### Collaborators : @indigopyj
### Reference
#### [Deepgaze](https://github.com/mpatacchiola/deepgaze)
#### [PyImageSearch](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
