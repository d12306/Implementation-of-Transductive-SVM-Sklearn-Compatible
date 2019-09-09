# Implementation-of-Transductive-SVM-Sklearn-Compatible

The main code is from https://github.com/tmadl/semisup-learn, but it is very out-dated, so I reimplemented it for correct usgae in my own research. 

The main class is in tsvm.py  

The detailed implementation is in qns3vm.py  

Try to set up the transductive setting before you call the sklean-like function, such as setting the labels of the unlabeled dataset to [-1] and concatenate the training and testing set together. The final prediction is the one for all the data points of the inputs. We should split them apart for the training and testing data. 
