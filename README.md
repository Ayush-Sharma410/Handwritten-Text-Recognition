# Handwritten-Text-Recognition

Problem Statement: 
To improve the accuracy and efficiency of Handwritten Text Recognition systems to better recognize and convert handwritten text into machine-encoded text, despite the variability in handwriting styles, the complexity of the written script, and the low quality of input images. 
 
Description of problem: 
Handwritten Text Recognition (HTR) systems face several challenges due to the variability of handwriting styles, the complexity of the written script, and the low quality of input images. These challenges can result in low accuracy rates and errors in recognizing handwritten text, making it difficult to use HTR systems for practical applications such as document digitization or postal address recognition. Addressing this problem requires the development of innovative techniques and algorithms for preprocessing, segmentation, feature extraction, machine learning, and error correction, as well as the use of high-quality data and advanced hardware and software technologies. By addressing this problem, HTR systems can become more effective and reliable tools for digitization, data entry, and other applications, leading to significant benefits for businesses, organizations, and individuals. 
Page 1 of 8 
Code and Explanation: 
The following is the google colab link for the code. 
https://colab.research.google.com/drive/1wg9gBMaAslEomghRwSzXaFXkONoQbEV?usp=sharing#scrollTo=b27fJoaegkjO 
 
  
 
Prescriptions folder is to be added before running the program: 
https://drive.google.com/drive/folders/1Ptexl9CiKC9iWy09Pl6fvSX49ilTRjGB 
	• 	Prescriptions will have pictures of various handwritten texts. 
 
Directory structure: 
  
 
Page 2 of 8 
 
We install pytesseract in Google Collaboratory. 
  
 
IMPORTING USED MODULES 
Brief description of the imported modules: 
1.	os: The OS module in Python provides functions for interacting with the operating system. OS comes under Python’s standard utility modules. This module provides a portable way of using operating system-dependent functionality. The *os* and *os.path* modules include many functions to interact with the file system. 
 
2.	cv2: cv2 is the module import name for opencv-python, "Unofficial pre-built CPU-only OpenCV packages for Python". The traditional OpenCV has many complicated steps involving building the module from scratch, which is unnecessary. 
 
3.	pytesseract: Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and “read” the text embedded in images. 
 
4.	numpy: NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
 
5.	Keras ImageDataGenerator is used to take the inputs of the original data and then transform it on a random basis, returning the output resultant containing solely the newly changed data. 
 
6.	from keras.models import Sequential: A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.   
 
7.	from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense: The Flatten layer converts the 28x28x50 output of the convolutional layer into a single one-dimensional vector, that can be used as input for a dense layer. The last dense layer has the most parameters. This layer connects every single output 'pixel' from the convolutional layer to the 10 output classes. 
 
8.	from keras.callbacks import EarlyStopping: Early Stopping in Keras. Keras supports the early stopping of training via a callback called EarlyStopping. This callback allows you to specify the performance measure to monitor, the trigger, and once triggered, it will stop the training process. 
 
  
 
Next, the directory path where the images are stored is defined, along with the image size and batch size. 
 
  
 
Two data generators are then created using the Keras ImageDataGenerator class, one for training and one for validation. The training generator applies several data augmentation techniques to increase the variability of the training data, while the validation generator only rescales the images. 
 
 
 
 
Page 4 of 8 
  
 
Afterward, the model is defined as a sequential neural network with several convolutional, pooling, and dense layers. The first layer is a 2D convolutional layer with 32 filters, followed by a max pooling layer, and then two more sets of 2D convolutional and max pooling layers with 64 and 128 filters, respectively. The output of the final max pooling layer is flattened, and then two dense layers are added with ReLU activation function, followed by a final dense layer with a sigmoid activation function to produce binary classification output. 
 
  
 
The model is then compiled with an optimizer, loss function, and performance metric. 
 
  
 
 
The model is trained using the fit() method. An early stopping callback is used to monitor the validation loss and stop the training if the validation loss does not improve after a certain number of epochs. 
 
  
 
The code is evaluating a machine learning model and performing optical character recognition (OCR) on a set of images. 
Firstly, the code sets up a test data generator using the ImageDataGenerator class from the Keras library, which rescales pixel values of images to between 0 and 1. It then loads a set of test data from a directory using flow_from_directory method and sets the target image size, batch size, color mode, and class mode. 
The model.evaluate method is called to evaluate the trained model on the test data, and the resulting test loss and accuracy are stored in the test_loss and test_acc variables, respectively. 
If an exception is thrown, it prints "train the data". This indicates that the model hasn't been trained before, so it needs to be trained before evaluating on test data. 
 
  
 
 
 
 
 
 
Page 6 of 8 
The code then loops through all the files in a specified image folder and loads each image in grayscale. It uses the pytesseract.image_to_string method from the pytesseract library to perform OCR on the image and extract any text present in the image. The recognized text is printed along with the corresponding filename. 
 
  
 
Implementation: 
 
Sample 1: 
  
Output 1: 
  
 
 
 
 
Sample 2: 
  
 
Output 2: 
  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
