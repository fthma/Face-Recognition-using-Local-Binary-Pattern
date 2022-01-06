# Face-Recognition-using-Local-Binary-Pattern

In this project we use the AR face dataset for face recognition using Local Binary Pattern for 
texture extraction .The AR dataset contains over 4000 colour face images of 126 subjects, 
including frontal views of faces with different facial expressions, illumination conditions and 
occlusions. For this project, we will only use a subset of the dataset with large variations in 
both illumination and expression, which corresponds to 50 male subjects and 50 female 
subjects. For each subject there are two sections, one for training and the other for testing. 
Each section contains 7 images per subject. We take each image in the dataset and find the 
LBP for each of the pixels in the image and then find the histogram of the LBP matrix. The 
histogram of training images is then used for training the SVM classifier with different 
kernels viz linear, rbf, poly and also the KNN classifier. The accuracy of each of the classifier 
is found and compared
