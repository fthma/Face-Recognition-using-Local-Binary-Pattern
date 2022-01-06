from scipy.io import loadmat
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#load the dataset
print("loading the dataset")
ar=loadmat('AR_database.mat')

Training=ar['Tr_dataMatrix']
Training=ar['Tr_dataMatrix']
TrainingTrans=np.transpose(Training)
Labels=ar['Tr_sampleLabels']

TestLabels=ar['Tt_sampleLabels']
TestingData=ar['Tt_dataMatrix']
Testing=np.transpose(TestingData)

#function to get 3x3 windows of eac pixel in the image 
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
def getLBP(TrainingImages):
    Histograms=[]
    for tr in TrainingTrans:
        Image=tr.reshape(120,165)
        I=np.resize(Image,(64,64))
        PaddedI=np.pad(I,((1,1),(1,1)),'edge')
        LBPArray=[]
        for (x, y, window) in sliding_window(PaddedI, stepSize=1, windowSize=(3, 3)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != 3 or window.shape[1] != 3:
                continue
            Zp=window-window[1][1]
            Zp=Zp.astype(np.int8)
            S_Zp=np.where(Zp<0, 0,1)
            lbp=S_Zp[1][2]*np.power(2,0) + S_Zp[2][2]*np.power(2,1) + S_Zp[1][1]*np.power(2,2) + S_Zp[2][0]*np.power(2,3) + S_Zp[1][0]*np.power(2,4) + S_Zp[0][0]*np.power(2,5) + S_Zp[0][1]*np.power(2,6) + S_Zp[0][2]*np.power(2,7) 
            LBPArray.append(lbp)
        LBPArray=np.asarray(LBPArray).reshape((64,64))
        #h=np.histogram(LBPArray)
        #np.savetxt('test.txt',LBPArray)
        h=np.histogram(LBPArray.ravel(),256,[0,256])
        Histograms.append(h[0])
    
    return Histograms

def SVM_Classification(TrainingData,TrainLabels,TestData,TestLabels,k='linear'):
    
      
    Train=np.asarray(TrainingData)
    #Labels=np.asarray(TrainLabels)
    
    Test=np.asarray(TestData)
    #TestLabels=np.asarray(TestLabels)
    
    clf=SVC(kernel=k,gamma='scale')
    clf.fit(TrainingData,TrainLabels)
    
    prediction=clf.predict(TestData)
    
    Acc_Score=accuracy_score(TestLabels,prediction)
    
    
    
    print("For SVM classified with kernel={}".format(k))
    print("Accuracy Score is :{}".format(Acc_Score))
    

    return prediction

def KNNCLAssifier(TrainingData,TrainingLabels,TestData,TestingLabels):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier()
    neigh.fit(TrainingData,TrainingLabels)
    
    prediction=neigh.predict(TestData)
    
    Acc_Score=accuracy_score(TestingLabels,prediction)
    #print(AS)
    print("For KNN classifier with 5 neighbouring nodes is:")
    print("Accuracy Score is :{}".format(Acc_Score))
    
    return prediction




print("Finding LBP  for training data")
TrainingLBP=getLBP(TrainingTrans)
print("Finding LBP  for testing data")
TestingLBP=getLBP(Testing)

#training SVM with linear kernel
print("----------------------------")
print("Training the data with linear SVM")
Linear_predic=SVM_Classification(TrainingLBP,Labels.ravel(),TestingLBP,TestLabels.ravel())

#training the SVM with kernel=rbf
print("----------------------------")
print("Training the data with rbf SVM")
predic=SVM_Classification(TrainingLBP,Labels.ravel(),TestingLBP,TestLabels.ravel(),k='rbf')

#training the SVM with kernel=polynomial
print("----------------------------")
print("Training the data with polynomial SVM")
predic=SVM_Classification(TrainingLBP,Labels.ravel(),TestingLBP,TestLabels.ravel(),k='poly')
accuracy_score(TestLabels.ravel(),predic)

#training the data over KNN 
print("----------------------------")
print("Training the data with KNN classifier")
p=KNNCLAssifier(TrainingLBP,Labels.ravel(),TestingLBP,TestLabels.ravel())





    