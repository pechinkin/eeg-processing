### EEG PROCESSING
Study project in BIT, Beijing. Subject : Brain Computer Interface.

#### Task
There are given files with labels, time of trial starts (duration - 5 sec), EEG data of experiments, where person imagined movement of right hand or both feet. Neet to map signals activity with one of these actions. Preprocess data, extract features, classify and test your model.

#### Steps I did
1. Preprocessing (I used only 3 electrodes, which are associated with imaginary movements - Cz, C3, C4), which includes 
1.1 Baseline correction
1.2 Filtering (0.5 ... 30 Hz)
1.3 PCA
1.4 CAR
1.5 Epoching

2. Extracting features
**Time domain**
2.1.1 RMS
2.1.2 Mean amplitude
2.1.1 Variance
2.1.3 Kurtosis
**Frequency domain**
2.2.1 Power
2.2.2 Peak frequency
2.2.3 Spectral centroid

3. Classification
I used 70% of data for training and remaining part for testing. For classification i used Support Vector Machine (SVM).

Accuracy of trained model is 60%.