### EEG PROCESSING
Study project in BIT, Beijing. Subject : Brain Computer Interface.

#### Task
There are given files with labels, time of trial starts (duration - 5 sec), EEG data of experiments, where person imagined movement of right hand or both feet. Neet to map signals activity with one of these actions. Preprocess data, extract features, classify and test your model.

#### Steps I did
1. Preprocessing (I used only 3 electrodes, which are associated with imaginary movements - Cz, C3, C4), which includes 
- Baseline correction
- Filtering (0.5 ... 30 Hz)
- PCA
- CAR
- Epoching

2. Extracting features
   
`Time domain`
- RMS
- Mean amplitude
- Variance
- Kurtosis

`Frequency domain`
- Power
- Peak frequency
- Spectral centroid

3. Classification
I used 70% of data for training and remaining part for testing. For classification i used Support Vector Machine (SVM).

Accuracy of trained model is 60%.
