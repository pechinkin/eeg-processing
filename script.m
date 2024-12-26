%% Load the data
load("EEGdata.mat");
load("label.mat");
load("trial.mat");

fs = 512; % Sampling frequency

%% Preprocessing
relevantElectrodes = [5, 7, 9]; % C3, Cz, C4 | motor activity
data = EEGdata(:, relevantElectrodes);

% Baseline correction
for i = 1:size(data, 2)
    data(:, i) = data(:, i) - mean(data(:, i));
end

% Filtering
fpass = [0.2 30];
order = 4;
[b, a] = butter(order, fpass / (fs / 2), 'bandpass');
data = filtfilt(b, a, data);

% PCA
centeredData = data - mean(data, 1);
covarianceMatrix = cov(centeredData);
[eigenVectors, eigenValues] = eig(covarianceMatrix);
[eigenValues, sortedIndices] = sort(diag(eigenValues), 'descend');
eigenVectors = eigenVectors(:, sortedIndices);
explainedVariance = cumsum(eigenValues) / sum(eigenValues);
explainedVarianceThreshold = 0.97;
numComponents = find(explainedVariance >= explainedVarianceThreshold, 1);
projectedData = centeredData * eigenVectors(:, 1:numComponents);

% CAR
averageSignal = mean(projectedData, 2);
carData = projectedData - averageSignal;

%% Epoching
t = 5 * fs; % Duration of each trial in samples (5 seconds)
epochs = cell(length(trial), 1);

for i = 1:length(trial)
    startPoint = trial(i);
    endPoint = startPoint + t - 1;
    epoch = carData(startPoint:endPoint, :);
    epochs{i} = epoch;
end

%% Feature Extraction
windowSize = fs / 4; % 0.25-second window
overlap = 0.5; % 50% overlap
dataFeatures = [];

for epochIdx = 1:length(epochs)
    eegsignal = epochs{epochIdx};
    numWindows = floor((size(eegsignal, 1) - windowSize) / (windowSize * (1 - overlap))) + 1;
    features = [];

    for i = 1:numWindows
        startIndex = 1 + (i - 1) * windowSize * (1 - overlap);
        endIndex = startIndex + windowSize - 1;

        if endIndex > size(eegsignal, 1)
            endIndex = size(eegsignal, 1);
        end

        windowData = eegsignal(startIndex:endIndex, :);

        meanAmplitude = mean(windowData, 1);
        rmsValue = sqrt(mean(windowData.^2, 1));
        variance = var(windowData, 0, 1);
        kurtosisValue = kurtosis(windowData, 1);

        [Pxx, freq] = pwelch(windowData, windowSize, round(0.5 * windowSize), [], fs);

        freq = freq(:);

        numChannels = size(Pxx, 2);
        thetaPower = zeros(1, numChannels);
        muPower = zeros(1, numChannels);
        betaPower = zeros(1, numChannels);
        peakFreq = zeros(1, numChannels);
        spectralCentroid = zeros(1, numChannels);

        for ch = 1:numChannels
            channelPxx = Pxx(:, ch);

            totalPower = sum(channelPxx);

            thetaPower(ch) = sum(channelPxx(freq >= 4 & freq <= 8)) / totalPower;
            muPower(ch) = sum(channelPxx(freq >= 8 & freq <= 12)) / totalPower;
            betaPower(ch) = sum(channelPxx(freq >= 13 & freq <= 30)) / totalPower;

            [~, peakIdx] = max(channelPxx);
            peakFreq(ch) = freq(peakIdx);

            spectralCentroid(ch) = sum(freq .* channelPxx) / sum(channelPxx);
        end

        windowFeatures = [meanAmplitude, rmsValue, variance, kurtosisValue, thetaPower, muPower, betaPower, spectralCentroid, peakFreq];
        features = [features; windowFeatures];
    end

    dataFeatures = [dataFeatures; features];
end


%% Prepare Data for Classification

rng(30);

dataFeatures = normalize(dataFeatures);
trainRatio = 0.7;
numSamples = length(label);
numTrain = round(trainRatio * numSamples);

indices = randperm(numSamples);

trainIdx = indices(1:numTrain);
testIdx = indices(numTrain+1:end);

trainFeatures = dataFeatures(trainIdx, :);
trainLabels = label(trainIdx);

testFeatures = dataFeatures(testIdx, :);
testLabels = label(testIdx);

%% Train Classifier
classifier = fitcsvm(trainFeatures, trainLabels, 'KernelFunction', 'linear', 'Standardize', true);

trainPredictions = predict(classifier, trainFeatures);
trainAccuracy = mean(trainPredictions == trainLabels) * 100;

fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);

%% Test Classifier
testPredictions = predict(classifier, testFeatures);

testAccuracy = mean(testPredictions == testLabels) * 100;
confMat = confusionmat(testLabels, testPredictions);

fprintf('Testing Accuracy: %.2f%%\n', testAccuracy);
disp('Confusion Matrix:');
disp(confMat);
