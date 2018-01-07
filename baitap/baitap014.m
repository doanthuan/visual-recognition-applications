function baitap014()

imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');

% extract input images to hog features
% get hog feature size
img1D = imgTrainAll(:, 1);
img2D = reshape(img1D, 28, 28);
featuresVector = extractHOGFeatures(img2D);
nSize = length(featuresVector);

nTrainData = size(imgTrainAll, 2);
featuresDataTrain = zeros(nSize, nTrainData);

for i = 1:nTrainData
    img2D = reshape(imgTrainAll(:, i), 28, 28);
    featuresDataTrain(:,i) = extractHOGFeatures(img2D);
end

% build model from feature vector
Mdl = fitcknn(featuresDataTrain', lblTrainAll);

% load & extract test images to feature vector
imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');

img1D = imgTestAll(:, 1);
img2D = reshape(img1D, 28, 28);
featuresVector = extractHOGFeatures(img2D);
nSize = length(featuresVector);
nTestData = size(lblTestAll, 2);

featuresDataTest = zeros(nSize, nTestData);
for i = 1:nTestData
    img1D = imgTestAll(:, i);
    img2D = reshape(img1D, 28, 28);
    featuresDataTest(:,i) = extractHOGFeatures(img2D);
end

% predict test data
lblResult = predict(Mdl, featuresDataTest');
nResult = (lblResult == lblTestAll);
nCount = sum(nResult);
fprintf('\n So luong mau dung: %d\n', nCount);

end