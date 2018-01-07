function baitap012()
imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');

imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');

imgI1D = imgTrainAll(:,1);
imgI2D = reshape(imgI1D, 28, 28);
featureVector = extractLBPFeatures(imgI2D);
nSize = length(featureVector);
nTrainData = size(imgTrainAll, 2);
featuresDataTrain = zeros(nSize, nTrainData);
for i = 1:nTrainData
    imgI1D = imgTrainAll(:, i);
    imgI2D = reshape(imgI1D, 28, 28);
    featuresDataTrain(:, i) = extractLBPFeatures(imgI2D);
end

Mdl = fitcknn(featuresDataTrain', lblTrainAll);

imgI1D = imgTestAll(:, 1);
imgI2D = reshape(imgI1D, 28, 28);
featureVector = extractLBPFeatures(imgI2D);


nNumTestImages = size(imgTestAll, 2);

imgTestAll_hist = zeros(nBins, nNumTestImages);
for i = 1:nNumTestImages
    imgTestAll_hist(:, i) = imhist(imgTestAll(:, i), nBins);
end

lblResult = predict(Mdl, imgTestAll_hist');
nResult = (lblResult == lblTestAll);
nCount = sum(nResult);
fprintf('\n So luong mau dung: %d \n', nCount);

end