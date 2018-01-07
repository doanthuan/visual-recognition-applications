function baitap029()
rootFolder = fullfile('DataTrain');
categories = {'0','1','2','3','4','5','6','7','8','9'};
imdsDataTrain = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl01 = countEachLabel(imdsDataTrain);
minSetCount = min(tbl01{:,2});
imdsDataTrain = splitEachLabel(imdsDataTrain, minSetCount, 'randomize');


imdsDataTrain.ReadFcn = @(filename)readAndPreprocessImage(filename);
net = alexnet();
featureLayer = 'fc7';
featuresDataTrain = activations(net, imdsDataTrain, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
lblDataTrain = imdsDataTrain.Labels;

classifier = fitcecoc(featuresDataTrain, lblDataTrain, 'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

rootFolder = fullfile('DataTest');
categories = {'0','1','2','3','4','5','6','7','8','9'};
imdsDataTest = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imdsDataTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
featuresDataTest = activations(net, imdsDataTest, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
lblDataTest = imdsDataTest.Labels;

lblResult = predict(classifier, featuresDataTest);
nResult = (lblResult == lblTestAll);
nCount = sum(nResult);
fprintf('\n So luong mau dung: %d\n', nCount);
end