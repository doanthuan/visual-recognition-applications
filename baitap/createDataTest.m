function createDataTest()
imgTrainAll = loadMNISTImages('./t10k-images.idx3-ubyte');
lblTrainAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
nTrainImages = size(imgTrainAll, 2);

%% Create Folders
for i = 0:9
    mkdir(['DataTest\' num2str(i)]);
end

index = zeros(1, 10);
for i = 1:nTrainImages
    imgI = imgTrainAll(:, i);
    img2D = reshape(imgI, 28, 28);
    category = lblTrainAll(i);
    %% Padding fname with leading zeros
    strFileName = ['image_' num2str(index(1, category + 1), '%05d'),'.jpg'];
    strPath = ['DataTest\' num2str(category) '\' strFileName];
    %% Save images
    imwrite(img2D, strPath);
    index(1, category + 1) = index(1, category + 1) + 1;
end