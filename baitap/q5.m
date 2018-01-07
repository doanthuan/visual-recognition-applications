function q5(nNumber)
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');

    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    imgTest = imgTestAll(:, nNumber);
    lblPredictTest = predict(Mdl, imgTest');
    
    
     figure;
     img2D = reshape(imgTest, 28, 28);
     imshow(img2D);
     strLblPredictTest = num2str(lblPredictTest);
     title(strLblPredictTest);
end
