function q6(nNumber)
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');

    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    imgTest = imgTestAll(:, nNumber);
    lblPredictTest = predict(Mdl, imgTest');
    lblImageTest = lblTestAll(nNumber);
    
     figure;
     img2D = reshape(imgTest, 28, 28);
     imshow(img2D);
     
     strLabelImage = ['Ban dau ', num2str(lblTestAll(nNumber)), '.'];
     strLabelImage = [strLabelImage, ' Du doan: ', num2str(lblPredictTest), '.'];
     
     if( lblPredictTest == lblImageTest)
         strLabelImage = [strLabelImage, ' Ket qua dung. '];
     else
         strLabelImage = [strLabelImage, ' Ket qua sai. '];
     end
     
     title(strLabelImage);
end
