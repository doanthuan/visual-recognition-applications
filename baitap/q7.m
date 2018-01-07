function q7(nNumber)
imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');

imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');

Mdl = fitcknn(imgTrainAll', lblTrainAll);

nResult = 0;
nLabelTest = size(lblTestAll, 1);
for i = 1:nLabelTest
    lblTest = lblTestAll(i);
    if( lblTest == nNumber)
        imgTest = imgTestAll(:, i);
        lblPredictTest = predict(Mdl, imgTest');
        if( lblPredictTest ~= lblTest)
            nResult = nResult + 1;
        end
    end
end
figure;
title(num2str(nResult));
end
