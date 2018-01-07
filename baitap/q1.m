function q1(numbers)

    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    for i = numbers
        %img = imgTrainAll(:, n);
        lblImg = lblTrainAll(i);
        %fprintf('\n Label: %d', lblImg);
        disp(lblImg);
    end
    
end
