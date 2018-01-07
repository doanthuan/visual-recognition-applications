function q2(nNumber)

    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
     figure;
     img = imgTestAll(:, nNumber);
     img2D = reshape(img, 28, 28);
     strLabelImage = num2str(lblTestAll(nNumber));
     strLabelImage = [strLabelImage, '(', num2str(nNumber), ')'];
     imshow(img2D);
     title(strLabelImage);
    
end
