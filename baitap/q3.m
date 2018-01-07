function q3()

imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');

result = zeros(10, 1);
numLabels = size(lblTrainAll, 1);
for i = 0:9
    for j = 1:numLabels
        lblNumber = lblTrainAll(j);
        if(i == lblNumber)
            result(i+1, 1) = result(i+1, 1) + 1;    
        end
    end
end
result

end