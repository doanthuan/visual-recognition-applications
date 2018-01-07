function th2()
    q1 = randi([1, 200]);
    fprintf('\n Q1: %d', q1);
    
    A = ones(10, 10);
    q2 = A(3, 5);
    fprintf('\n Q2: %d', q2);
    
    A = zeros(100, 200);
    fprintf('\n Q3:');
    A
    
    q4 = size(A, 1);
    fprintf('\n Q4: %d', q4);
    
    q5 = A(:, 10);
    fprintf('\n Q5:');q5
    
    q6 = A(10, :);
    fprintf('\n Q6:');q6
    
    q7 = zeros(784, 1);
    %q7 = rand(1, 784);
    q7 = reshape(q7, 28, 28);
    fprintf('\n Q7:');q7
    
    figure;
    imshow(q7);
    
end