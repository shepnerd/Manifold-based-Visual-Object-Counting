err_mat = zeros(8,1);

for i = 1 : 5
    gt_mat = resArray{i}.trueCount;
    pred_mat = resArray{i}.estCount;
    err_mat(i) = mean(abs(pred_mat-gt_mat));
end

err_mat(6) = mean(err_mat(1:5));
err_mat(7) = std(err_mat(1:5));

for i = 1 : 5
    gt_mat = resArray{1}.trueCount;
    pred_mat = resArray{1}.estCount;
    err_mat(i) = mean((pred_mat-gt_mat+120).^2);
end

err_mat(8) = mean(err_mat(1:5));
fprintf('\n\n* Result > KMVOC folders average error = %f~%f~%f\n', err_mat(6), err_mat(7), err_mat(8));