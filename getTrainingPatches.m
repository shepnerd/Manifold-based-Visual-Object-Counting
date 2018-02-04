function [ features ] = getTrainingPatches( gradIm, patch_size, patch_step, opt)
% UNTITLED3 Summary of this function goes here
% Detailed explanation goes here
[row, col, gNum] = size(gradIm);

xrow = 1:patch_step:row-patch_size+1;
ycol = 1:patch_step:col-patch_size+1;
xlen = length(xrow);
ylen = length(ycol);

if opt == 1
    features = zeros(xlen*ylen*2,patch_size^2*gNum);
else
    features = zeros(xlen*ylen*1,patch_size^2*gNum);
end

sp = xlen*ylen;
for x = 1:xlen
    for y = 1:ylen
        r = xrow(x);
        c = ycol(y);
        feaPatch = gradIm(r:r+patch_size-1,c:c+patch_size-1,:);
        idx = (x-1)*ylen+y;
        features(idx,:) = feaPatch(:)';
        if opt == 1
            t = flip(feaPatch,2);
            features(idx+sp,:) = t(:)';
        end
    end
end

end

