function [ Di, Dd, nK ] = compressAnchor( Di, Dd, nK )



[subidx, examples] = litekmeans(Di', nK);
%                 [subidx, examples] = kmeans(Di', nt);
nK = size(examples,1);
Dd_t = zeros(size(Dd,1),nK);
for j = 1 : nK
    cluster = Dd(:,subidx==j);
    Dd_t(:,j) = sum(cluster,2) / size(cluster,2);
end
Di = examples';
Dd = Dd_t;

end

