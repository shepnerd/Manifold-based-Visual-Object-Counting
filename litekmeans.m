function [ label, centroid ] = litekmeans(X, k)
% perform k-means clustering
% X: n*d data matrix
% k: number of seeds
X = X';
[d, n] = size(X);
last = 0;

label = ceil(k*rand(1,n));

while any(label ~= last)
    [u,~,label] = unique(label);
    k = length(u);
    E = sparse(1:n,label,1,n,k,n); % transform label into indicator matrix
    
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));
    last = label';
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
end

[~,~,label] = unique(label);

centroid = zeros(d, k);

for i = 1 : k
    cluster = X(:,label == i);
    centroid(:,i) = sum(cluster,2) / size(cluster,2);
end
centroid = centroid';

end

