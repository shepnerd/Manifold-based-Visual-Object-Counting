function [ cnt, denImg] = EVOCounting_v3( im, db, conf)
% im : query image

[he, wi, ~] = size(im);

patchSize = conf.patchSize;
patchStep = conf.patchStep;
lambda = conf.lambda;
kernel = conf.kernel;
scalar_ker = conf.scalar_ker;
mean_patch = db.mean;
coef = db.coef;

grid = sampling_grid([he,wi], [patchSize patchSize], [patchSize-patchStep ...
    patchSize-patchStep], [0 0], 1);

patches = zeros(size(grid,1),size(grid,2),size(im,3),size(grid,3));
for i = 1 : size(im,3)
    imt = im(:,:,i);
    patches(:,:,i,:) = imt(grid);
%     for j = 1 : size(grid,3)
%         patches(:,:,i,j) = imt(grid(:,:,j));
%     end
end
% patches = double(im(grid));
featVec = reshape(patches, [size(patches,1)*size(patches,2)*size(patches,3) size(patches,4)]);
nsamples = size(featVec,2);
featVec = coef*(featVec-repmat(mean_patch,[1 nsamples]));
if conf.opt == 4
    l2 = sum(featVec.^2).^0.5+eps;
    l2n = repmat(l2,size(featVec,1),1);
%     featVec = 0.3*featVec ./ l2n;
%     featVec = 0.6*featVec ./ l2n;
    featVec = featVec ./ l2n;
end

cc = 0;
tic;

centroids = db.centroids;
nAtom = size(centroids,2);
denPatches = zeros(patchSize*patchSize,nsamples);

split = conf.split;
opt = conf.opt;
K = conf.K;

switch opt
    case 1
        % least square
        computeW = @(x,D) ((D'*D)\D'*x);
    case 2
        % energy constraint
        computeW = @(x,D) ((D'*D+lambda*eye(size(D,2)))\D'*x);
    case 3
        % nonnegative least square
        computeW = @(x,D) (lsqnonneg(D,x));
    case 4
        sparsity = 3;
        computeW = @(x,D) (omp(D'*x,D'*D,sparsity));
    otherwise
        computeW = @(x,D) ((D'*D+lambda*eye(size(D,2)))\D'*x);
end

normalize = @(a)(a./sum(a));

% t1 = 0; %% time recording
% t2 = 0; %% tiem recording
nf = size(featVec,1);
nd = patchSize.^2;
for i = 1 : nsamples
    f = featVec(:,i);
    [~,idx] = min(sum((centroids-repmat(f,[1 nAtom])).^2));
    
    if split(idx) == 1
        subcentroid = db.subcentroids{idx};
        [~,sidx] = min(sum((subcentroid - repmat(f, [1, size(subcentroid,2)])).^2));
        subspace = db.dict{idx,sidx};
        denspace = db.dens{idx,sidx};
    else
        subspace = db.dict{idx,1};
        denspace = db.dens{idx,1};
    end
    nSubspace = size(subspace,2);
    
    if opt < 6
        nK = min(nSubspace,K);
        dict = zeros(nf,nK);
        dist = sum((subspace - repmat(f, [1, nSubspace])).^2);
        dens = zeros(nd,nK);
        for t = 1 : nK
            [~, ridx] = min(dist);
            dict(:,t) = subspace(:,ridx);
            dist(ridx) = 1e60;
            dens(:,t) = denspace(:,ridx);
        end
        if opt == 4
            l2 = sum(dict.^2).^0.5+eps;
            l2n = repmat(l2,size(dict,1),1);
            dict = dict./l2n;
        end
        if sum(f.^2)^0.5 == 0
            denPatches(:,i) = mean(dens,2);
        else
            w = computeW(f,dict);
            w = normalize(w); 
            denPatches(:,i) = dens * w; 
        end
    else
        switch opt
            case 6
                dif = subspace'*f;
            case 7
                dif = (subspace-repmat(f,[1 nSubspace])).^2;
                dif = scalar_ker(sum(dif))';
            otherwise
                dif = abs(subspace-repmat(f,[1 nSubspace]));
                dif = scalar_ker(sum(dif))';
        end      
        denPatches(:,i) = denspace*dif;
    end
    
    cc = cc + 1;
    if mod(cc,500) == 0
        fprintf('*');
    end
end

toc;
denImg = overlap_add(denPatches,[he,wi],grid);
cnt = sum(denImg(:));
fprintf('\n');

% fprintf('t1 > %f s, t2 > %f s\n', t1, t2); %% time recording

end