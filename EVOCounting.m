function [ cnt, denImg, synImg, cntMat ] = EVOCounting( im, db, patchSize, patchStep, conf)
% im : query image
% patch_size : as its name depicts
% patch_step : as its name depicts
[h, w] = size(im);

gridx = 1 : patchStep : w - patchSize + 1;
gridy = 1 : patchStep : h - patchSize + 1;

denImg = zeros(size(im));
synImg = zeros(size(im));
cntMat = zeros(size(im));

cc = 0;
tic;

centroids = db.centroids;
nAtom = size(centroids,2);

computeW = @(x,D) (D\x);

split = conf.split;
opt = conf.opt;
K = conf.K;
if opt == 1
    % least square
    computeW = @(x,D) (D\x);
else
    if opt == 2
        lambda = 1e-3;
        % energy constraint
        computeW = @(x,D) ((D'*D+lambda*eye(size(D,2)))\D'*x);
    else
        if opt == 3
            % nonnegative least square
            computeW = @(x,D) (lsqnonneg(D,x));
        else
            if opt == 4
                % sparsity
                sparsity = 5;
                computeW = @(x,D) (omp(D'*x,D'*D,sparsity));
                if opt == 5
                    % nonnegative sparsity
                else
                    % energy constraint
                    lambda = 1e-3;
                    computeW = @(x,D) ((D'*D+lambda*eye(size(D,2)))\D'*x);
                end
            end
        end
    end
end

normalize = @(a)(a./sum(a));

% t1 = 0; %% time recording
% t2 = 0; %% tiem recording

for ii = 1 : length(gridx)
    for jj = 1 : length(gridy)
%         tic; %% time recording
        
        xx = gridx(ii);
        yy = gridy(jj);
        patch = single(im(yy : yy + patchSize - 1, xx : xx + patchSize - 1, :));
        featVec = patch(:);
        
%         t1 = t1 + toc; %% time recording
        if norm(featVec) > 0

            [~,idx] = min(sum((centroids - repmat(featVec, [1, nAtom])).^2));
        
            if split(idx) == 1
                subcentroid = db.subcentroids{idx};
                [~,sidx] = min(sum((subcentroid - repmat(featVec, [1, size(subcentroid,2)])).^2));
                subspace = db.dict{idx,sidx};
                denspace = db.dens{idx,sidx};
            else
                subspace = db.dict{idx,1};
                denspace = db.dens{idx,1};
            end
            nSubspace = size(subspace,2);
            nK = min(nSubspace,K);
            dict = zeros(size(featVec,1),nK);
            dist = sum((subspace - repmat(featVec, [1, nSubspace])).^2);
            dens = zeros(size(featVec,1),nK);
            for t = 1 : nK
                [~, ridx] = min(dist);
                dict(:,t) = subspace(:,ridx);
                dist(ridx) = 1e60;
                dens(:,t) = denspace(:,ridx);
            end
        
            w = computeW(featVec,dict);
            w = normalize(w);
            estDenPatch = dens * w;
        else
            estDenPatch = zeros(size(featVec));
        end
        
        denImg(yy:yy+patchSize-1,xx:xx+patchSize-1) = denImg(yy:yy+patchSize-1,xx:xx+patchSize-1)+reshape(estDenPatch,[patchSize,patchSize]);
        cntMat(yy:yy+patchSize-1,xx:xx+patchSize-1) = cntMat(yy:yy+patchSize-1,xx:xx+patchSize-1)+1;
        
        synPatch = reshape(centroids(:,idx(1)),[patchSize,patchSize]);
        synImg(yy:yy+patchSize-1,xx:xx+patchSize-1) = reshape(synPatch,[patchSize,patchSize]);
        
%         t2 = t2 + toc; %% time recording
        
        cc = cc + 1;
        if mod(cc,500) == 0
            fprintf('*');
        end
    end
end

toc;
fprintf('\n');
idx = (denImg < 0);
denImg(idx) = 0;
denImg = denImg ./ cntMat;
cnt = sum(denImg(:));

% fprintf('t1 > %f s, t2 > %f s\n', t1, t2); %% time recording

end

