
% example-based visual object counting
% Code copyright: Shepnerd
% Contact: wygamle@pku.edu.cn

clear all;

initGlobalVars;

dbstop if error;

errArray = zeros(folders + 2, 1);
resArray = cell(folders, 1);

trueCount = zeros(nTest, 1);
estCount = zeros(nTest, 1);
denMats = cell(1, nTest);

for k = 1 : folders
    
    % dictName = ['dict1' '.mat'];
    dictName = [saveName '_folder' num2str(k) '.mat'];
    if exist(dictName,'file')
        load(dictName,'Dict','centroids');
    else
        [features, images, gtDensities, mask] = createQuickAccess(quickFileName, dataset, totalNum);
        
        nf = length(features);
        if donormalize
            for kt = 1 : nf
                images{kt} = im2double(images{kt});
                features{kt} = im2double(uint16(features{kt}));
            end
        end
        %
        data.features = features;
        data.images = images;
        data.gtDensities = gtDensities;
        params.nTrain = nTrain;
        params.nTest = nTest;
        params.trRange = trRange;
        params.teRange = teRange;

        [trSet, teSet] = genTrTe(data, params);

        [trFeats, trRaws, trDen] = stripPatches(trSet, region.size, region.trStep);
        
        %% added new processing step
        disp(['pca ...']);
        if dopca
            mean_patch = mean(trRaws);
            trRaws = trRaws - repmat(mean_patch,[size(trRaws,1) 1]);
            [coef, score, l] = pca(trRaws);
            for kk = 1 : length(l) - 1
                if sum(l(1:kk+1)) / sum(l) > 0.99
                    break;
                end
            end
            kk = kk + 1;
            coef = coef(:,1:kk);
            trRaws = trRaws*coef;
        else
            mean_patch = zeros(1,size(trRaws,2));
            coef = eye(size(trRaws,2));
        end
        
        disp(['Training data prepared ...']);

        disp(['Train anchored neighborhood examplar ...']);

        K = region.types;
        [idx, centroids] = litekmeans(trRaws, K);
        K1 = size(centroids,1);
        disp(['Learning the projective matrix ...']);
        dict = cell(K1,K);
        dens = cell(K1,K);
        subcentroids = cell(K1,1);
        split = zeros(K1,1);
        for i = 1 : K1
            subRaws = trRaws(idx == i,:);
            subDens = trDen(idx == i,:);
            scnt = size(subRaws,1);
            
            if scnt > 2*K
                [sidx, scentroids] = litekmeans(subRaws,K);
                K2 = size(scentroids,1);
                subcentroids{i} = scentroids';
                for j = 1 : K2
                    dict{i,j} = subRaws(sidx==j,:)';
                    dens{i,j} = subDens(sidx==j,:)';
                end
                split(i) = 1;
            else
                dict{i,1} = subRaws';
                dens{i,1} = subDens';
            end
            
            if opt > 5
                if split(i) == 0
                    nK = size(dict{i,1},2);
                    Di = dict{i,1};
                    Dd = dens{i,1};
                    if nK > num_of_instances_locally
                        nK = num_of_instances_locally;
                        [Di,Dd,nK] = compressAnchor(Di,Dd,nK);
                    end
                    
                    ThetaY = zeros(nK,nK);
                    switch opt
                        case 6
                            ThetaY = scalar_ker(Di'*Di);
                        case 7
                            for ii = 1 : nK
                                tv = Di(:,ii);
                                tmp = (bsxfun(@minus, tv, Di)).^2;
                                ThetaY(ii,:) = scalar_ker(sum(tmp));
                            end
                        otherwise
                            for ii = 1 : nK
                                tv = Di(:,ii);
                                tmp = abs(bsxfun(@minus, tv, Di));
                                ThetaY(ii,:) = scalar_ker(sum(tmp));
                            end
                    end
                    dens{i,1} = Dd/(ThetaY+lambda*eye(nK));
                    dict{i,1} = Di;
                else
                    nPart = size(subcentroids{i},2);
                    for z = 1 : nPart
                        nK = size(dict{i,z},2);
                        Di = dict{i,z};
                        Dd = dens{i,z};
                        
                        if nK > num_of_instances_locally
                            nK = num_of_instances_locally;
                            [Di,Dd,nK] = compressAnchor(Di,Dd,nK);
                        end
                        
                        ThetaY = zeros(nK,nK);
                        switch opt
                            case 6
                                ThetaY = scalar_ker(Di'*Di);
                            case 7
                                for ii = 1 : nK
                                    tv = Di(:,ii);
                                    tmp = (bsxfun(@minus, tv, Di)).^2;
                                    ThetaY(ii,:) = scalar_ker(sum(tmp));
                                end
                            otherwise
                                for ii = 1 : nK
                                    tv = Di(:,ii);
                                    tmp = abs(bsxfun(@minus, tv, Di));
                                    ThetaY(ii,:) = scalar_ker(sum(tmp));
                                end
                        end
                        dens{i,z} = Dd/(ThetaY+lambda*eye(nK));
                        dict{i,z} = Di;
                    end
                end
            end
        end
        
        db.centroids = centroids';
        db.subcentroids = subcentroids';
        db.dict = dict;
        db.dens = dens;
        db.split = split;
        db.mean = mean_patch';
        db.coef = coef';
        
        save(dictName,'db');
    end
    
    trueDensity = teSet.gtDensities{1};
    r = cropImage(size(trueDensity), region.size, region.teStep);
    if useMask
        mask = mask(r.h,r.w);
    end
%     t = 0; %% time recording
    conf.split = db.split;
    conf.opt = opt;
    conf.K = Kt;
    conf.patchSize = region.size;
    conf.patchStep = region.teStep;
    conf.kernel = kernel;
    conf.scalar_ker = scalar_ker;
    conf.lambda = lambda;
    for i = 1 : nTest
        trueDensity = teSet.gtDensities{i};
        if useMask
            trueDensity(mask == 0) = 0;
        end
        trueDensity = trueDensity(r.h,r.w);
        trueCount(i) = sum(trueDensity(:));
        im = teSet.images{i};
        im = im(r.h,r.w);
        
%         tic; %% time recording
        [estCount(i), denMats{i}] = EVOCounting_v2(im, db, conf);
%         t = t + toc; %% time recording
        
        if useMask
            denMats{i}(mask == 0) = 0;
            estCount(i) = sum(denMats{i}(:));
        end
        
        fprintf('Image #%d: trueCount = %f, ANR predicted count = %f...\n',...
            i + nTest, trueCount(i), estCount(i));
    end
    
%     fprintf('totoal time = %f s, mean time = %f s...\n', t, t / nTest); %% time recording
    
    result.estCount = estCount;
    result.trueCount = trueCount;
    result.denMats = denMats;
    result.meanerr = mean(abs(trueCount - estCount));
    
    resArray{k} = result;
    errArray(k) = result.meanerr;
    disp('------');
    fprintf('Folder %03d > ANR average error = %f\n', k, ...
        mean(abs(trueCount-estCount)));
end

errArray(folders + 1) = mean(errArray(1 : folders));
errArray(folders + 2) = std(errArray(1 : folders));

save(saveName, 'errArray', 'resArray');

fprintf('\n\n* Result > ANR %d folders average error = %f~%f\n', folders,...
    errArray(folders+1), errArray(folders+2));

