% example-based visual object counting
% Code copyright: Shepnerd
% Contact: wygamle@pku.edu.cn

clear all;

initGlobalVars;

mkdir(saveName);

dbstop if error;

errArray = zeros(folders + 2, 1);
resArray = cell(folders, 1);

trueCount = zeros(nTest, 1);
estCount = zeros(nTest, 1);
denMats = cell(1, nTest);
synMats = cell(1, nTest);
cntMats = cell(1, nTest);

for k = 1 : folders
    
    % dictName = ['dict1' '.mat'];
    dictName = [saveName '_folder' num2str(k) '.mat'];
    if exist(dictName,'file')
        load(dictName,'Dict','centroids');
    else
        [features, images, gtDensities, mask] = createQuickAccess(quickFileName, dataset, totalNum);

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
        end
        
        db.centroids = centroids';
        db.subcentroids = subcentroids';
        db.dict = dict;
        db.dens = dens;
        db.split = split;
        
        save(dictName,'db');
    end
%     t = 0; %% time recording
    conf.split = db.split;
    conf.opt = opt;
    conf.K = Kt;
    for i = 1 : nTest
        if useMask
            teSet.gtDensities{i}(mask == 0) = 0;
        end
        
        trueCount(i) = sum(teSet.gtDensities{i}(:));
        im = teSet.images{i}(2:end,2:end);
        
%         tic; %% time recording
        [estCount(i), denMats{i}, synMats{i}, cntMats{i}] = EVOCounting(im, db, region.size, region.teStep, conf);
%         t = t + toc; %% time recording
        
        if strcmp(dataset,'cell')
            outIm = zeros(246, 246, 3);
            outIm(:, :, 3) = synMats{i};
        else
            outIm = synMats{i};
        end
        
        if useMask
            denMats{i}(mask == 0) = 0;
            estCount(i) = sum(denMats{i}(:));
        end

        imwrite(uint8(outIm), [saveName '/syn_' num2str(i+nTest) '_folder_' num2str(k) '.png']);
        fprintf('Image #%d: trueCount = %f, ANR predicted count = %f...\n',...
            i + nTest, trueCount(i), estCount(i));
    end
    
%     fprintf('totoal time = %f s, mean time = %f s...\n', t, t / nTest); %% time recording
    
    result.estCount = estCount;
    result.trueCount = trueCount;
    result.meanerr = mean(abs(trueCount - estCount));
    result.synMats = synMats;
    result.denMats = denMats;
    
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

