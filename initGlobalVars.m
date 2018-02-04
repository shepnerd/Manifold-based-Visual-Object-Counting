% target dataset
addpath('Data');
% dataset = 'fish';
% dataset = 'bee';
dataset = 'bird';
% dataset = 'ucsd';
% dataset = 'mall';
% dataset = 'cell';
% dataset = 'embryocell';
% dataset = 'fly';
dataset_Mode = 3;
% just for ucsd as it contains many forms of training and testing.
modeName = {'max','down','up','min','dense'}; 
% decide whether use the mask to filter out some outliers
useMask = false;
if strcmp(dataset,'ucsd') || strcmp(dataset,'mall')
    useMask = true;
end

% test resolution
testResolution = 1;

% data dimension reduction
% dopca = false;
dopca = true;

% data normalization
donormalize = true;

% add subroute path
addpath(genpath(['ompbox' filesep]));

% patch setting
region.size = 7;
region.teStep = 4;
region.trStep = 4;

% the number of examplar
region.types = 24;

% the number of training samples.
% nTrain = 2;
if strcmp(dataset,'ucsd') || strcmp(dataset,'mall')
    nTrain = 800;
end

if strcmp(dataset,'cell')
    nTrain = 2;
end
if strcmp(dataset,'bird')
    nTrain = 1;
end
if strcmp(dataset,'bee')
    nTrain = 16;
end
if strcmp(dataset, 'fish')
    nTrain = 16;
end

if strcmp(dataset, 'embryocell')
    nTrain = 4;
end

if strcmp(dataset, 'fly')
    nTrain = 16;
    % the range of training samples
    trRange = 1:50;

    % the number of testing samples.
    nTest = 50;
    assert(nTest < 101);

    % the range of testing samples
    teRange = 51:100;
end

if strcmp(dataset,'cell')
%     region.types = min(nTrain * 4, 24);
    region.types = nTrain * 2;
end

if strcmp(dataset,'cell')
    
    assert(nTrain < 101);

    % the range of training samples
    trRange = 1:100;

    % the number of testing samples.
    nTest = 100;
    assert(nTest < 101);

    % the range of testing samples
    teRange = 101:200;
else
    if strcmp(dataset,'mall')
        trRange = 1:800;
        
        % the number of testing samples.
        nTest = 1200;
        
        % the range of testing samples
        teRange = 801:2000;
        if testResolution == 1
            trRange = 1 : 40 : 800;
            nTrain = length(trRange);
            teRange = 801 : 12: 2000;
            nTest = length(teRange);
        end
    else % dataset is ucsd
        if strcmp(dataset,'ucsd')
            switch dataset_Mode
                case 1
                    % the range of training samples
                    trRange = 600 : 5 : 1400;
                case 2
                    trRange = 1205 : 5 : 1600;
                case 3
                    %trRange = 805 : 5 : 1100;
                    trRange = 805 : 10 : 1100;
                case 4
                    trRange = 640 : 80 : 1360;
                    
                case 5
                    trRange = 601 : 1400;
            end
            if nTrain > length(trRange)
                nTrain = length(trRange);
            end

            teRange = 1 : 2000;
            t = ones(2000,1);
            t(trRange) = 0;
            teRange = teRange(logical(t));
            % for short review
            teRange = teRange(1:50:end);
            nTest = length(teRange);
        else
            if strcmp(dataset,'bird')
                trRange = 1;
        
                % the number of testing samples.
                nTest = 1;

                % the range of testing samples
                teRange = 2:2;
            else
                if strcmp(dataset, 'bee')
                    trRange = 1:68;
                    
                    nTest = 50;
                    
                    teRange = 69:118;
                else
                    if strcmp(dataset,'fish')
                        trRange = 1:69;
                        nTest = 60;
                        teRange = 70:129;
                    end
                end
            end
        end
    end
end

% the cross-validation number
folders = 5;
if strcmp(dataset,'ucsd') || strcmp(dataset,'mall') || strcmp(dataset, 'bird')
    folders = 1;
end

% the number of all images, including testing and training ones.
totalNum = 200;
if ~strcmp(dataset,'cell')
    totalNum = 2000;
end
if strcmp(dataset, 'bird')
    totalNum = 3;
end
if strcmp(dataset, 'bee')
    totalNum = 118;
end
if strcmp(dataset, 'fish')
    totalNum = 129;
end
% assert(totalNum == (length(trRange) + length(teRange)));

% regularized parameter for controlling the counting results
lambda = 1e-3;

% the dimensions of feature vectors
vd = region.size^2;

% the maximum number in an anchored neighborhood
num_of_instances_locally = 4096;

% the amount of nearest neighbors
Kt = 32;

pca_coefficients = 0.99;

opt = 7;
switch opt
    case 6
        p = 1.4;
    case 7
        sd = 2.4;
    case 8
        sd = 11.4;
    case 9
        sd = 1.4;
    case 4
        addpath(genpath([pwd filesep 'ompbox']));
end

KernelDefinitation;
% the choice of constraint
% 1: non
% 2: energy
% 3: non-negativeness
% 4: sparsity
% 5: nonnegative sparsity
% 6: polynomial kernel
% 7: rbf kernel
% 8: laplacian kernel
% 9: exponential kernel
% 0: relu kernel

kernel = {};
scalar_ker = {};
if opt <= 10 && opt > 5
    if opt < 10
        kernel = kernels{opt};
        scalar_ker = scalar_kers{opt};
    else
        kernel = kernels{0};
        scalar_ker = scalar_kers{0};
    end
end

if strcmp(dataset,'ucsd')
    dataset = 'ucsd_hf';
end

if strcmp(dataset,'mall')
%     dataset = 'mall_x2';
end

if strcmp(dataset,'bee')
    dataset = 'bee_mid';
end

if strcmp(dataset,'cell')
    dataset = 'cell_dsift';
end

if strcmp(dataset,'bird')
    dataset = 'bird_dsift';
end

% quick access
quickFileName = [pwd filesep 'Data' filesep 'example_' dataset '.mat'];

% saving folder for synthetic images generated by algorithm
if ~exist('sd','var')
    sd = 0;
end
saveName = ['EVOC_' dataset '+' ...
        datestr(now,'yyyy-mm-dd_HH-MM-SS') ...
    '_P+' num2str(region.size) '_teS+' num2str(region.teStep) '_trS+' ...
    num2str(region.trStep) '_train+' num2str(nTrain) '_atoms' ...
    num2str(region.types) '_cross+' num2str(folders) '_ker+' num2str(opt) '_sd+' num2str(sd)];
if strcmp(dataset, 'ucsd')
    saveName = ['EVOC_' dataset '_M+' modeName{dataset_Mode} '+' ...
        datestr(now,'yyyy-mm-dd_HH-MM-SS') ...
    '_P+' num2str(region.size) '_teS+' num2str(region.teStep) '_trS+' ...
    num2str(region.trStep) '_train+' num2str(nTrain) '_atoms' ...
    num2str(region.types) '_cross+' num2str(folders) '_ker+' num2str(opt) '_sd+' num2str(sd)];
end