% Evaluate accuracy of segmentation (modified from VOCEVALSEG)
% prints out the per class and overall segmentation accuracies.
% Accuracies are given using the intersection/union metric:
%   true positives / (true positives + false positives + false negatives)

% folder path
GT_path = '/mnt/3TB/Datasets/OutdoorSceneTest300/OutdoorSceneTest300_anno';
% res_path = '/home/xtwang/projects/SFT-GAN/data/save_byteimg';
res_path = '/home/xtwang/remote/46/DATA/xtwang/Projects/SFT-GAN/data/save_byteimg';

nCls = 8; % with background
num = nCls;
confcounts = zeros(num);
count=0;
filepaths = dir(fullfile(GT_path,'*.png'));

for i=1:length(filepaths)
    imname = filepaths(i).name;
    base_name = imname(1:end-4);
    % ground truth label file
    gtim = imread(fullfile(GT_path, filepaths(i).name));
    gtim = double(gtim);
    % results file
    resim = imread(fullfile(res_path, [base_name, '_bic.png']));
    resim = double(resim);

%     rm_row = size(resim,1) - size(gtim,1);
%     rm_col = size(resim,2) - size(gtim,2);
%     if rm_row > 0,
%         resim(end-rm_row+1:end, :)=[];
%     else
%         gtim(end+rm_row+1:end, :)=[];
%     end
%     if rm_col > 0,
%         resim(:,end-rm_col+1:end)=[];
%     else
%         gtim(:,end+rm_col+1:end)=[];
%     end

    % Check validity of results image
    maxlabel = max(resim(:));
    if (maxlabel>nCls)
        error('Results image ''%s'' has out of range value %d (the value should be <= %d)',imname,maxlabel,nCls);
    end
    szgtim = size(gtim); szresim = size(resim);
    if any(szgtim~=szresim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end
    %pixel locations to include in computation
    locs = gtim<255;
    % joint histogram
    sumim = 1+gtim+resim*num;
    hs = histc(sumim(locs),1:num*num);
    count = count + numel(find(locs));
    confcounts(:) = confcounts(:) + hs(:);
end

% confusion matrix - first index is true label, second is inferred label
conf = zeros(num);
conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
% rawcounts = confcounts;

% Percentage correct labels measure
overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
fprintf('Percentage of pixels correctly labelled overall: %6.2f%%\n',overall_acc);

accuracies = zeros(nCls,1);
fprintf('Accuracy for each class (intersection/union measure)\n');
cls_names = {'background', 'sky', 'water', 'grass', 'mountain', 'building', 'plant', 'animal'};
% 0, background
% 1, sky
% 2, water
% 3, grass
% 4, mountain
% 5, building
% 6, plant
% 7, animal
% 8, void
for j=1:num
   gtj=sum(confcounts(j,:));
   resj=sum(confcounts(:,j));
   gtjresj=confcounts(j,j);
   % The accuracy is: true positive / (true positive + false positive + false negative)
   % which is equivalent to the following percentage:
   accuracies(j)=100*gtjresj/(gtj+resj-gtjresj);

   fprintf('  %2d - %10s: %6.2f%%\n',j-1,cls_names{j}, accuracies(j));
end
accuracies = accuracies(1:end);
avacc = mean(accuracies);
fprintf('-------------------------\n');
fprintf('Average accuracy: %6.2f%%\n',avacc);