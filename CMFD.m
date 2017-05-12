%% Initialization and Segmentation
DEBUG = true;
path = 'data/MICC_F600/_r30horses.png';
if DEBUG == false
    [file, full_path] = uigetfile('*', 'Pick a picture');
    path = strcat(full_path, file);
end

%% Read an image and transform into single type
copy_img = imread(path);
Is = im2single(copy_img);
[c_r, c_c, c_channel] = size(Is);
%% Parameter settings for SLIC-based segmentation (parameters are chosen for MICC-F600 dataset)

r_size = 50;
reg = 0.8;
img_size = c_r * c_c;

if img_size > 3000*2000
    r_size = 200;
elseif img_size > 2000*1000
    r_size = 150;
elseif img_size > 1000*600
    r_size = 100;
else
    r_size = 50;
end

SEGMENTS = vl_slic(Is, r_size, reg);
%% Visualize segmentation result
%VisSegmentation(copy_img,SEGMENTS);

%% First Matching (Robust matching)
%% 1. Keypoint Extraction (SIFT)
% Compute gray scale image
Ig = rgb2gray(copy_img);

% Compute SIFT features
[f,d] = vl_sift(single(Ig));

% Patch - keypoints map (patch_num: [keypoint1, keypoint2, ...])
patchToKeypoints = containers.Map('KeyType','int32','ValueType','any');

% Initialize container
maxPatch = max(SEGMENTS(:));
for k = 0:maxPatch
    if ~(isKey(patchToKeypoints, k))
        patchToKeypoints(k) = [];
    end
end

% Assign keypoints to each patch
[feature_elements, num_of_features] = size(f);
for n_f = 1:num_of_features
    feature_x = f(1,n_f);
    feature_y = f(2,n_f);
    feature_d = double(d(:,n_f));
    feature_d = feature_d / norm(feature_d);
    cur_seg = SEGMENTS(int32(feature_y),int32(feature_x));
    patchToKeypoints(cur_seg) = [patchToKeypoints(cur_seg); transpose(feature_d)];
end

threshold = 10 * num_of_features / (maxPatch+1); % including [0-maxPatch]
% Patch matching using k-d tree
matched_list = [];
for p1 = 0:maxPatch-1
    keypoints_p1 = patchToKeypoints(p1); % [descriptor 1; descriptor 2; ...]
    num_of_keypoints_p1 = size(keypoints_p1, 1);
    point1_set = transpose(keypoints_p1);
    for p2 = p1+1:maxPatch
        num_matched_keypoints = 0;
        keypoints_p2 = patchToKeypoints(p2); % [descriptor 1; descriptor 2; ...]
        num_of_keypoints_p2 = size(keypoints_p2, 1);
        point2_set = transpose(keypoints_p2);
        kdtree = vl_kdtreebuild(point2_set);
        [index,distance] = vl_kdtreequery(kdtree, point2_set, point1_set, 'NumNeighbors', 10);
        
        for k = 1:num_of_keypoints_p1
            for cand = 1:10
                if distance(cand, k) < 0.16
                    num_matched_keypoints = num_matched_keypoints + 1;
                end
            end
        end
        if num_matched_keypoints > threshold
            display(num_matched_keypoints);
        end
    end
end

%% Second Matching (Iteration)
%% 
