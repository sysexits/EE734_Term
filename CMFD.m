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
    SEGMENTS = vl_slic(Is, r_size, reg);
elseif img_size > 2000*1000
    r_size = 150;
    SEGMENTS = vl_slic(Is, r_size, reg);
elseif img_size > 1000*600
    r_size = 100;
    SEGMENTS = vl_slic(Is, r_size, reg);
else
    r_size = 50;
    SEGMENTS = vl_slic(Is, r_size, reg);
end

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
    cur_seg = SEGMENTS(int32(feature_y),int32(feature_x));
    patchToKeypoints(cur_seg) = [patchToKeypoints(cur_seg); transpose(feature_d)];
end

clearvars feature_d;

N = 10; % paramter of K-NN
threshold = N * num_of_features / (maxPatch+1); % including [0-maxPatch]
% Patch matching using k-d tree
matching_cases = zeros(maxPatch+1, maxPatch+1);
for p1 = 0:maxPatch
    keypoints_p1 = patchToKeypoints(p1); % [descriptor 1; descriptor 2; ...]
    num_of_keypoints_p1 = size(keypoints_p1, 1);
    point1_set = transpose(keypoints_p1);
    num_matched_keypoints = 0;
    keypoints_remain_ps = zeros(128, num_of_features - num_of_keypoints_p1);
    index_patch_keynum = zeros(num_of_features - num_of_keypoints_p1, 2);
    index = 1;
    matching_state = zeros(maxPatch+1,2);
    
    for p2 = 0:maxPatch
        if (p1 ~= p2)
            keypoints_p2 = patchToKeypoints(p2); % [descriptor 1; descriptor 2; ...]
            num_of_keypoints_p2 = size(keypoints_p2, 1);
            point2_set = transpose(keypoints_p2);
            for k2 = 1:num_of_keypoints_p2
                keypoints_remain_ps(:, index) = point2_set(:,k2);
                index_patch_keynum(index,:) = [p2, k2];
                index = index + 1;
            end
        end
    end
    
    kdtree = vl_kdtreebuild(keypoints_remain_ps);
    [index,distance] = vl_kdtreequery(kdtree, keypoints_remain_ps, point1_set, 'NumNeighbors', 10);
    
    local_matching = 0;
    for k = 1:num_of_keypoints_p1
        for cand1 = 1:9
            if distance(cand1, k) / distance(cand1+1,k) < 0.4
                num_matched_keypoints = num_matched_keypoints + 1;
                local_matching = local_matching + 1;
                p2_matched_index = index(cand1,k);
                p2_num = index_patch_keynum(p2_matched_index, 1);
                matching_state(p2_num + 1, 1) = p2_num;
                matching_state(p2_num + 1, 2) = matching_state(p2_num + 1, 2) + 1;
            end
        end
    end
    
    if (local_matching > threshold)
        display([p1 local_matching]);
        display(matching_state);
    end
    
    clearvars point1_set;
    clearvars point2_set;
    clearvars keypoints_remain_ps;
    clearvars index_patch_keynum;
end

%{
kdtree = vl_kdtreebuild(point2_set);
            [index,distance] = vl_kdtreequery(kdtree, point2_set, point1_set, 'NumNeighbors', 10);
            
            matching_state(p2 + 1, 1) = p2;
            local_matching = 0;
            for k = 1:num_of_keypoints_p1
                for cand1 = 1:9
                    for cand2 = cand1 + 1:10
                        if distance(cand1, k) / distance(cand2,k) < 0.04
                            num_matched_keypoints = num_matched_keypoints + 1;
                            local_matching = local_matching + 1;
                        end
                    end
                end
            end
            matching_state(p2 + 1, 2) = local_matching;
%}

%% Second Matching (Iteration)
%% 
