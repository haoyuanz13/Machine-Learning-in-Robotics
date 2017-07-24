%% read data 
im = load('camX.mat');  % image set
vicon = load('imuRawX.mat'); % only used to extract time information

euler = load('Estimated_UKF_EulerAngle.mat'); % UKF estimate result
euler = euler.f_EA;
ea_reverse = [euler(:, 3), euler(:, 2), euler(:, 1)];
rot = eul2rotm(ea_reverse); % estimate rotation matrix
%% normalize timeline 
ts_im = im.ts;
cam = im.cam;
ts_rot = vicon.ts;

[tt, timeline] = size(ts_im);
ts_im = ts_im - ts_im(1);
ts_rot = ts_rot - ts_rot(1);

rot_index = zeros(1, timeline);
for t = 1 : timeline
    pivot = ts_im(t);
    [val, index] = min(abs(ts_rot - pivot));
    rot_index(t) = index;
end
%% image coordinate to world
row = 240;
col = 320;
total = row * col;
r = 1 : row;
c = 1 : col;
[cc, rr] = meshgrid(c, r);

rr = reshape(rr, [76800, 1]);
cc = reshape(cc, [76800, 1]);
ori_pos = [rr, cc];

longitude = (160 - cc) .* (pi / 3 / 320); % set the center position in image as original point
latitude = (120 - rr) .* (pi / 4 / 240);

[x, y, z] = sph2cart(longitude, latitude, 1);
pos = [x'; y'; z'];
%% stitch image
im_sti = uint8(zeros(960, 1920, 3));
figure
% choose suitable interval to construct panorma
for k = 1 : timeline 
    img = cam(:, :, :, k);
    ind = rot_index(k);
    
    roti = rot(:, :, ind);
    world = (roti * pos)';
    
    [az, ele, r] = cart2sph(world(:, 1), world(:, 2), world(:, 3));
    
    theta = (az + pi)./ (pi / 3 / 320);
    height = 961 - (ele + (pi / 2)) ./ (pi / 4 / 240);
    new_pos = [floor(height), floor(theta)];
    
    for i = 1 : total
        newr = new_pos(i, 1);
        newc = new_pos(i, 2);
        if newr <= 0 || newc <= 0
            continue;
        end
        orir = ori_pos(i, 1);
        oric = ori_pos(i, 2);
        if im_sti(newr, newc, :) == 0
            im_sti(newr, newc, :) = img(orir, oric, :);
        end
    end
    imshow(im_sti);
end

    
    






