function [ table_of_drirs ] = RandomizeSimulationLoop( rand_room, mic_array, n_drirs, dist_s_to_r )

% rounding safety offset: dealing with double may cause s_pos to be
% located outside the room due to rounding
rso = 0.02;

row = n_drirs;
rooms = struct(                         ...
    'dim', cell(1,row),                 ...
    's_pos', cell(1,row),               ...
    'r_pos', cell(1,row),               ...
    'absorption', cell(1,row),          ...
    'scattering', cell(1,row)           ...
);

%% 1 Define table to store each DRIR in a row
table_of_drirs = struct(                ...
    'FS', cell(1,row),                  ...
    'limit', cell(1,row),               ...
    'ac', cell(1,row),                  ...
    'Nsft', cell(1,row),                ...
    'nNodes', cell(1,row),              ...
    'radius', cell(1,row),              ...
    'impulseResponses', cell(1,row),    ...
    'quadratureGrid', cell(1,row),      ...
    'averageAirTemp', cell(1,row),      ...
    'downsample', cell(1,row),          ...
    'irOverlay', cell(1,row),           ...
    'Position', cell(1,row),            ...
    'meta', cell(1,row)                 ...
    );

%% 2 Randomize
% 2.1 Dimensions
diff = rand_room.dim(:, :, 2) - rand_room.dim(:, :, 1);
diff_mat =  repmat(diff, row, 1);
low_bound = rand_room.dim(:, :, 1);
low_bound_mat = repmat(low_bound, row, 1);
dims = diff_mat.*rand(row, 3) + low_bound_mat; % row x 3 random within boudaries
dims_cell = mat2cell(dims, ones(1, row), 3);
[rooms.dim] = dims_cell{:};

if dist_s_to_r < mic_array.radius
    disp('Minimum distance of source to receiver center must be at least the array radius');
    dist_s_to_r = mic_array.radius;
end

% receiver offset
r_offset = dist_s_to_r + 2 * rso;

% 2.2 r_pos - avoid wall intersection:
% Since r_pos is center of mic array, there must be an offset of the
% array radius 1x at beginning and 1x at the end in each direction (x,y,z)
% so that mic array does not intersect the room's walls
r_pos = (dims - (2 * r_offset)).*rand(row, 3) + r_offset;
r_pos_cell = mat2cell(r_pos, ones(1, row), 3);
[rooms.r_pos] = r_pos_cell{:};

% 2.3 s_pos - avoid intersection with mic array and provide distance:
% Choose randomly from two sets of random positions - in front of and
% behind the mic array, or between origin and array as well as between
% array end wall, with offset considering dist source to receiver

in_front_behind_r = zeros(row, 3, 2);   % 2 before or behind receiver
in_front_behind_r(:,:,1) = (r_pos - dist_s_to_r -2*rso).*rand(row, 3) + rso;
in_front_behind_r(:,:,2) = (dims - r_pos - dist_s_to_r - 2*rso).*rand(row, 3) ...
                            + r_pos + dist_s_to_r + rso;
% Randomly choose between the position before and behind the array
ids = round(rand(row, 3)) + 1;
n = size(ids, 1) * size(ids, 2);
idx = [1:n] +(ids(1:n)-1) * n;          % linear indices
s_pos = reshape(in_front_behind_r(idx), size(ids, 1), size(ids, 2));
s_pos_cell = mat2cell(s_pos, ones(1, row), 3);
[rooms.s_pos] = s_pos_cell{:};

% 2.4 absorption
diff = rand_room.absorption(:,:,2) - rand_room.absorption(:,:,1);
diff_mat = repmat(diff, 1, 1, row);             % [wall x freqs x row]
low_bound = rand_room.absorption(:,:,1);
low_bound_mat = repmat(low_bound, 1, 1, row);   % [wall x freqs x row]
absorptions = diff_mat.*rand(size(diff_mat)) + low_bound_mat;
[m,n,k] = size(absorptions);
absorptions_cell = mat2cell(absorptions,m,n,ones(k,1));
[rooms.absorption] = absorptions_cell{:};

% 2.5 scattering
% 2.4 absorption
diff = rand_room.scattering(:,:,2) - rand_room.scattering(:,:,1);
diff_mat = repmat(diff, 1, 1, row);             % [wall x freqs x row]
low_bound = rand_room.scattering(:,:,1);
low_bound_mat = repmat(low_bound, 1, 1, row);   % [wall x freqs x row]
scatterings = diff_mat.*rand(size(diff_mat)) + low_bound_mat;
[m,n,k] = size(scatterings);
scatterings_cell = mat2cell(scatterings,m,n,ones(k,1));
[rooms.scattering] = scatterings_cell{:};

%% 3 Simulate each of the DRIRs
for i = 1:n_drirs
    room = rooms(i);
    room.freqs = rand_room.freqs;
    disp(['------------> ', num2str(i)]);
    table_of_drirs(i) = SimulateDrir(room, mic_array);
end


end