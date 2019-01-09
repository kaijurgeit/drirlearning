%-----------------------------------------------------
% Simulate shoebox rand_room and render one directional RIR
% 1 Configure Room and Microphone Array
% 2 Start simulation loop with randomized attributes
% 3 Save DRIR table to disk
%-----------------------------------------------------

%% 1 Configuration
rootPath = pwd;
idcs = strfind(rootPath, '\');
mainPath = rootPath(1:idcs(end-2)-1);
audioPath = 'D:\Workspace\_Output\Matlab';
isCenter = true;

% 1.1 Random Room with lower and upper boundaries as range of random
rand_room = struct();
rand_room.dim = [5.0, 5.0, 2.5];                         % lower
rand_room.dim(:,:,2) = [10.0, 10.0, 4.0];                % upper
rand_room.freqs = [100, 200, 400, 800, 1600, 3200, 6400];% constant
rand_room.absorption = ones(6,7) .* 0.5;                 % lower
rand_room.absorption(:,:,2) = ones(6,7) .* 1.0;          % upper
rand_room.scattering = ones(6,7) .* 0.01;                % lower
rand_room.scattering(:,:,2) = ones(6,7) .* 1.0;          % upper

% 1.2 Microphone Array
mic_array = struct();
mic_array.amp_limit = 30;           % amplification Limit (was 30, Keep the result numerical stable at low frequencies)
mic_array.config = 1;               % array configuration
mic_array.order = 4;                % order
mic_array.n_nodes = 38;             % order drir.Nsft corresponds to certain drir.nNodes
mic_array.radius = 0.042;           % fix drir.radius --> certain upper frequency

%% 2 Start simulation loop with randomized attributes
tic;
table_of_drirs = RandomizeSimulationLoop( rand_room, mic_array, 10, 0 );
toc;
save([audioPath 'table_of_drirs_mockup.mat'], 'table_of_drirs');