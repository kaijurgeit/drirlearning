function [ drir ] = SimulateDrir( room, mic_array )
    %% 1 DRIR struct
    rate = 48000;

    drir = struct;
    drir.FS = 48000;

    drir.limit = mic_array.amp_limit;       % amplification Limit (was 30, Keep the result numerical stable at low frequencies)
    drir.ac = mic_array.config;             % array configuration
    drir.Nsft = mic_array.order;            % order
    drir.nNodes = mic_array.n_nodes;        % order drir.Nsft corresponds to certain drir.nNodes
    drir.radius = mic_array.radius;         % fix drir.radius --> certain upper frequency
    f_a = drir.Nsft*343/(2*pi*drir.radius);
    disp(f_a);


    %% 2 Setup room_setup and Microphone Array
    % 2.1 Generate Lebedev Grid
    [mic_array.grid, mic_array.n_nodes, ~] = sofia_lebedev(mic_array.n_nodes, 0);
    RIR = zeros(2500, rate);

    % 2.2 Shift Lebedev grid in carteesian cordiantes
    [x_, y_, z_] = sph2cart(mic_array.grid(:,1), pi/2 - mic_array.grid(:,2), mic_array.radius);
    x = x_ + room.r_pos(1);
    y = y_ + room.r_pos(2);
    z = z_ + room.r_pos(3);

    % 2.3 Add Source and receivers
    sources = AddSource([],'Type','omnidirectional', 'Location', room.s_pos);
    % TODO: cardioid
    receivers = AddReceiver([], 'Type','cardioid','Location',[x(1),y(1),z(1)],'Orientation',[rad2deg(mic_array.grid(1,1)),rad2deg(pi/2 - mic_array.grid(1,2)),0],'UnCorNoise',true);
    receivers.Fs = rate;

    for idx = 2:mic_array.n_nodes
        % orientation: yaw = azimuth / pitch = elev
        receivers = AddReceiver(receivers, 'Type','cardioid','Location',[x(idx),y(idx),z(idx)],'Orientation',[rad2deg(mic_array.grid(idx,1)),rad2deg(pi/2 - mic_array.grid(idx,2)),0],'UnCorNoise',true);
        % PlotSimSetup(sources,receivers,room_setup)
        disp(['processing IR ' num2str(idx) '/' num2str(mic_array.n_nodes)]);
    end

    room_setup = SetupRoom('Dim',room.dim,'Freq',room.freqs,'Absorption',room.absorption,'Scattering',room.scattering);
    options = MCRoomSimOptions('Fs', rate);

    %% 3 Simulate DRIR
    % 3.1 room_setup Simulation (TODO: RAM usage!)
    tmp = RunMCRoomSim(sources, receivers, room_setup, options);
    impulse_response_length = length(tmp{1});
    for idx = 1:mic_array.n_nodes
        RIR(idx,1:impulse_response_length) = tmp{idx};
    end
    % PlotSimSetup(sources,receivers,room_setup)

    % 3.2 Make sofia-object
    drir.impulseResponses = RIR;
    [~, timeSamples] = size(RIR);
    drir.quadratureGrid = mic_array.grid;
    drir.averageAirTemp = 20;
    drir.downsample = 1;
    drir.irOverlay = timeSamples;
    drir.Position = room.r_pos;

    drir.meta.name = 'meta';
    drir.meta.dim = room.dim;
    drir.meta.s_pos = room.s_pos;
    drir.meta.r_pos = room.r_pos;
    drir.meta.freqs = room.freqs;
    drir.meta.absorption = room.absorption;
    drir.meta.scattering = room.scattering;
end

