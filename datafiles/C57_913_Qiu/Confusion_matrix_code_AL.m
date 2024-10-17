% Input options
plot_alltrials = true; %Set true to plot confusion  matricess using all trials toguether
plot_subsets = true; %Set true to plot confusion matrices of trials with shared start locations

plot_chance = true; %Set true to plot confusion matrices for shuffle data. 
chance_repeat = 5; %Input chance repeat you want to plot

quality_floor = 1; %Set minimun number of trials needed in each subset to generate the corresponding plot

% Input size of spatial bins used in decoding (in cm)
pbin_size = 10;

% Input reward zone limits (in cm)
rzstart = 460;
rzend = 495;

% Calculate and plot confusion matrix
positions = unique(decoder_targets_lgt(~isnan(decoder_targets_lgt)));

dec_pos_count_lgt = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
dec_pos_count_drk = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
Norm_dec_pos_count_lgt = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
Norm_dec_pos_count_drk = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
dec_pos_count_drk_lgttrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
Norm_dec_pos_count_drk_lgttrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
dec_pos_count_lgt_drktrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
Norm_dec_pos_count_lgt_drktrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);

if plot_alltrials == true
    r_positions_lgt = reshape(decoder_targets_lgt, 1, []);
    d_positions_lgt = reshape(decoded_position_lgt, 1, []);
    r_positions_drk = reshape(decoder_targets_drk, 1, []);
    d_positions_drk = reshape(decoded_position_drk, 1, []);
    
    for iRpos = 1:length(positions)
        Rpos = positions(iRpos);
        dpos_lgt = d_positions_lgt(find(r_positions_lgt == Rpos));
        dpos_drk = d_positions_drk(find(r_positions_drk == Rpos));
        
        for iDpos = 1:length(positions)
            Dpos = positions(iDpos);
            dec_pos_count_lgt(Rpos, Dpos) = sum(dpos_lgt == Dpos);
            dec_pos_count_drk(Rpos, Dpos) = sum(dpos_drk == Dpos);
        end
        Rpos_sum_lgt = sum(dec_pos_count_lgt(Rpos, :), 'omitnan');
        Norm_dec_pos_count_lgt(Rpos, :) =  dec_pos_count_lgt(Rpos, :)./Rpos_sum_lgt;
        Rpos_sum_drk = sum(dec_pos_count_drk(Rpos, :), 'omitnan');
        Norm_dec_pos_count_drk(Rpos, :) =  dec_pos_count_drk(Rpos, :)./Rpos_sum_drk;
    end
    
    figure
    subplot(1, 2, 1);
    p1 = heatmap(Norm_dec_pos_count_lgt, 'Colormap', parula);
    xlabel('Decoded postion (10cm bins)');
    ylabel('VR position (10cm bins)');
    title('Light confusion matrix');
    %Find hetamap handles to draw lines where landmarks are
    origState = warning('query', 'MATLAB:structOnObject');
    cleanup = onCleanup(@()warning(origState));
    warning('off','MATLAB:structOnObject')
    S = struct(p1);
    ax = S.Axes;
    clear('cleanup')
    hm.GridVisible = 'off'; % Set heatmap grid invisible.
    % Draw red dasshed lines flanking landmarks
    col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
    xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
    xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
    yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
    yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
    
    subplot(1, 2, 2);
    p2 = heatmap(Norm_dec_pos_count_drk, 'Colormap', parula);
    xlabel('Decoded postion (10cm bins)');
    ylabel('VR position (10cm bins)');
    title('Dark confusion matrix');
    %Find hetamap handles to draw lines where landmarks are
    origState = warning('query', 'MATLAB:structOnObject');
    cleanup = onCleanup(@()warning(origState));
    warning('off','MATLAB:structOnObject')
    S = struct(p2);
    ax = S.Axes;
    clear('cleanup')
    hm.GridVisible = 'off'; % Set heatmap grid invisible.
    % Draw red dasshed lines flanking landmarks
    col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
    xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
    xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
    yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
    yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
    
    if plot_drk_lgttrained == true
        d_positions_drk_lgttrained = reshape(decoded_position_drk_lgttrained, 1, []);
        
        for iRpos = 1:length(positions)
            Rpos = positions(iRpos);
            dpos_drk_lgttrained = d_positions_drk_lgttrained(find(r_positions_drk == Rpos));
            
            for iDpos = 1:length(positions)
                Dpos = positions(iDpos);
                dec_pos_count_drk_lgttrained(Rpos, Dpos) = sum(dpos_drk_lgttrained == Dpos);
            end
            Rpos_sum_drk_lgttrained = sum(dec_pos_count_drk_lgttrained(Rpos, :), 'omitnan');
            Norm_dec_pos_count_drk_lgttrained(Rpos, :) =  dec_pos_count_drk_lgttrained(Rpos, :)./Rpos_sum_drk_lgttrained;
        end
        
        figure
        p3 = heatmap(Norm_dec_pos_count_drk_lgttrained, 'Colormap', parula);
        xlabel('Decoded postion (10cm bins)');
        ylabel('VR position (10cm bins)');
        title('Dark (lgt trained) confusion matrix');
        %Find hetamap handles to draw lines where landmarks are
        origState = warning('query', 'MATLAB:structOnObject');
        cleanup = onCleanup(@()warning(origState));
        warning('off','MATLAB:structOnObject')
        S = struct(p3);
        ax = S.Axes;
        clear('cleanup')
        hm.GridVisible = 'off'; % Set heatmap grid invisible.
        % Draw red dasshed lines flanking landmarks
        col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
        xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
    end
    
    if plot_lgt_drktrained == true
        d_positions_lgt_drktrained = reshape(decoded_position_lgt_drktrained, 1, []);
        
        for iRpos = 1:length(positions)
            Rpos = positions(iRpos);
            dpos_lgt_drktrained = d_positions_lgt_drktrained(find(r_positions_lgt == Rpos));
            
            for iDpos = 1:length(positions)
                Dpos = positions(iDpos);
                dec_pos_count_lgt_drktrained(Rpos, Dpos) = sum(dpos_lgt_drktrained == Dpos);
            end
            Rpos_sum_lgt_drktrained = sum(dec_pos_count_lgt_drktrained(Rpos, :), 'omitnan');
            Norm_dec_pos_count_lgt_drktrained(Rpos, :) =  dec_pos_count_lgt_drktrained(Rpos, :)./Rpos_sum_lgt_drktrained;
        end
        
        figure
        p4 = heatmap(Norm_dec_pos_count_lgt_drktrained, 'Colormap', parula);
        xlabel('Decoded postion (10cm bins)');
        ylabel('VR position (10cm bins)');
        title('Light (drk trained) confusion matrix');
        %Find hetamap handles to draw lines where landmarks are
        origState = warning('query', 'MATLAB:structOnObject');
        cleanup = onCleanup(@()warning(origState));
        warning('off','MATLAB:structOnObject')
        S = struct(p4);
        ax = S.Axes;
        clear('cleanup')
        hm.GridVisible = 'off'; % Set heatmap grid invisible.
        % Draw red dasshed lines flanking landmarks
        col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
        xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
    end       
end

% Calculate and plot confusion matrices of trial subsets
if plot_subsets == true
    
    % Light subsets
    figure
    % Discretize start locations if needed
    if discrete_start == false 
        start_pos_bins = (1:4:40);
        DtStart_tag_lgt = discretize(tStart_tag_lgt, start_pos_bins);
    end
    
    % Filter subsets to plot
    start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
    
    for i = 1:numel(start_positions)
        start_pos = start_positions(i);
        if ~isempty(find(DtStart_tag_lgt == start_pos, 1))
            trials_subset = find(DtStart_tag_lgt == start_pos);
            if length(trials_subset) >= quality_floor
                decoder_targets_lgt_subset = decoder_targets_lgt(trials_subset, :);
                decoded_position_lgt_subset = decoded_position_lgt(trials_subset, :);
            else
                decoder_targets_lgt_subset = NaN;
                decoded_position_lgt_subset = NaN;
            end
            
            r_positions_lgt = reshape(decoder_targets_lgt_subset, 1, []);
            d_positions_lgt = reshape(decoded_position_lgt_subset, 1, []);
            
            for iRpos = 1:length(positions)
                Rpos = positions(iRpos);
                dpos_lgt = d_positions_lgt(find(r_positions_lgt == Rpos));
                
                for iDpos = 1:length(positions)
                    Dpos = positions(iDpos);
                    dec_pos_count_lgt(Rpos, Dpos) = sum(dpos_lgt == Dpos);
                end
                Rpos_sum_lgt = sum(dec_pos_count_lgt(Rpos, :), 'omitnan');
                Norm_dec_pos_count_lgt(Rpos, :) =  dec_pos_count_lgt(Rpos, :)./Rpos_sum_lgt;
            end
            
            subplot(2, round(length(start_positions)/2), i);
            p1 = heatmap(Norm_dec_pos_count_lgt, 'Colormap', parula);
            xlabel('Decoded postion (10cm bins)');
            ylabel('VR position (10cm bins)');
            title('Light confusion matrix');
            %Find hetamap handles to draw lines where landmarks are
            origState = warning('query', 'MATLAB:structOnObject');
            cleanup = onCleanup(@()warning(origState));
            warning('off','MATLAB:structOnObject')
            S = struct(p1);
            ax = S.Axes;
            clear('cleanup')
            hm.GridVisible = 'off'; % Set heatmap grid invisible.
            % Draw red dasshed lines flanking landmarks
            col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
            xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        end
    end
    
    %Dark subsets
    figure
    % Discretize start locations if needed
    if discrete_start == false 
        start_pos_bins = (1:4:40);
        DtStart_tag_drk = discretize(tStart_tag_drk, start_pos_bins);
    end
    
    % Filter subsets to plot
    start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
    
    for i = 1:numel(start_positions)
        start_pos = start_positions(i);
        if ~isempty(find(DtStart_tag_drk == start_pos, 1))
            trials_subset = find(DtStart_tag_drk == start_pos);
            if length(trials_subset) >= quality_floor
                decoder_targets_drk_subset = decoder_targets_drk(trials_subset, :);
                decoded_position_drk_subset = decoded_position_drk(trials_subset, :);
            else
                decoder_targets_drk_subset = NaN;
                decoded_position_drk_subset = NaN;
            end
            r_positions_drk = reshape(decoder_targets_drk_subset, 1, []);
            d_positions_drk = reshape(decoded_position_drk_subset, 1, []);
            
            for iRpos = 1:length(positions)
                Rpos = positions(iRpos);
                dpos_drk = d_positions_drk(find(r_positions_drk == Rpos));
                
                for iDpos = 1:length(positions)
                    Dpos = positions(iDpos);
                    dec_pos_count_drk(Rpos, Dpos) = sum(dpos_drk == Dpos);
                end
                Rpos_sum_drk = sum(dec_pos_count_drk(Rpos, :), 'omitnan');
                Norm_dec_pos_count_drk(Rpos, :) =  dec_pos_count_drk(Rpos, :)./Rpos_sum_drk;
            end
            
            subplot(2, round(length(start_positions)/2), i);
            p1 = heatmap(Norm_dec_pos_count_drk, 'Colormap', parula);
            xlabel('Decoded postion (10cm bins)');
            ylabel('VR position (10cm bins)');
            title('Dark confusion matrix');
            %Find hetamap handles to draw lines where landmarks are
            origState = warning('query', 'MATLAB:structOnObject');
            cleanup = onCleanup(@()warning(origState));
            warning('off','MATLAB:structOnObject')
            S = struct(p1);
            ax = S.Axes;
            clear('cleanup')
            hm.GridVisible = 'off'; % Set heatmap grid invisible.
            % Draw red dasshed lines flanking landmarks
            col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
            xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        end
    end
    
    % Dark subsets trained with light data
    if plot_drk_lgttrained == true
        figure
        % Discretize start locations if needed
        if discrete_start == false 
            start_pos_bins = (1:4:40);
            DtStart_tag_drk = discretize(tStart_tag_drk, start_pos_bins);
        end
        
        % Filter subsets to plot
        start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
        
        for i = 1:numel(start_positions)
            start_pos = start_positions(i);
            if ~isempty(find(DtStart_tag_drk == start_pos, 1))
                trials_subset = find(DtStart_tag_drk == start_pos);
                if length(trials_subset) >= quality_floor
                    decoder_targets_drk_subset = decoder_targets_drk(trials_subset, :);
                    decoded_position_drk_lgttrained_subset = decoded_position_drk_lgttrained(trials_subset, :);
                else
                    decoder_targets_drk_subset = NaN;
                    decoded_position_drk_lgttrained_subset = NaN;
                end
                r_positions_drk = reshape(decoder_targets_drk_subset, 1, []);
                d_positions_drk_lgttrained = reshape(decoded_position_drk_lgttrained_subset, 1, []);
                
                for iRpos = 1:length(positions)
                    Rpos = positions(iRpos);
                    dpos_drk_lgttrained = d_positions_drk_lgttrained(find(r_positions_drk == Rpos));
                    
                    for iDpos = 1:length(positions)
                        Dpos = positions(iDpos);
                        dec_pos_count_drk_lgttrained(Rpos, Dpos) = sum(dpos_drk_lgttrained == Dpos);
                    end
                    Rpos_sum_drk_lgttrained = sum(dec_pos_count_drk_lgttrained(Rpos, :), 'omitnan');
                    Norm_dec_pos_count_drk_lgttrained(Rpos, :) =  dec_pos_count_drk_lgttrained(Rpos, :)./Rpos_sum_drk_lgttrained;
                end
                
                subplot(2, round(length(start_positions)/2), i);
                p1 = heatmap(Norm_dec_pos_count_drk_lgttrained, 'Colormap', parula);
                xlabel('Decoded postion (10cm bins)');
                ylabel('VR position (10cm bins)');
                title('Dark (lgt trained) confusion matrix');
                %Find hetamap handles to draw lines where landmarks are
                origState = warning('query', 'MATLAB:structOnObject');
                cleanup = onCleanup(@()warning(origState));
                warning('off','MATLAB:structOnObject')
                S = struct(p1);
                ax = S.Axes;
                clear('cleanup')
                hm.GridVisible = 'off'; % Set heatmap grid invisible.
                % Draw red dasshed lines flanking landmarks
                col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
                xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            end
        end
    end
    
    % Light subsets trained with dark data
    if plot_lgt_drktrained == true
        figure
        % Discretize start locations if needed
        if discrete_start == false 
            start_pos_bins = (1:4:40);
            DtStart_tag_lgt = discretize(tStart_tag_lgt, start_pos_bins);
        end
        
        % Filter subsets to plot
        start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
        
        for i = 1:numel(start_positions)
            start_pos = start_positions(i);
            if ~isempty(find(DtStart_tag_lgt == start_pos, 1))
                trials_subset = find(DtStart_tag_lgt == start_pos);
                if length(trials_subset) >= quality_floor
                    decoder_targets_lgt_subset = decoder_targets_lgt(trials_subset, :);
                    decoded_position_lgt_drktrained_subset = decoded_position_lgt_drktrained(trials_subset, :);
                else
                    decoder_targets_lgt_subset = NaN;
                    decoded_position_lgt_drktrained_subset = NaN;
                end
                
                r_positions_lgt = reshape(decoder_targets_lgt_subset, 1, []);
                d_positions_lgt_drktrained = reshape(decoded_position_lgt_drktrained_subset, 1, []);
                
                for iRpos = 1:length(positions)
                    Rpos = positions(iRpos);
                    dpos_lgt_drktrained = d_positions_lgt_drktrained(find(r_positions_lgt == Rpos));
                    
                    for iDpos = 1:length(positions)
                        Dpos = positions(iDpos);
                        dec_pos_count_lgt_drktrained(Rpos, Dpos) = sum(dpos_lgt_drktrained == Dpos);
                    end
                    Rpos_sum_lgt_drktrained = sum(dec_pos_count_lgt_drktrained(Rpos, :), 'omitnan');
                    Norm_dec_pos_count_lgt_drktrained(Rpos, :) =  dec_pos_count_lgt_drktrained(Rpos, :)./Rpos_sum_lgt_drktrained;
                end
                
                subplot(2, round(length(start_positions)/2), i);
                p1 = heatmap(Norm_dec_pos_count_lgt_drktrained, 'Colormap', parula);
                xlabel('Decoded postion (10cm bins)');
                ylabel('VR position (10cm bins)');
                title('Light (dark trained) confusion matrix');
                %Find hetamap handles to draw lines where landmarks are
                origState = warning('query', 'MATLAB:structOnObject');
                cleanup = onCleanup(@()warning(origState));
                warning('off','MATLAB:structOnObject')
                S = struct(p1);
                ax = S.Axes;
                clear('cleanup')
                hm.GridVisible = 'off'; % Set heatmap grid invisible.
                % Draw red dasshed lines flanking landmarks
                col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
                xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            end
        end
    end
end

if plot_chance == true
    % Extract decoded position in desired chance repeat
    decoded_position_iChance_lgt = squeeze(decoded_position_chance_lgt(chance_repeat, :, :, :));
    decoded_position_iChance_drk = squeeze(decoded_position_chance_drk(chance_repeat, :, :, :));
    
    % Calculate and plot confusion matrix
    positions = unique(decoder_targets_lgt(~isnan(decoder_targets_lgt)));
    
    Cdec_pos_count_lgt = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    Cdec_pos_count_drk = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    CNorm_dec_pos_count_lgt = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    CNorm_dec_pos_count_drk = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    Cdec_pos_count_drk_lgttrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    CNorm_dec_pos_count_drk_lgttrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    Cdec_pos_count_lgt_drktrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    CNorm_dec_pos_count_lgt_drktrained = nan(positions(1)+length(positions)-1, positions(1)+length(positions)-1);
    
    if plot_alltrials == true
        r_positions_lgt = reshape(decoder_targets_lgt, 1, []);
        d_positions_lgt = reshape(decoded_position_iChance_lgt, 1, []);
        r_positions_drk = reshape(decoder_targets_drk, 1, []);
        d_positions_drk = reshape(decoded_position_iChance_drk, 1, []);
        
        for iRpos = 1:length(positions)
            Rpos = positions(iRpos);
            dpos_lgt = d_positions_lgt(find(r_positions_lgt == Rpos));
            dpos_drk = d_positions_drk(find(r_positions_drk == Rpos));
            
            for iDpos = 1:length(positions)
                Dpos = positions(iDpos);
                Cdec_pos_count_lgt(Rpos, Dpos) = sum(dpos_lgt == Dpos);
                Cdec_pos_count_drk(Rpos, Dpos) = sum(dpos_drk == Dpos);
            end
            Rpos_sum_lgt = sum(Cdec_pos_count_lgt(Rpos, :), 'omitnan');
            CNorm_dec_pos_count_lgt(Rpos, :) =  Cdec_pos_count_lgt(Rpos, :)./Rpos_sum_lgt;
            Rpos_sum_drk = sum(Cdec_pos_count_drk(Rpos, :), 'omitnan');
            CNorm_dec_pos_count_drk(Rpos, :) =  Cdec_pos_count_drk(Rpos, :)./Rpos_sum_drk;
        end
        
        figure
        subplot(1, 2, 1);
        p1 = heatmap(CNorm_dec_pos_count_lgt, 'Colormap', parula);
        xlabel('Decoded postion (10cm bins)');
        ylabel('VR position (10cm bins)');
        title('Light chance confusion matrix');
        %Find hetamap handles to draw lines where landmarks are
        origState = warning('query', 'MATLAB:structOnObject');
        cleanup = onCleanup(@()warning(origState));
        warning('off','MATLAB:structOnObject')
        S = struct(p1);
        ax = S.Axes;
        clear('cleanup')
        hm.GridVisible = 'off'; % Set heatmap grid invisible.
        % Draw red dasshed lines flanking landmarks
        col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
        xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        
        subplot(1, 2, 2);
        p2 = heatmap(CNorm_dec_pos_count_drk, 'Colormap', parula);
        xlabel('Decoded postion (10cm bins)');
        ylabel('VR position (10cm bins)');
        title('Dark chance confusion matrix');
        %Find hetamap handles to draw lines where landmarks are
        origState = warning('query', 'MATLAB:structOnObject');
        cleanup = onCleanup(@()warning(origState));
        warning('off','MATLAB:structOnObject')
        S = struct(p2);
        ax = S.Axes;
        clear('cleanup')
        hm.GridVisible = 'off'; % Set heatmap grid invisible.
        % Draw red dasshed lines flanking landmarks
        col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
        xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
        yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        
        if plot_drk_lgttrained == true
            decoded_position_iChance_drk_lgttrained = squeeze(decoded_position_chance_drk_lgttrained(chance_repeat, :, :, :));
            d_positions_drk_lgttrained = reshape(decoded_position_iChance_drk_lgttrained, 1, []);
            
            for iRpos = 1:length(positions)
                Rpos = positions(iRpos);
                dpos_drk_lgttrained = d_positions_drk_lgttrained(find(r_positions_drk == Rpos));
                
                for iDpos = 1:length(positions)
                    Dpos = positions(iDpos);
                    Cdec_pos_count_drk_lgttrained(Rpos, Dpos) = sum(dpos_drk_lgttrained == Dpos);
                end
                Rpos_sum_drk_lgttrained = sum(Cdec_pos_count_drk_lgttrained(Rpos, :), 'omitnan');
                CNorm_dec_pos_count_drk_lgttrained(Rpos, :) =  Cdec_pos_count_drk_lgttrained(Rpos, :)./Rpos_sum_drk_lgttrained;
            end
            
            figure
            p3 = heatmap(CNorm_dec_pos_count_drk_lgttrained, 'Colormap', parula);
            xlabel('Decoded postion (10cm bins)');
            ylabel('VR position (10cm bins)');
            title('Dark (lgt trained) chance confusion matrix');
            %Find hetamap handles to draw lines where landmarks are
            origState = warning('query', 'MATLAB:structOnObject');
            cleanup = onCleanup(@()warning(origState));
            warning('off','MATLAB:structOnObject')
            S = struct(p3);
            ax = S.Axes;
            clear('cleanup')
            hm.GridVisible = 'off'; % Set heatmap grid invisible.
            % Draw red dasshed lines flanking landmarks
            col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
            xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        end
        
        if plot_lgt_drktrained == true
            decoded_position_iChance_lgt_drktrained = squeeze(decoded_position_chance_lgt_drktrained(chance_repeat, :, :, :));
            d_positions_lgt_drktrained = reshape(decoded_position_iChance_lgt_drktrained, 1, []);
            
            for iRpos = 1:length(positions)
                Rpos = positions(iRpos);
                dpos_lgt_drktrained = d_positions_lgt_drktrained(find(r_positions_lgt == Rpos));
                
                for iDpos = 1:length(positions)
                    Dpos = positions(iDpos);
                    Cdec_pos_count_lgt_drktrained(Rpos, Dpos) = sum(dpos_lgt_drktrained == Dpos);
                end
                Rpos_sum_lgt_drktrained = sum(Cdec_pos_count_lgt_drktrained(Rpos, :), 'omitnan');
                CNorm_dec_pos_count_lgt_drktrained(Rpos, :) =  Cdec_pos_count_lgt_drktrained(Rpos, :)./Rpos_sum_lgt_drktrained;
            end
            
            figure
            p4 = heatmap(CNorm_dec_pos_count_lgt_drktrained, 'Colormap', parula);
            xlabel('Decoded postion (10cm bins)');
            ylabel('VR position (10cm bins)');
            title('Light (drk trained) chance confusion matrix');
            %Find hetamap handles to draw lines where landmarks are
            origState = warning('query', 'MATLAB:structOnObject');
            cleanup = onCleanup(@()warning(origState));
            warning('off','MATLAB:structOnObject')
            S = struct(p4);
            ax = S.Axes;
            clear('cleanup')
            hm.GridVisible = 'off'; % Set heatmap grid invisible.
            % Draw red dasshed lines flanking landmarks
            col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
            xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
            yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
        end
        
    end
    if plot_subsets == true
        % Light subsets
        figure
        % Discretize start locations if needed
        if discrete_start == false
            start_pos_bins = (1:4:40);
            DtStart_tag_lgt = discretize(tStart_tag_lgt, start_pos_bins);
        end
        
        % Filter subsets to plot
        start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
        
        for i = 1:numel(start_positions)
            start_pos = start_positions(i);
            if ~isempty(find(DtStart_tag_lgt == start_pos, 1))
                trials_subset = find(DtStart_tag_lgt == start_pos);
                if length(trials_subset) >= quality_floor
                    decoder_targets_lgt_subset = decoder_targets_lgt(trials_subset, :);
                    decoded_position_lgt_subset = decoded_position_iChance_lgt(trials_subset, :);
                else
                    decoder_targets_lgt_subset = NaN;
                    decoded_position_lgt_subset = NaN;
                end
                
                r_positions_lgt = reshape(decoder_targets_lgt_subset, 1, []);
                d_positions_lgt = reshape(decoded_position_lgt_subset, 1, []);
                
                for iRpos = 1:length(positions)
                    Rpos = positions(iRpos);
                    dpos_lgt = d_positions_lgt(find(r_positions_lgt == Rpos));
                    
                    for iDpos = 1:length(positions)
                        Dpos = positions(iDpos);
                        Cdec_pos_count_lgt(Rpos, Dpos) = sum(dpos_lgt == Dpos);
                    end
                    Rpos_sum_lgt = sum(Cdec_pos_count_lgt(Rpos, :), 'omitnan');
                    CNorm_dec_pos_count_lgt(Rpos, :) =  Cdec_pos_count_lgt(Rpos, :)./Rpos_sum_lgt;
                end
                
                subplot(2, round(length(start_positions)/2), i);
                p1 = heatmap(CNorm_dec_pos_count_lgt, 'Colormap', parula);
                xlabel('Decoded postion (10cm bins)');
                ylabel('VR position (10cm bins)');
                title('Light chance confusion matrix');
                %Find hetamap handles to draw lines where landmarks are
                origState = warning('query', 'MATLAB:structOnObject');
                cleanup = onCleanup(@()warning(origState));
                warning('off','MATLAB:structOnObject')
                S = struct(p1);
                ax = S.Axes;
                clear('cleanup')
                hm.GridVisible = 'off'; % Set heatmap grid invisible.
                % Draw red dasshed lines flanking landmarks
                col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
                xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            end
        end
        
        %Dark subsets
        figure
        % Discretize start locations if needed
        if discrete_start == false
            start_pos_bins = (1:4:40);
            DtStart_tag_drk = discretize(tStart_tag_drk, start_pos_bins);
        end
        
        % Filter subsets to plot
        start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
        
        for i = 1:numel(start_positions)
            start_pos = start_positions(i);
            if ~isempty(find(DtStart_tag_drk == start_pos, 1))
                trials_subset = find(DtStart_tag_drk == start_pos);
                if length(trials_subset) >= quality_floor
                    decoder_targets_drk_subset = decoder_targets_drk(trials_subset, :);
                    decoded_position_drk_subset = decoded_position_iChance_drk(trials_subset, :);
                else
                    decoder_targets_drk_subset = NaN;
                    decoded_position_drk_subset = NaN;
                end
                r_positions_drk = reshape(decoder_targets_drk_subset, 1, []);
                d_positions_drk = reshape(decoded_position_drk_subset, 1, []);
                
                for iRpos = 1:length(positions)
                    Rpos = positions(iRpos);
                    dpos_drk = d_positions_drk(find(r_positions_drk == Rpos));
                    
                    for iDpos = 1:length(positions)
                        Dpos = positions(iDpos);
                        Cdec_pos_count_drk(Rpos, Dpos) = sum(dpos_drk == Dpos);
                    end
                    Rpos_sum_drk = sum(Cdec_pos_count_drk(Rpos, :), 'omitnan');
                    CNorm_dec_pos_count_drk(Rpos, :) =  Cdec_pos_count_drk(Rpos, :)./Rpos_sum_drk;
                end
                
                subplot(2, round(length(start_positions)/2), i);
                p1 = heatmap(CNorm_dec_pos_count_drk, 'Colormap', parula);
                xlabel('Decoded postion (10cm bins)');
                ylabel('VR position (10cm bins)');
                title('Dark chance confusion matrix');
                %Find hetamap handles to draw lines where landmarks are
                origState = warning('query', 'MATLAB:structOnObject');
                cleanup = onCleanup(@()warning(origState));
                warning('off','MATLAB:structOnObject')
                S = struct(p1);
                ax = S.Axes;
                clear('cleanup')
                hm.GridVisible = 'off'; % Set heatmap grid invisible.
                % Draw red dasshed lines flanking landmarks
                col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
                xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
            end
        end
        
        % Dark subsets trained with light data
        if plot_drk_lgttrained == true
            decoded_position_iChance_drk_lgttrained = squeeze(decoded_position_chance_drk_lgttrained(chance_repeat, :, :, :));
            figure
            % Discretize start locations if needed
            if discrete_start == false
                start_pos_bins = (1:4:40);
                DtStart_tag_drk = discretize(tStart_tag_drk, start_pos_bins);
            end
            
            % Filter subsets to plot
            start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
            
            for i = 1:numel(start_positions)
                start_pos = start_positions(i);
                if ~isempty(find(DtStart_tag_drk == start_pos, 1))
                    trials_subset = find(DtStart_tag_drk == start_pos);
                    if length(trials_subset) >= quality_floor
                        decoder_targets_drk_subset = decoder_targets_drk(trials_subset, :);
                        decoded_position_drk_lgttrained_subset = decoded_position_iChance_drk_lgttrained(trials_subset, :);
                    else
                        decoder_targets_drk_subset = NaN;
                        decoded_position_drk_lgttrained_subset = NaN;
                    end
                    r_positions_drk = reshape(decoder_targets_drk_subset, 1, []);
                    d_positions_drk_lgttrained = reshape(decoded_position_drk_lgttrained_subset, 1, []);
                    
                    for iRpos = 1:length(positions)
                        Rpos = positions(iRpos);
                        dpos_drk_lgttrained = d_positions_drk_lgttrained(find(r_positions_drk == Rpos));
                        
                        for iDpos = 1:length(positions)
                            Dpos = positions(iDpos);
                            Cdec_pos_count_drk_lgttrained(Rpos, Dpos) = sum(dpos_drk_lgttrained == Dpos);
                        end
                        Rpos_sum_drk_lgttrained = sum(Cdec_pos_count_drk_lgttrained(Rpos, :), 'omitnan');
                        CNorm_dec_pos_count_drk_lgttrained(Rpos, :) =  Cdec_pos_count_drk_lgttrained(Rpos, :)./Rpos_sum_drk_lgttrained;
                    end
                    
                    subplot(2, round(length(start_positions)/2), i);
                    p1 = heatmap(CNorm_dec_pos_count_drk_lgttrained, 'Colormap', parula);
                    xlabel('Decoded postion (10cm bins)');
                    ylabel('VR position (10cm bins)');
                    title('Dark (lgt trained) chance confusion matrix');
                    %Find hetamap handles to draw lines where landmarks are
                    origState = warning('query', 'MATLAB:structOnObject');
                    cleanup = onCleanup(@()warning(origState));
                    warning('off','MATLAB:structOnObject')
                    S = struct(p1);
                    ax = S.Axes;
                    clear('cleanup')
                    hm.GridVisible = 'off'; % Set heatmap grid invisible.
                    % Draw red dasshed lines flanking landmarks
                    col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
                    xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                    xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                    yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                    yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                end
            end
        end
        
        % Light subsets trained with dark data
        if plot_lgt_drktrained == true
            decoded_position_iChance_lgt_drktrained = squeeze(decoded_position_chance_lgt_drktrained(chance_repeat, :, :, :));
            figure
            % Discretize start locations if needed
            if discrete_start == false
                start_pos_bins = (1:4:40);
                DtStart_tag_lgt = discretize(tStart_tag_lgt, start_pos_bins);
            end
            
            % Filter subsets to plot
            start_positions = unique(DtStart_tag_lgt(~isnan(DtStart_tag_lgt)));
            
            for i = 1:numel(start_positions)
                start_pos = start_positions(i);
                if ~isempty(find(DtStart_tag_lgt == start_pos, 1))
                    trials_subset = find(DtStart_tag_lgt == start_pos);
                    if length(trials_subset) >= quality_floor
                        decoder_targets_lgt_subset = decoder_targets_lgt(trials_subset, :);
                        decoded_position_lgt_drktrained_subset = decoded_position_iChance_lgt_drktrained(trials_subset, :);
                    else
                        decoder_targets_lgt_subset = NaN;
                        decoded_position_lgt_drktrained_subset = NaN;
                    end
                    
                    r_positions_lgt = reshape(decoder_targets_lgt_subset, 1, []);
                    d_positions_lgt_drktrained = reshape(decoded_position_lgt_drktrained_subset, 1, []);
                    
                    for iRpos = 1:length(positions)
                        Rpos = positions(iRpos);
                        dpos_lgt_drktrained = d_positions_lgt_drktrained(find(r_positions_lgt == Rpos));
                        
                        for iDpos = 1:length(positions)
                            Dpos = positions(iDpos);
                            Cdec_pos_count_lgt_drktrained(Rpos, Dpos) = sum(dpos_lgt_drktrained == Dpos);
                        end
                        Rpos_sum_lgt_drktrained = sum(Cdec_pos_count_lgt_drktrained(Rpos, :), 'omitnan');
                        CNorm_dec_pos_count_lgt_drktrained(Rpos, :) =  Cdec_pos_count_lgt_drktrained(Rpos, :)./Rpos_sum_lgt_drktrained;
                    end
                    
                    subplot(2, round(length(start_positions)/2), i);
                    p1 = heatmap(CNorm_dec_pos_count_lgt_drktrained, 'Colormap', parula);
                    xlabel('Decoded postion (10cm bins)');
                    ylabel('VR position (10cm bins)');
                    title('Light (dark trained) chance confusion matrix');
                    %Find hetamap handles to draw lines where landmarks are
                    origState = warning('query', 'MATLAB:structOnObject');
                    cleanup = onCleanup(@()warning(origState));
                    warning('off','MATLAB:structOnObject')
                    S = struct(p1);
                    ax = S.Axes;
                    clear('cleanup')
                    hm.GridVisible = 'off'; % Set heatmap grid invisible.
                    % Draw red dasshed lines flanking landmarks
                    col = [110, 130, 190, 210, 270, 290, 350, 370, 440, 460]/pbin_size; %Landmark location in tunnel
                    xline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                    xline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                    yline(ax, (col+1), 'w--'); % Set value col +1 to account for the first colums (bin 0)
                    yline(ax, [rzstart/pbin_size+1.1, rzend/pbin_size+1.1], 'r--'); % Set value rzstart/end +1 to account for the first two colums (dark tag and bin 0). Add 0.1 to see the two overlapping lines
                end
            end
        end
    end
end