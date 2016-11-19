function Save_nn_plots_for_py(exper)

% 'del1.2_abs1.0_T42'

[u,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'u','noavg');
[t,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'temp','noavg');
[q,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'shum_avg','noavg');
[w,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'w_avg','noavg');
[stream,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'streamfctn','neg','noavg');
[rhum_avg,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'rhum_avg','noavg');
[buoyancy_freq,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'buoyancy_freq','noavg');
[theta_e,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'theta_e','noavg');
[theta_e_sat,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'theta_e_sat','noavg');
[vq,lat2,sigma,theta,lon] = avg_fields_jgd(exper,'vshum_avg','neg','noavg');

[tas,lat2,~,theta,lon] = avg_fields_jgd(exper,'t_surf','x20','noavg');


%Get netcdf files for 24x data
[f,~,~] = experiment_jgd(exper);

%Initialize
conv=[]; cond=[]; time=[];
if strcmp(exper,'nowetconvection_abs1.0')
    cond=nan; cond_q999 = nan;
    conv=nan; conv_q999 = nan;
    lon = ncread(f.x4{1},'lon');
    latfull = ncread(f.x4{1},'lon');
else
    %Get data from each file will be lon x lat 
    lon = ncread(f.x24{1},'lon');
    latfull = ncread(f.x24{1},'lat'); 
    %Load precip data cross all files
    for i=1:length(f.x24)
        convload = ncread(f.x24{i},'convection_rain');
        conv     = cat(3,conv,convload);
        condload = ncread(f.x24{i},'condensation_rain');
        cond     = cat(3,cond,condload);
        timeload = ncread(f.x24{i},'time');
        time=[time;timeload];
        clear convload condload timeload
    end

    % Convert to mm/day
    cond = cond*3600*24;
    conv = conv*3600*24;

    cond = permute(cond,[2,1,3]); %now lat x lond x time
    conv = permute(conv,[2,1,3]); %now lat x lond x time
    cond = reshape(cond,[size(cond,1), size(cond,2)*size(cond,3)]);
    conv = reshape(conv,[size(conv,1), size(conv,2)*size(conv,3)]);
%     [lat2,cond] = avg_idealized_lat(latfull,cond);
%     [lat2,conv] = avg_idealized_lat(latfull,conv);
    %Get extreme stats
    for i=1:size(cond,1)
        cond_q999(i) = quantile(cond(i,:),.999);
        conv_q999(i) = quantile(conv(i,:),.999);
    end
    %(for now just store avgs)
    cond = squeeze(mean(cond,2)); %avg in time and lon
    conv = squeeze(mean(conv,2));
end

lat = lat2;


save(sprintf('~/mitbox/scripts/nn/data/%s_climo.mat',exper),...
    'u','t','q','w','lat','sigma','cond','conv','tas','cond_q999',...
    'conv_q999','stream','rhum_avg','buoyancy_freq','theta_e','theta_e_sat','vq')

% [f,~,~] = experiment_jgd(exper);
% bk=ncread(f.x4{1},'bk');

