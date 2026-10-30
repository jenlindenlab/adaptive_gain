%% Extraction of Envelope, LogEnv & Adaptive Gain (with multiple tau adapt) for Dataset 2

clear all
close all
clc

folderPreprocessedData='' % where the preprocessed EEG and audio are
folderPreprocessedData='C:\Users\adele\OneDrive - University College London\Documents\TRF_offset\Data_Etard\brainsteamandcomprehension\DataWithAudio'
%% Parameters
new_fs=200; %to downsample both EEG and Audio representation
%EEG
fs_eeg=512;


% Audio
SNR=30
fs_audio=48000;

%LogEnv
epsilon=1*10^(-10);

%Adaptive Gain
taus_integ_sec =0.005
taus_adapt_sec =[ 0.005 0.01 0.025  0.05 0.075 0.1 0.25 0.5];
initial_silence_s=3;
Tmax_s=max(taus_adapt_sec)+max(taus_integ_sec)


% General
languages=['EN';'NL']
addpath(folderPreprocessedData)


%% Main loop to extract data from all participants
list=dir([folderPreprocessedData '\*.mat'])

for part=1:length(list)
    tstart(part) = tic;
    for lang=1:2
        language=languages(lang,:);
        load(list(part).name)
        resp_=[];
        audio_=[];
        for iii=1:length(trials)
            if strcmp(trials{iii}.condition,'clean')==1 || strcmp(trials{iii}.condition,'cleanDutch')==1
                if strcmp(trials{iii}.session,language)==1
                    fs_eeg=trials{iii}.fs;
                    fs_audio_init=trials{iii}.audioFs;
                    info_{iii}.subject=trials{iii}.subject;
                    info_{iii}.condition=trials{iii}.condition;
                    info_{iii}.condPart=trials{iii}.condPart;
                    info_{iii}.session=trials{iii}.session;
                    info_{iii}.fs=new_fs;
                    audio_=[audio_; trials{iii}.audio];
                    resp_=[resp_ double(trials{iii}.data)];
                end
            end
        end

        if exist('info_')==1 && isempty(resp_)==0;
            indice=1
            for iii=1:1
                info{iii}=info_{iii};
                [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
                EEG = pop_importdata('dataformat','array','nbchan',63,'data','resp_','srate',1000,'pnts',0,'xmin',0);
                [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','blou','gui','off');
                EEG = pop_reref( EEG, []);
                [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','1_ref','overwrite','on','gui','off');
                EEG = pop_resample( EEG, new_fs);
                [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off');
                eeglab redraw;
                resp=double(ALLEEG.data(:,(5*new_fs)+1:end-(5*new_fs))); 
                audio=resample(audio_,48000,fs_audio_init);
                fs_audio=48000;
                  %% Add really low background noise to allow the AG algorithm to react
                clear sig sound
                sig=audio;
                noise=rand(length(sig),1)-0.5;
                signal_power = sum(abs(sig).*abs(sig))/length(sig);
                noise_power = sum(abs(noise).*abs(noise))/length(noise);
                K_diff = (signal_power/noise_power)*10^(-(SNR)/10);  % Scale factor
                new_noise_diff = sqrt(K_diff)*noise;
                sound=sig+new_noise_diff(1:length(sig));

                %% Audio Representations extractions
                FW=abs(sound);
                LogEnv=log10( FW+epsilon);


                stim(indice).name='Env'; temp=decimate(FW,fs_audio/new_fs);
                stim(indice).data=temp((5*new_fs)+1:end-(5*new_fs)); indice=indice+1; clear temp

                stim(indice).name='LogEnv'; temp=decimate(LogEnv,fs_audio/new_fs);
                stim(indice).data=temp((5*new_fs)+1:end-(5*new_fs)); indice=indice+1; clear temp

                for integ=1:length(taus_integ_sec)

                    for adapt=1:length(taus_adapt_sec)
                        tau_adapt=taus_adapt_sec(adapt)
                        tau_integ=taus_integ_sec(integ)

                        if tau_integ<=tau_adapt

                            [AG]= adaptivegain_extract(LogEnv,fs_audio,tau_integ,tau_adapt,initial_silence_s,Tmax_s);

                            stim(indice).name=['AG_adapt_' num2str(tau_adapt) '_integ_' num2str(tau_integ)]; temp=decimate(AG,fs_audio/new_fs);
                            stim(indice).data=temp((5*new_fs)+1:end-(5*new_fs)); indice=indice+1; clear temp

                        end
                    end
                end
                data(iii).resp=resp;
                data(iii).stim=stim;
                clear resp stim
            end
            filenamesave=['Testcleancode\' list(part).name(1:3) '_' language '_forTRF.mat']
            save(filenamesave,"data","info",'-v7.3')
            indice=1
            clear resp resp_ stim_ stim info data

        end
    end
end


%% Adaptive Gain function

function [AGv_out]= adaptivegain_extract(s_t_in,fs_audio,taus_integ_sec,taus_adapt_sec,initial_silence_s,Tmax_s)

taus_integ_samples=taus_integ_sec .* fs_audio;
taus_adapt_samples = (taus_adapt_sec .* fs_audio)+1;



% Assign variable input arguments.
scale_integ = 1;
scale_adapt = 1;
sat_adapt = 1;
expnt_adapt = 1;
durexp_in_tau = 5;

latency = 0;


initialsilence=rand(fs_audio*initial_silence_s,1)./1000 ;
s_t=[initialsilence; s_t_in];


% Generate exponential causal windows for integration and adaptation.
% Note that these windows are designed to compute a weighted average,
% i.e. the weights on timepoints in the exponential filter are scaled
% so that the sum of the weights equals 1.  Also note that we are using
% causalwinsum_internal instead of Matlab's conv, so the window is a simple
% decaying exponential in preceding time, with the highest value at the end
% (corresponding to current time point).

Tmax=durexp_in_tau.*(taus_integ_samples+taus_adapt_samples);



T=[Tmax:-1:0];

v_=(exp(-T/taus_adapt_samples)-exp(-T/taus_integ_samples))./(taus_adapt_samples-taus_integ_samples);
v=v_/(sum(v_));

% Construct output after latency shift alone.
rlat_t = nan(size(s_t));
rlat_t(latency+1:end) = s_t(1:end-latency);

% Then convolve with temporal integration window (and scale by
% scale_integ).
% rlatinteg_t = scale_integ * causalwinsum_internal(rlat_t, expwin_integ, 'full');

% Finally, compute adaptive gain and apply this gain to rlatinteg_t to get
% the final output, r_t.  Note that we incorporate the
% exponent from the normalization equation, raising each element of
% rlatinteg_t to the power expnt_adapt (both in calculation of adaptwin_t
% in the normalization equation denominator, and the numerator for
% calculation of rlatintegadapt_t).  If this exponent is 1 then the
% equation reduces to simpler form; normalization of the rlatinteg_t output
% by a gain factor inversely proportional to the exponentially weighted
% average of rlatinteg_t.

AGv_win_t=  causalwinsum_internal(rlat_t, v, 'full');

AGv_t =  1./(1+AGv_win_t);
Vonly=AGv_win_t(length(initialsilence)+1:end);
;
AGv_out=AGv_t(length(initialsilence)+1:end);

end




function out_t = causalwinsum_internal(in_t, window, output_type)
% OUT_T = causalwinsum_internal(IN_T, WINDOW, OUTPUT_TYPE)
%
% Perform a causal summation of inputs falling within WINDOW.
% IN_T is a time-varying input vector, WINDOW is a window vector over
% preceding time points. OUT_T is the time-varying output vector obtained
% by performing, for each time point T in OUT_T, a WINDOW-weighted sum
% over the preceding time points T-length(WINDOW)+1:T.  OUTPUT_TYPE is a
% string, either 'full' or 'valid'.
% If 'full', then length(OUT_T) = length(IN_T).
% If 'valid', then length(OUT_T) = length(IN_T)-length(WINDOW)+1, because
% the output is undefined for time points in IN_T before T=length(WINDOW).
%
% Note that this function differs from CONV in that it essentially performs
% a causal convolution; i.e., the convolution window is not symmetric
% around time point T.
%
% Written by J. Linden, 25 June 2013.
% Last modified 25 June 2013.

% Variable declarations.
inlength = length(in_t);
winlength = length(window);

% Make sure IN_T and WINDOW are row vectors.
in_t = in_t(:)';
window = window(:)';

% Now do the weighted summing.
out_t = nan(size(in_t));
for t = winlength:inlength
    out_t(t) =  window * in_t(t-winlength+1:t)';
end;

% Remove initial NaNs, if desired.
switch output_type
    case 'full'
        % do nothing; OUT_T is already as requested
    case 'valid'
        out_t = out_t(winlength:end);
    otherwise
        warning('No output_type specified in causalwinsum_internal.m')
end;


end









