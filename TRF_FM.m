%% TRF Processing, Forward model
% Require: mTRF Toolbox
 
clear all
close all
clc


sr = 200;
tmin = -100;
tmax = 600;
lenght_segment_s=30;
lambdas_all = 10.^[-10:10]

list=dir(['*forTRF_*.mat'])

for partind=1:length(list)
    close all
    load(list(partind).name)

    if strcmp(partind).name(5:6),'EN')
        Fcz=56;
    elseif strcmp(partind).name(5:6),'SD')
        Fcz=22;
    end
    data_start=data;
    clear data
    fs=sr;
    conditions=[1:10]
    for fe=1:length(data_start.stim)

        stim(fe).name=data_start(1).stim(fe).name;
        stim(fe).data=[]
        resp=[];

        for l=1:length(data_start)
            [a,b]=size(data_start(1).stim(conditions(fe)).data)
            if a==1
                stim(fe).data=[stim(fe).data  data_start(l).stim(conditions(fe)).data];
                resp=[ resp data_start(l).resp];
            else

                stim(fe).data=[stim(fe).data data_start(l).stim(conditions(fe)).data'];
                resp=[ resp data_start(l).resp];
            end
        end
        data(fe).feature=stim(fe).name;

        data(fe).stim=zscore(stim(fe).data);
        data(1).resp=normalize(resp);
        data(fe).stim(isnan(data(fe).stim))=0;
    end

    nfold=round(length(data(1).resp)/lenght_segment_s*sr);

    for i=1:length(data)
        disp(data(i).feature)
        lambdas{i}=lambdas_all;
        [a,b]=size(data(i).stim)
        if a>1
            data(i).stim=data(i).stim';
        end
        [data(i).predall, data(i).testall,data(i).modelall,data(i).lambda]=cv_model_test(data(i).stim',data(1).resp',nfold,lambdas{i},tmin,tmax,fs,Fcz)
        clc
    end
    data = rmfield(data, {'stim','resp','predall'});
    name=['Results_FM_\' list(partind).name(1:6) '.mat']
    save(name,"data","-v7.3")
    clear data
    close all

end


%% Function

function [predall, testall,modelall,lambda]=cv_model_test(stim,resp,nfold,lambdas,tmin,tmax,fs,Fcz)
close all
if length(lambdas)==1
    lambda_all=lambdas;
else
    cv = mTRFcrossval(stim,resp,fs,1,tmin,tmax,lambdas,...
        'zeropad',0,'fast',1,'split',nfold,'method','Ridge');

    nlambda=length(lambdas);

    temp=squeeze(mean(cv.r,1));
    [rmax,idx] = max(temp(:,Fcz));;
    lambda_all = mode(lambdas(idx));
clear temp
end

for testfold = 1:nfold

    [stimtrain,resptrain,stimtest,resptest] = mTRFpartition(stim,resp,nfold,...
        testfold);
    lambda(testfold)=lambda_all;
    model = mTRFtrain(stimtrain,resptrain,fs,1,tmin,tmax,lambda(testfold),'zeropad',0,'verbose',0,'method','Ridge');
    [pred,test] = mTRFpredict(stimtest,resptest,model,'zeropad',0,'verbose',0);
    predall{testfold}=pred';
    testall{testfold}=test.r;
    modelall{testfold}=model;
end

end