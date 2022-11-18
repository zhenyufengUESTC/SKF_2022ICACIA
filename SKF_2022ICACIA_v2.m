clear all;close all;
%%
% @ Copyright Zhengyu Feng @ UESTC.
% @ Date 2021.11.15.
% @ Version V_1.0.
%% Wasserstein KF 算法
% 初始化参数
% KF模型参数
n = 4;
m = 1;
N = 500;% 迭代次数
% 协方差矩阵
xe = ones(n,N);
xx = ones(n,N);

del_T = 0.001 ; %
F = [1 0 del_T 0; 0 1 0 del_T; 0 0 1 0; 0 0 0 1];
% F = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
H = [ 1 1 0 0] ;




%% 算法迭代 kk 次
KK = 500;
tic
for kk = 1:KK
    % 状态高斯噪声
    v1 = randn(n,N)*0.1;
    % 生成观测噪声非高斯噪声
    % v = randn(n,N)*1;
    p1 = randn(n,N)*0.1;p2 = randn(n,N)*15;
    v = randn(n,N);%%高斯噪声
    vp = rand(n,N);
    for jj = 1:n
        for tt = 1:N
            if vp(jj,tt) > 0.95
                v(jj,tt) = p2(jj,tt);
            else
                v(jj,tt) = p1(jj,tt);
            end
        end
    end
    % 生成观测噪声非高斯噪声
    v = randn(m,N)*1;
    q1 = randn(m,N)*0.1;q2 = randn(m,N)*15;
    v = randn(m,N);%%高斯噪声
    vp = rand(m,N);
    for jj = 1:m
        for tt = 1:N
            if vp(jj,tt) > 0.95
                v(jj,tt) = q2(jj,tt);
            else
                v(jj,tt) = q1(jj,tt);
            end
        end
    end
    Q = (v1 * v1')/N;%% 必须知道状态噪声协方差 Q
    R = (v * v')/N;
    
    % MSE算法初始化参数
    xee2 = xe(:,1);
    xe2 = xe(:,1);
    Pk2 = eye(n);
   
    % FENG MSE KF
    xee_NN = xe(:,1);
    xe_NN = xe(:,1);
    KG_NN = ones(n, m);
    

    %% 状态方程扰动
    for ii = 2:N
%         delta_t = -delta + 2 * delta * rand(1);% [-1,1]区间的均匀分布随机数*delta_t
        %                 delta_t = 0;
%         FF = [0.9802,  0.0196+0.099*delta_t*1; 0, 0.9802]; %  .1196  .98022 .0196*delta_t+0.099*15 + 
        %         H =  [1 + 0.099*1*delta_t,-1];
        % KF 真实值
        xx(:,ii) = F * xx(:,ii - 1) + v1(:,ii);
        yy(:,ii) = H * xx(:,ii) +  v(:,ii);
        

        %% MSE KF 算法
        % 必须知道状态噪声协方差 Q
        yy_MSE =  yy(:,ii);
        xee2 = xe2(:,ii-1);
        Pke2 = F * Pk2 * F'+ Q;% ;
        G_MSE = Pke2 * H' * inv(H*Pke2*H' + R); %
        xee2 = F * xee2 + G_MSE*(yy_MSE -H*F*xee2);
        Pk2 = Pke2 - G_MSE*H*Pke2;
        xe2(:,ii) = xee2;
        Err_MSE_KF(kk,ii) = norm(xe2(:,ii) - xx(:,ii));
        
        
%          %% MSE KF New 算法
%         xee_New = xe_New(:,ii-1);
%         Pke3 = F * Pk_New * F'+ Q; % ;
%         Pk_New = inv(H'*H + inv(Pke3));%*inv(R)
%         xee_New = Pk_New*(H'*inv(R)*yy_MSE + inv(Pke3)*F*xee_New); %
%         xe_New(:,ii) = xee_New;
%         Err_MSE_KF_New(kk,ii) = norm(xe_New(:,ii) - xx(:,ii));

        %% NN KF  算法
        yy_NN =  yy(:,ii);
        xee_NN_step(:, ii) = F * xe_NN(:, ii-1);
%         xee_NN_ = xe_NN(:, ii-1);
%         xee_NN_k = xee_NN_step(:, ii);
%         delta_x = xee_NN_k - xee_NN_;  %
%         delta_y = (yy_NN - H * xee_NN_step(:, ii));  %
%         KG_NN  = (delta_x * delta_y') * inv(  delta_y * delta_y');  %
%         xee_NN_t = xee_NN_step(:, ii) + KG_NN * (yy_NN - H * xee_NN_step(:, ii));  %
        xee_NN_ = xe_NN(:, ii-1);
        xee_NN_k = xee_NN_step(:, ii) ;
        for ll = 1:100
            delta_x = xee_NN_k - xee_NN_;  %
            delta_y = (yy_NN - H * xee_NN_k);  %
            KG_NN  = (delta_x * delta_y') * inv(  delta_y * delta_y');  %
            xee_NN_t = xee_NN_step(:, ii) + KG_NN * (yy_NN - H * xee_NN_step(:, ii));  %
%             xee_NN_t = xee_NN_k + KG_NN * (yy_NN - H * xee_NN_k);  %
            xee_NN_ = xee_NN_k;
            xee_NN_k = xee_NN_t;  % 
            if (norm(xee_NN_k-xee_NN_)/norm(xee_NN_)) <= 0.000001
                break;
            end
        end
        xe_NN(:,ii) = xee_NN_t;
        Err_NN_KF(kk,ii) = norm(xe_NN(:,ii) - xx(:,ii));
        
     

       
        
        
    end
%     fprintf('%d-th Iteration...\n',kk);

end
toc
disp(['运行时间: ',num2str(toc/KK)]);

%% 绘图
figure; hold on;%
plot(10*log10(mean(Err_MSE_KF)));plot(10*log10(mean(Err_NN_KF)));%plot(10*log10(mean(Err_MSE_KF_New)));plot(10*log10(mean(Err_Feng_MCC_KF)));
legend('Err-MSE-KF','Err-Feng-KF');%, 'Err-Feng-MCC-KF','Err-MSE-KF-New'

[mean(mean(Err_MSE_KF)),mean(mean(Err_NN_KF))]%, mean(mean(Err_Feng_MCC_KF))mean(mean(Err_MSE_KF_New)),


