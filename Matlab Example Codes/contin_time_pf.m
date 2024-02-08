

clear all
close all

% Continuous time Levy model

M=1000;
alpha=1.4;
mu_W=1;
sigma_W=2.5;
T=500;


% Dimension of observation vector:
M_v=1;

% Observation noise scaling parameter:
kappa_V=10;

sigma_v=kappa_V*sigma_W;

% Initial prior scaling parameter for mu_W:
kappa_W=100;

% Prior for sigma_W^2 (IG):
alpha_W=0.000000000001;
beta_W=0.000000000001;


% Generate series of  Gammas
%[X, V, m, sum_Gammas_1_alpha,  delta, Gammas, sum_Gammas_2_alpha, m_resid, V_resid, sum_Gammas_1_alpha_M0, sum_Gammas_2_alpha_M0] = stable_series( M, alpha, mu_W, sigma_W);

% Generate uniforms:
%U=T*rand(length(Gammas),1);

% Langevin model:

% Mean reversion coefficient:
theta=-0.005;
A=[0 1; 0 theta]

eA0=[0 1/theta; 0 1];
eA1=[1 -1/theta; 0 0];

M1=[1/theta^2 1/theta;  1/theta 1];
M2=[-2/theta^2 -1/theta; -1/theta 0];
M3=[1/theta^2 0; 0 0];

v1=[1/theta;1]; 
v2=[-1/theta; 0];

time_axis=[0; sort(T*rand(T-1,1))];
%time_axis=[0:T];

% Initial value:
X_true=[0; 0];
drift_true=[0;0];
sde_true=[0;0];
% Average number of jumps per unit time:
c=10;

b_M=alpha/(alpha-1)*c^((alpha-1)/alpha);

% Observation matrix:
Z=[1 0 0];

% Observation noise covariance:
%C_v=sigma_v^2;
C_v=kappa_V^2;

C_e=eye(3);

% Generate some data:
for t=2:length(time_axis)
    
    t_i=time_axis(t);
    delta_t_i=t_i-time_axis(t-1);

   % Generate data: 
    [X_true(:,t),drift_true(:,t),sde_true(:,t),y(t),exp_A_delta_t{t},m_true(:,t),S_true{t},cov_sde{t},R{t},drift_correct_true(:,t)]=update_stable_langevin(X_true(:,t-1),drift_true(:,t-1),sde_true(:,t-1),theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, mu_W, sigma_W, time_axis(t-1),v1,v2,M1,M2,M3,sigma_v);
    Trans_true{t}=[exp_A_delta_t{t} m_true(:,t)-drift_correct_true(:,t);0 0 1];
    H{t}=[R{t}' [0; 0];0 0 0]./sigma_W;
   
end    

%load dataEurUS
%start_time=7.3259e5;
%end_time=start_time+0.2;
%find_data=find(last_traded.t_l>start_time&last_traded.t_l<end_time);
%time_axis=last_traded.t_l(find_data)-start_time;
%y=last_traded.z_l(find_data);
%plot(time_axis,y);
%y=y-y(1);
%time_axis=time_axis*1000;
%
% X_true=0*X_true;


figure(3)
subplot(211),
%plot(time_axis,X_true(1,:)',time_axis,y,'--')
plot(time_axis,y,'--')

title('X_{t} and y_t (dashed), \alpha=0.8, \mu_W=0, \sigma_W=2.5')
subplot(212)
%plot(time_axis,X_true(2,:)')
title('dX_t/dt, \theta=-0.3')

N_particles=100;

% Initial state for Kalman filter:

for n=1:N_particles
% Initial value:
X{n}=[0; 0];
drift{n}=[0;0];
sde{n}=[0;0];
drift_correct{n}=[0;0];

% Initial covariance for Kalman filter:
P{n,1}=zeros(3,3);
P{n,1}(3,3)=kappa_W;
a{n}=[0;0;0];

sigma_W_samp(n)=sigma_W(1);
mu_W_samp(n)=mu_W(1);

particle_weight(n,1)=0;

%particle_store{n}.X=X{n};

end

% Run particle filter:
profile on
for t=2:length(time_axis)  %Iterate over each time step
    t
    
    t_i=time_axis(t);
    delta_t_i=t_i-time_axis(t-1);

   for n=1:N_particles
       
   % Generate hidden state: 
%    [X{n}(:,t),drift{n}(:,t),sde{n}(:,t),y_tmp,exp_A_delta_t{t},m{n}(:,t),S{n,t},cov_sde{n,t},R{n,t},drift_correct{n}(:,t)]=update_stable_langevin(X{n}(:,t-1),drift{n}(:,t-1),sde{n}(:,t-1),theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, mu_W(n), sigma_W(n), time_axis(t-1),v1,v2,M1,M2,M3,sigma_v);
    %[X{n}(:,t),drift{n}(:,t),sde{n}(:,t),y_tmp,exp_A_delta_t{t},m{n}(:,t),S{n,t},cov_sde{n,t},R{n,t},drift_correct_tmp]=update_stable_langevin(X{n}(:,t-1),drift{n}(:,t-1),sde{n}(:,t-1),theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, mu_W_samp, sigma_W_samp(n), time_axis(t-1),v1,v2,M1,M2,M3,sigma_v);
    [X{n}(:,t),drift{n}(:,t),sde{n}(:,t),y_tmp,exp_A_delta_t{t},m{n}(:,t),S{n,t},cov_sde{n,t},R{n,t},drift_correct_tmp]=update_stable_langevin(X{n}(:,t-1),drift{n}(:,t-1),sde{n}(:,t-1),theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, 1, 1, time_axis(t-1),v1,v2,M1,M2,M3,kappa_V);
    
    drift_correct{n}(:,t)=drift_correct_tmp;
    
    Trans{n,t}=[exp_A_delta_t{t} m{n}(:,t)-drift_correct{n}(:,t);0 0 1];
    %H{n,t}=[R{n,t}' [0; 0];0 0 0]./sigma_W;
    H{n,t}=[R{n,t}' [0; 0];0 0 0];
    
   % Kalman filter to estimate state: 
    [a{n}(:,t),P{n,t},a_pred{n}(:,t),P_pred{n,t},log_like(n,t),y_samp(n,t),exp_bit_like(n,t),log_bit_like(n,t)]=kalman_update_pred(a{n}(:,t-1),P{n,t-1},y(t),Z,C_v,{Trans{n,t}},{H{n,t}},{C_e});
    
   % Posterior parameters for sigma_W^2 (IG):
    alpha_W_post=alpha_W+t/2;
    beta_W_post(n)=beta_W+sum(exp_bit_like(n,:));
    
   % mean and var of posterior for sigma_W^2: 
    mean_sigma_W_post(n,t)=max(0,-beta_W_post(n)/(alpha_W_post-1)); 
    var_sigma_W_post(n,t)=max(0,mean_sigma_W_post(n,t)^2/(alpha_W_post-2)); 
    
   % Mode of likelihood: 
    sigma_W_hat(n,t)=sqrt(-2*sum(exp_bit_like(n,:))/(t));
    
   % Marginal likelihood:    
    log_likelihood(n,t)=sum(log_bit_like(n,:))+alpha_W*log(beta_W)-(alpha_W+t/2)*log(beta_W-sum(exp_bit_like(n,:)))+gammaln(t/2+alpha_W)-gammaln(alpha_W); 
   
   % Incremental likelihood:
    log_like_inc(n,t)=log_bit_like(n,t)-(alpha_W+t/2)*log(beta_W-sum(exp_bit_like(n,1:t)))+(alpha_W+(t-1)/2)*log(beta_W-sum(exp_bit_like(n,1:t-1)))+gammaln(t/2+alpha_W)-gammaln((t-1)/2+alpha_W); 
   
    particle_weight(n,t)=particle_weight(n,t-1)+log_like_inc(n,t);
    
    
   end
    
   % Resample particles:
    weight_sum=0;
    log_weight=[];
    for q=1:N_particles
       log_weight(q)=particle_weight(q,t); 
    end
   % (Special step that stops numerical problems in weight calculation):  
    log_weight=log_weight-max(log_weight);
    weight=exp(log_weight);
    weight=weight./sum(weight);
   
    [resample_index]=resample_ressir(weight);
    
    
    X={X{resample_index}};
    drift={drift{resample_index}};
    sde={sde{resample_index}};
    drift_correct={drift_correct{resample_index}};

    
    
   % Initial covariance for Kalman filter:
   for q=1:N_particles
      for tau=1:t 
          P{q,tau}=P{resample_index(q),tau};
      end
   end 
    
   mean_sigma_W_post=mean_sigma_W_post(resample_index,:);
   var_sigma_W_post=var_sigma_W_post(resample_index,:);
   
    a={a{resample_index}};
  
    mean_sigma_W_post=mean_sigma_W_post(resample_index,:);
    
    log_like(:,1:t)=log_like(resample_index,:);
    
    y_samp=y_samp(resample_index,:);
    exp_bit_like=exp_bit_like(resample_index,:);
    log_bit_like=log_bit_like(resample_index,:);
    
    hold off
    particle_weight(:,t)=0;
    if (floor(t/50)==t/50)
        t
        H_fig=figure(1);
         
        clf 
        subplot(211),
        
    hold on
    
    for q=1:N_particles
       plot(a{q}(1,:)')
    end
    plot(y(1:t),'--')
    hold off
    subplot(212)
    for q=1:N_particles
       plot(a{q}(2,:)')
    hold on
    
    end
    %plot(X_true(2,1:t),'--')
    drawnow
    
    movie_frame(t/50)=getframe(H_fig);
    
    figure(2)
    clf
    
    subplot(211),
        
    hold on
    
    for q=1:N_particles
       plot(a{q}(3,:)')
    end
    title('mu_W')
    hold off
    subplot(212)
    semilogy(time_axis(1:t),mean_sigma_W_post',time_axis(1:t),mean_sigma_W_post'+sqrt(var_sigma_W_post)','--',time_axis(1:t),mean_sigma_W_post'-sqrt(var_sigma_W_post)','--')
    title(sigma_W)
    drawnow
    
    
    end
    
   
    
    
    
    
end    

    figure(1)
        clf 
        subplot(211),
        
    hold on
    
    for q=1:N_particles
       plot(a{q}(1,:)')
    end
    plot(y(1:t),'--')
    title('EURUSD exchange rate, 2006, one day, \alpha=1.2, \theta=-0.005, \kappa_V=10, 400 particles. Marginal SMC sample paths for X_t solid' )
    hold off
    subplot(212)
    for q=1:N_particles
       plot(a{q}(2,:)')
    hold on
    
    end
    title('Marginal SMC sample paths for dX_t/dt')
%    plot(X_true(2,1:t),'--')
    drawnow
    
    figure(2)
    clf
    
    subplot(211),
    
    
    hold on
    
    for q=1:N_particles
       plot(a{q}(3,:)')
    end
    hold off
    subplot(212)
    semilogy(time_axis(1:t),mean_sigma_W_post',time_axis(1:t),mean_sigma_W_post'+sqrt(var_sigma_W_post)','--',time_axis(1:t),mean_sigma_W_post'-sqrt(var_sigma_W_post)','--')
    
    
    
    drawnow
    

vidObj = VideoWriter('EURUSD.avi');
vidObj.FrameRate=3;
open(vidObj);
writeVideo(vidObj,movie_frame);
close(vidObj)
profile viewer



% figure
% for q=1:N_particles
%     plot(a{q}(1,:)')
% end
% plot(y,'--')
% 
% figure
% hold on
% subplot(211),plot(time_axis,X(1,:)),title('Langevin model, \alpha=1.5, \mu_W=10, \sigma_W=10, \theta=-0.1'), ylabel('X_t')
% hold on
% subplot(212),plot(time_axis,X(2,:)),ylabel('d{X}_t/dt')
% drawnow
% figure
%  
% plot(time_axis,X(2,:),time_axis,sde(2,:))