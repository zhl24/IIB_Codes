clear all
close all

% Continuous time Levy model

M=1000;
alpha=1.5;
mu_W=-1;
sigma_W=10;
T=1000;

% Generate series of  Gammas
%[X, V, m, sum_Gammas_1_alpha,  delta, Gammas, sum_Gammas_2_alpha, m_resid, V_resid, sum_Gammas_1_alpha_M0, sum_Gammas_2_alpha_M0] = stable_series( M, alpha, mu_W, sigma_W);

% Generate uniforms:
%U=T*rand(length(Gammas),1);

% Langevin model:

% Mean reversion coefficient:
theta=-0.1;
A=[0 1; 0 theta]

eA0=[0 1/theta; 0 1];
eA1=[1 -1/theta; 0 0];

M1=[1/theta^2 1/theta;  1/theta 1];
M2=[-2/theta^2 -1/theta; -1/theta 0];
M3=[1/theta^2 0; 0 0];

v1=[1/theta;1]; 
v2=[-1/theta; 0];

time_axis=sort(T*rand(1000,1));
%time_axis=[0:T];

% Initial value:
X=[0; 0];


% Average number of jumps per unit time:
c=10; 

b_M=alpha/(alpha-1)*c^((alpha-1)/alpha);

for t=2:length(time_axis)
    
    t_i=time_axis(t);
    delta_t_i=t_i-time_axis(t-1);
   % Deterministic component:
    X_t_i_det=(exp(theta*delta_t_i)*eA0+eA1)*X(:,t-1);

   % Generate series of  Gammas:
    [Y, V, m, sum_Gammas_1_alpha,  delta, Gammas, sum_Gammas_2_alpha, m_resid, V_resid, sum_Gammas_1_alpha_M0, sum_Gammas_2_alpha_M0] = stable_series( c*delta_t_i, alpha, mu_W, sigma_W);

    
    
   % Generate jump times:
    U_i=rand(length(Gammas),1)*delta_t_i+time_axis(t-1);
    
    Gamma_i=Gammas;
    Gamma_i_1_alpha=Gamma_i.^(-1/alpha);
    Gamma_i_2_alpha=Gamma_i.^(-2/alpha);
    
    sum_0=sum(Gamma_i_1_alpha);
    sum_1=sum(Gamma_i_1_alpha.*exp(theta*(t_i-U_i)));
    sum_2=sum(Gamma_i_2_alpha.*exp(2*theta*(t_i-U_i)));
    sum_3=sum(Gamma_i_2_alpha.*exp(theta*(t_i-U_i)));
    sum_4=sum(Gamma_i_2_alpha); 

    m=delta_t_i^(1/alpha)*(sum_0*v2+sum_1*v1);
    S=delta_t_i^(2/alpha)*(sum_2*M1+sum_3*M2+sum_4*M3);
    
    % Centering term:
    drift=(alpha>1)*b_M*(1/theta*(exp(theta*delta_t_i)-1)*[1/theta; 1]-delta_t_i*[1/theta; 0]);
   
    % Linear sde term:
    cov_sde=(exp(2*theta*delta_t_i)-1)/(2*theta)*M1+(exp(theta*delta_t_i)-1)/(theta)*M2+delta_t_i*M3;
    cov_sde=cov_sde*(sigma_W^2+mu_W^2)*alpha/(2-alpha)*c^(1-2/alpha);
    
    
    if cond(sigma_W^2*S+cov_sde)<1e12
        
       R=chol(sigma_W^2*S+cov_sde);
        
    else 
       
      % Cheat!!
       disp('ill-conditioned covariance!')
       R=0;
    
    end   
   % Generate next x:
    X(:,t)=X_t_i_det+mu_W*(m-drift)+R'*randn(2,1);
    
end    
hold on
subplot(211),plot(time_axis,X(1,:)),title('Langevin model, \alpha=1.5, \mu_W=10, \sigma_W=10, \theta=-0.1'), ylabel('X_t')
hold on
subplot(212),plot(time_axis,X(2,:)),ylabel('d{X}_t/dt')
drawnow
