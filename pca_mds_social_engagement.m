% =========================================================================
% Title:       Dimensionality Reduction on Social Media Engagement: PCA vs MDS
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Comparative analysis of Principal Component Analysis (PCA) and 
%              Classical Multidimensional Scaling (MDS) applied to social 
%              engagement metrics from 9 Thai retail sellers on Facebook.
%              The pipeline includes:
%                1. Data standardization (MAX / SD) and mean-centering.
%                2. PCA via Singular Value Decomposition (SVD) of the sample covariance matrix.
%                3. Classical MDS via double-centering of the squared Euclidean distance matrix.
%                4. Reconstruction error assessment and empirical verification of PCA-MDS equivalence.
%                5. 1D/2D latent space projections and feature loading analysis.
% =========================================================================

clear; clc; close all

%% ========================================================================
% 1. DATA INITIALIZATION & STANDARDIZATION
% =========================================================================

% Maximum values across engagement metrics (Comments, Shares, Likes, Love, Wow, Haha, Sad, Angry)
MAX=[20990.00 1260.00 4710.00 234.00 21.00 8.00 51.00 6.00
    12003.00 856.00 1744.00 225.00 26.00 40.00 23.00 31.00
    9452.00 1636.00 2344.00 282.00 57.00 102.00 8.00 12.00
    1734.00 247.00 497.00 55.00 3.00 5.00 2.00 2.00
    860.00 356.00 259.00 49.00 4.00 5.00 1.00 2.00
    6174.00 3424.00 2293.00 657.00 278.00 157.00 37.00 8.00
    779.00 304.00 186.00 106.00 3.00 8.00 2.00 1.00
    3800.00 757.00 4241.00 198.00 23.00 12.00 46.00 19.00
    17404.00 913.00 1917.00 220.00 14.00 43.00 14.00 10.00];

% Standard deviations across engagement metrics
SD=[588.36 33.23 605.85 12.66 1.36 0.57 1.67 0.22
    714.79 75.68 143.08 22.84 1.76 1.92 0.88 0.93
    2257.18 379.58 327.48 76.92 7.00 13.84 1.15 1.46
    425.78 43.96 84.93 16.43 0.71 1.15 0.44 0.38
    96.54 68.38 36.95 7.08 0.47 0.64 0.12 0.22
    833.34 388.37 495.92 137.32 41.49 14.84 2.91 1.38
    147.32 63.21 31.10 18.47 0.39 0.78 0.16 0.11
    501.04 128.87 381.03 39.47 2.60 1.46 2.21 0.94
    1780.94 144.01 140.10 27.44 1.97 4.40 1.44 0.87];

% Standardization and transposition: X is m x n (m = features, n = sellers)
X=MAX./SD; 
X=X';
[m,n]=size(X);
k=2; % Target reduced dimension

%% ========================================================================
% 2. PRINCIPAL COMPONENT ANALYSIS (PCA)
% =========================================================================

% Mean-centering across observations
mu=(X*ones(n,1))/n; 
Xt=X-(mu*ones(1,n)); 

% Sample covariance matrix
CovX=(Xt*Xt')/(n-1); 

% Singular Value Decomposition of the covariance matrix
[U,S,V] = svd(CovX);
vars=diag(S).*diag(S); % Variances associated with components

% Projection onto the first k principal components
u=zeros(m,k); 
for j=1:k 
    u(:,j)=U(:,j);
end
z=u'*Xt; % Coordinates in the reduced k-dimensional space

disp ('projected data z='); disp(z);
disp('k=2 principal components');disp(u);

% Reconstruction back to the original space
x=u*z; 
Xog=x+mu; 

% Reconstruction error evaluation
errPCA=norm(Xt-u*u'*Xt); 
errrelPCA=errPCA/norm(Xt); 
disp('PCA error=');disp(errrelPCA);

%% ========================================================================
% 3. PCA VISUALIZATION & DIAGNOSTICS
% =========================================================================

% Figure 1: 1D Projection (PC1)
figure(1)
hold on
plot (z(1,1),0,'b.',z(1,2),0,'m.', z(1,3),0,'y.', z(1,4),0,'g.', ...
    z(1,5),0,'r.','markersize', 18)
scatter(z(1,6),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0 0.4470 0.7410])
scatter(z(1,7),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4660 0.6740 0.1880])
scatter(z(1,8),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0.9290 0.6940 0.1250])
scatter(z(1,9),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4940 0.1840 0.5560])
xlabel('PC1')
legend('Seller 1','Seller 2','Seller 3','Seller 4','Seller 5','Seller 6', ...
    'Seller 7','Seller 8','Seller 9')
hold off

% Figure 2: 2D Projection (PC1 vs PC2)
figure(2)
hold on
plot(z(1,1),z(2,1),'b.',z(1,2),z(2,2),'m.', z(1,3),z(2,3),'y.', ...
    z(1,4),z(2,4),'g.', z(1,5),z(2,5),'r.','markersize', 18)
scatter(z(1,6),z(2,6),"MarkerEdgeColor",'w',"MarkerFaceColor",[0 0.4470 0.7410])
scatter(z(1,7),z(2,7),"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4660 0.6740 0.1880])
scatter(z(1,8),z(2,8),"MarkerEdgeColor",'w',"MarkerFaceColor",[0.9290 0.6940 0.1250])
scatter(z(1,9),z(2,9),"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4940 0.1840 0.5560])
hold off
xlabel('PC1'), ylabel('PC2')
legend('Seller 1','Seller 2','Seller 3','Seller 4','Seller 5','Seller 6', ...
    'Seller 7','Seller 8','Seller 9')

% Figure 3: Load Plot (Feature contributions)
figure(3)
plot(V(1,1),V(1,2),'r.',V(2,1),V(2,2),'r.',V(3,1),V(3,2),'b.', ...
    V(4,1),V(4,2),'b.',V(5,1),V(5,2),'b.',V(6,1),V(6,2),'b.', ...
    V(7,1),V(7,2),'b.',V(8,1),V(8,2),'b.','markersize', 15)
text(V(1,1)+0.01,V(1,2),'Comment')
text(V(2,1)+0.01,V(2,2),'Share')
text(V(3,1)+0.01,V(3,2),'Like')
text(V(4,1)+0.01,V(4,2),'Love')
text(V(5,1)+0.01,V(5,2),'Wow')
text(V(6,1)+0.01,V(6,2),'Haha')
text(V(7,1)+0.01,V(7,2),'Sad')
text(V(8,1)+0.01,V(8,2)-0.02,'Angry')

% Figure 4: Scree Plot and Cumulative Explained Variance
figure(4)
bar(vars) 
xlabel('eigenvector number'), ylabel('eigenvalue')

t=sum(vars);
y=cumsum(vars/t);
disp('varianza cumulativa'); disp(y);

%% ========================================================================
% 4. CLASSICAL MULTIDIMENSIONAL SCALING (MDS)
% =========================================================================

% Squared Euclidean distance matrix computation: D = D^2
D=(((diag(X'*X))*ones(1,n))+(ones(n,1)*(diag(X'*X)'))-(2*(X')*X)); 
[n,m]=size(D);
k=2;

% Construction of centering matrix H
I=eye(n);
O=ones(n);
H=I-((1/n)*O);

% Double centering operation to obtain the Gram matrix B
B=-0.5*(H*D*H');

% Spectral decomposition of B
[U,L,W] = svd(B);
v=zeros(n,k);
s=zeros(k,k);
for j=1:k
    v(:,j)=W(:,j); % First k principal eigenvectors
    for i=1:k
        s(i,j)=sqrt(L(i,j));
    end
end

% Coordinate recovery in the reduced space
z1=s*v';
disp ('projected data z1='); disp(z1); 

disp('k=2 eigenvalue of B'); disp(s);
disp('k=2 eigenvector of B');disp(v);

% MDS Gram reconstruction error
errMDS=norm((Xt'*Xt)-(z1'*z1));
errrelMDS=errMDS/norm(Xt'*Xt); 
disp('MDS error=');disp(errrelMDS);

% Equivalence verification between PCA and MDS coordinates
errPCAMDS=norm(z-z1); 

%% ========================================================================
% 5. MDS VISUALIZATION
% =========================================================================

% Figure 5: 1D MDS Projection (MD1)
figure(5)
hold on
plot (z1(1,1),0,'b.',z1(1,2),0,'m.', z1(1,3),0,'y.', z1(1,4),0,'g.', ...
    z1(1,5),0,'r.','markersize', 18)
scatter(z1(1,6),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0 0.4470 0.7410])
scatter(z1(1,7),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4660 0.6740 0.1880])
scatter(z1(1,8),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0.9290 0.6940 0.1250])
scatter(z1(1,9),0,"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4940 0.1840 0.5560])
xlabel('MD1')
legend('Seller 1','Seller 2','Seller 3','Seller 4','Seller 5','Seller 6', ...
    'Seller 7','Seller 8','Seller 9')
hold off

% Figure 6: 2D MDS Projection (MD1 vs MD2)
figure(6)
hold on
plot(z1(1,1),z1(2,1),'b.',z1(1,2),z1(2,2),'m.', z1(1,3),z1(2,3),'y.', ...
    z1(1,4),z1(2,4),'g.', z1(1,5),z1(2,5),'r.','markersize', 18)
scatter(z1(1,6),z1(2,6),"MarkerEdgeColor",'w',"MarkerFaceColor",[0 0.4470 0.7410])
scatter(z1(1,7),z1(2,7),"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4660 0.6740 0.1880])
scatter(z1(1,8),z1(2,8),"MarkerEdgeColor",'w',"MarkerFaceColor",[0.9290 0.6940 0.1250])
scatter(z1(1,9),z1(2,9),"MarkerEdgeColor",'w',"MarkerFaceColor",[0.4940 0.1840 0.5560])
hold off
xlabel('MD1'), ylabel('MD2')
legend('Seller 1','Seller 2','Seller 3','Seller 4','Seller 5','Seller 6', ...
    'Seller 7','Seller 8','Seller 9')
