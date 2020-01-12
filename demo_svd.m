% DEMO MATRIX FACTORIZATION WEEKS 3-4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOME SIMPLE MATRICES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create rank-1 matrix
d = [1 1 1 1 1 1]'*[1 2 3 4]
rank(d)
% create rank-1 matrix
d = [1 2 3 4 5 6]'*[1 2 3 4]
rank(d)
% create rank-2 matrix
d = [1 1 1 1 1; 1 2 1 2 1]' * [1 0 2; 0 1 1]
rank(d)
% create rank-2 matrix
d = [1 1 1 1 1; 1 2 1 2 1]' * rand(2,100);
size(d)
rank(d)
% create rank-0 matrix
d = zeros(20,4)
rank(d)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Consider a 2-dimensional Temple data set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load d_temple
d = d_temple;
size(d)
% scatter plot of d
figure
plot(d(:,1),d(:,2),'.r'); axis equal
%
% Interesting result related to graphics:
%   create a new set by matrix multiplication
d2 = d * randn(2);
plot(d2(:,1),d2(:,2),'.r'); axis equal
%   create a new set by matrix multiplication - rescaling
d2 = d * [1 0; 0 5];
plot(d2(:,1),d2(:,2),'.r'); axis equal
%   create a new set by matrix multiplication - rotation
rad = 1.1; d2 = d * [cos(rad) sin(rad); -sin(rad) cos(rad)];
plot(d2(:,1),d2(:,2),'.r'); axis equal
%
%
% create a 200 dimensional data set d2 from d
d2 = d * randn(2,200);
size(d2)
% plot the scatter plot of the first 10 attributes
plotmatrix(d2(1:10:end,1:10))
% what is the rank of d2?
rank(d2)
% Let us apply svd
[u,s,v] = svd(d2,0);
size(u)
size(s)
size(v)
% the first 5 singular values: 
s(1:5,1:5)
% project the original data to first 2 dimensions
d2_k = u(:,1:2)*s(1:2,1:2)*v(:,1:2)';
figure
plot(d2_k(:,1),d2_k(:,2),'.'); axis equal

%
% add a lot of noise to d2
d2 = d2 + randn(28226,200)/2;
% plot the scatter plot of the first 10 attributes
plotmatrix(d2(1:10:end,1:10))
% what is the rank of d2?
rank(d2)
% Let us apply svd
[u,s,v] = svd(d2,0);
% the first 5 singular values: 
s(1:5,1:5)
%
% the resulitng 2-dimensional projection
d_new = u(:,1:2)*s(1:2,1:2);
figure
plot(d_new(:,1),d_new(:,2),'.'); axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVD on Iris data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load fisheriris
d = meas;
plotmatrix(d)
%d = normalize(d);
[a,a,y] = unique(species);
scatter(d(:,1),d(:,2),80,y,'filled'); colorbar
[u,s,v] = svd(d,0);
figure
scatter(u(:,1),u(:,2),80,y,'filled'); colorbar
s
k = 2;
dk = u(:,1:k)*s(1:k,1:k)*v(:,1:k)';
rank(d)
rank(dk)
norm(d,'fro')
norm(d - dk)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVD on Newsgroups data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load 20news_w100.mat
whos
d = documents'; % now, rows are documents, columns are words
d1 = d + 1 - 1;  % convert to dense format
%
r = randi(16242,100,1); % select 100 random documents
for i=1:100
    r(i)
    wordlist(d(r(i),:)==1)' % example of the i-th document
    groupnames(newsgroups(r(i))) % type of the i-th document
    pause
end
%d1 = (d1 - repmat(mean(d1),16242,1)) ./ repmat(std(d1),16242,1);
[u,s,v] = svd(d1,0);
whos
text(v(:,1)+.5,v(:,2)+.5,wordlist,'FontSize',14)
figure
%scatter(u(:,1),u(:,2),30,newsgroups,'filled'); colorbar
%
%r = randi(16242,100,1); % select 100 random documents
%text(u(r,1)+.5,u(r,2)+.5,groupnames(newsgroups(r)),'FontSize',14)
% even fancier:
c = ['r','g','b','c'];
for i=1:4 
    q = find(newsgroups == i);
    r1 = randi(length(q),50,1); % select 50 random documents
    r = q(r1);
    text(u(r,2)+.5,u(r,3)+.5,groupnames(newsgroups(r)),'Color',c(i),'FontSize',14)
end

% Example of nonnegative matrix factorization (NMF) which approximates data
% matrix A as W*H
d1 = d + 1 - 1;  % convert to dense format
d2 = (d1 ) ./ repmat(std(d1),16242,1);
[W,H] = nnmf(d2,10);
for i = 1:10
    i
    wordlist(find(H(i,:)>0.10))
end
imagesc(W)

