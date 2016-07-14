function [alpha] = AdaPMMSC(X_a,X_b,na,nb,nc,dim,l2norm,maxiter) 
% This code is modified from the following codes 
% 1. code provided by Honglak Lee, Alexis
% Battle, Rajat Raina, and Andrew Y. Ng in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
% 2. code provided by Deng Cai,
% http://www.cad.zju.edu.cn/home/dengcai/Data/data.html

% The detail of the algorithm is described in the following paper:
% [1] Zhou Zhao, Hanqing Lu, Deng Cai, Xiaofei He, Yueting Zhuang. 
% "Partial Multi-Modal Sparse Coding via Adaptive Similarity Structure Regularization",
% ACM Multimedia (MM), 2016

% Written by Hanqing Lu <hanqinglucs@gmail.com>
% Copyright 2016 by Zhou Zhao, Hanqing Lu, Deng Cai, Xiaofei He, Yueting Zhuang.

dima = size(X_a,1);
dimb = size(X_b,1);
Da = rand(dima,dim);
Db = rand(dimb,dim);
alpha_a = rand(dim,na);
alpha_b = rand(dim,nb);
alpha_c = rand(dim,nc);
pa = rand(nc+na,nc+na);
pb = rand(nc+nb,nc+nb);
DPa = zeros(nc+na,nc+na);
DPb = zeros(nc+nb,nc+nb);
l = 0.95;
    for i = 1 : maxiter
        [pa,pb] = update_P([alpha_c alpha_a],[alpha_c alpha_b],alpha_c,pa,pb);
        Da = learn_basis(X_a, [alpha_c alpha_a], l2norm, Da);
        Db = learn_basis(X_b, [alpha_c alpha_b], l2norm, Db);
        for j = 1 : na 
            DPa(j,j) = (sum(pa(j,:))+sum(pa(:,j)))/2;
        end
        for j = 1 : nb
            DPb(j,j) = (sum(pb(j,:))+sum(pb(:,j)))/2;
        end
        La = DPa - (pa' + pa)/2;
        Lb = DPb - (pb' + pb)/2;
        pab = pa(:,1:nc)*pb(:,1:nc)';
        alpha_a = update_a(Da,alpha_a,La,pab,alpha_b,X_a);
        alpha_b = update_b(Db,alpha_b,Lb,pab,alpha_a,X_b);
        alpha_c = update_c(Da,La,alpha_a,X_a,Db,Lb,alpha_b,X_b,alpha_c);
    end
    
    options.ReducedDim = dim; 
    eigvector = PCA(X_b', options);
    alpha_hat = (X_b'*eigvector)';
    alpha = [alpha_c alpha_a alpha_b];
end

function [pa,pb]= update_P(alpha_a,alpha_b,alpha_c,pa,pb)
    na = size(alpha_a,2);
    nb = size(alpha_b,2);
    nc = size(alpha_c,2);
    da = zeros(na,na);
    db = zeros(nb,nb);
    
    for i = 1 : na 
        for k = 1 : nc
            da(i,k) = da(i,k) - norm(alpha_a(:,i)-alpha_c(:,k),'fro')^2;
            for j = 1 : nb
                da(i,k) = da(i,k) - norm(alpha_a(:,i)-alpha_b(:,j),'fro')^2*pb(k,j);
            end
        end
        for k = nc+1 : na
            da(i,k) = -norm(alpha_a(:,i)-alpha_b(:,j),'fro')^2/2;
        end
       
    end
    
    for i = 1 : na
        pa(i,:) = ProSimplex(da(i,:));
    end
    
    for i = 1 : nb 
        for k = 1 : nc
            db(i,k) = db(i,k) - norm(alpha_b(:,i)-alpha_c(:,k),'fro')^2;
            for j = 1 : na
                db(i,k) = db(i,k) - norm(alpha_b(:,i)-alpha_a(:,j),'fro')^2*pa(j,k);
            end
        end
        for k = nc+1 : nb
            db(i,k) = -norm(alpha_a(:,i)-alpha_b(:,j),'fro')^2/2; 
        end
    end
    
    for i = 1 : nb
        pb(i,:) = ProSimplex(db(i,:));
    end
end
function alpha_c = update_c(Da,La,alpha_a,X_a,Db,Lb,alpha_b,X_b,alpha_c)
    na = size(alpha_a,2);
    nb = size(alpha_b,2);
    nc = size(alpha_c,2);
    dim = size(alpha_c,1);
    h = zeros(dim,nc);
    
    for i = 1 : nc
        for j = 1 : na
            h(:,i) = h(:,i) + La(i,j)*alpha_a(:,j);
        end
        for j = 1 : nb
            h(:,i) = h(:,i) + Lb(i,j)*alpha_b(:,j);
        end
    end
    for i = 1 : nc
        alpha_c(:,i) = (Da'*Da + Db'*Db + (La(i,i)+Lb(i,i))*eye(dim,dim))\(Da'*X_a(:,i)+Db'*X_b(:,i)-h(:,i));
    end
end

function alpha_a = update_a(Da,alpha_a,La,pab,alpha_b,X_a)
   na = size(alpha_a,2);
   nb = size(alpha_b,2);
   dim = size(alpha_a,1);
   g = zeros(dim,na);
   for i = 1 : na
        for j = 1 : na
            g(:,i) = g(:,i) + La(i,j)*alpha_a(:,j);
        end
        for j = 1 : nb
            g(:,i) = g(:,i) + pab(i,j)*alpha_b(:,j);
        end
   end 
    for i = 1 : na
        alpha_a(:,i) = (Da'*Da + (La(i,i)+1)*eye(dim,dim))\(Da'*X_a(:,i)-g(:,i));
    end
end

function alpha_b = update_b(Db,alpha_b,Lb,pab,alpha_a,X_b)
   na = size(alpha_a,2);
   nb = size(alpha_b,2);
   dim = size(alpha_b,1);
   g = zeros(dim,nb);
   for i = 1 : nb
        for j = 1 : na
            g(:,i) = g(:,i) + Lb(i,j)*alpha_a(:,j);
        end
        for j = 1 : nb
            g(:,i) = g(:,i) + pab(i,j)*alpha_b(:,j);
        end
   end 
    for i = 1 : na
        alpha_a(:,i) = (Db'*Db + (Lb(i,i)+1)*eye(dim,dim))\(Db'*X_b(:,i)-g(:,i));
    end
end