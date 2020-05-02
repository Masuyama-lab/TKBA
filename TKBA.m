% 
% (c) 2019 Naoki Masuyama
% 
% These are the codes of Topological Kernel Bayesian Adaptive Resonance Theory 
% (TKBA) proposed in "N. Masuyama, C. K. Loo, and S. Wermter, A Kernel Bayesian 
% Adaptive Resonance Theory with A Topological Structure, International Journal 
% of Neural Systems, vol. 29, no. 5, pp. 1850052-1-1850052-20, January 2019."
% 
% Please contact "masuyama@cs.osakafu-u.ac.jp" if you have any problems.
% 
function [net] = TKBA(DATA, net)


numClusters = net.numClusters;   % Number of clusters
weight = net.weight;             % Mean of cluster
CountCluster = net.CountCluster; % Counter for each cluster
NewEdgedNode = net.NewEdgedNode; % Node which creates new edge

Lambda = net.Lambda; % Interval for Node deletion and topology construction
maxCIM = net.maxCIM; % Vigilance Parameter by CIM
kbrSig = net.kbrSig; % Kernel Bandwidth for Kernel Bayes Rule
cimSig = net.cimSig; % Kernel Bandwidth for CIM

% Parameters for Topology
edge = net.edge;         % connections (edges) matrix
ErrCIM = net.ErrCIM;     % CIM between clusters




% Classify and learn on each sample.
[numSamples, ~] = size(DATA);

  
for sampleNum = 1:numSamples

    % Current data sample
    pattern = DATA(sampleNum,:);

    % Find the winner cluster candidates based on CIM
    clusterCIM = CIM(pattern, weight, cimSig);
    [stateCIM, orderCIM] = sort(clusterCIM, 'ascend');

    % Number of clusters close to pattern in range of maxCIM
    NstateCIM = sum( stateCIM <= maxCIM );
    

    if NstateCIM == 0 % No clusters around pattern
        % Add Cluster
        numClusters                  = numClusters + 1;
        weight(numClusters,:)        = pattern;
        CountCluster(1, numClusters) = 1;
        NewEdgedNode(1, numClusters) = 0;
        ErrCIM(1, numClusters)       = 1;
        edge(numClusters, :)         = 0;
        edge(:, numClusters)         = 0;

    elseif NstateCIM >= 1
        
        
        % Extract clusters which are close to pattern
        EXweight       = weight(orderCIM(1:NstateCIM), :);
        EXorderCIM     = orderCIM(1:NstateCIM);   
        EXCountCluster = CountCluster(EXorderCIM);
        EXnumClusters  = size(EXweight,1);
        
        % Parameters for Kernel Bayes Rule
        paramKBR.Sig     = kbrSig;             % Kernel bandwidth
        paramKBR.numNode = EXnumClusters;      % Number of Clusters
        paramKBR.Eps     = 0.01/EXnumClusters; % Scaling Factor
        paramKBR.Delta   = 2*EXnumClusters;    % Scaling Factor
        paramKBR.gamma   = ones(size(pattern, 1),1) / size(pattern, 1); % Scaling Factor
        paramKBR.prior   = EXCountCluster' / sum(EXCountCluster);       % Prior Probability based on CountCluster
        
        % Kernel Bayes Rule
        [KernelPo, ~] = KernelBayesRule(pattern, EXweight, paramKBR);
        [~, numSortedProb] = sort(KernelPo, 'descend');
        
        EXs1 = numSortedProb(1);
        s1 = orderCIM( EXs1 );  % Best Matching Cluster Index
        
        % Update a winner Cluster
        bestWeight = (CountCluster(1,s1)/(CountCluster(1,s1)+1))*weight(s1,:) + (1/(CountCluster(1,s1)+1))*pattern;
        
        % Calculate CIM between winner cluster and pattern for Vigilance Test
        bestCIM = CIM(pattern, bestWeight, cimSig);
        
        
        % Vigilance Test
        if bestCIM <= maxCIM
            % Match Success   
            % Update Parameters
            weight(s1,:)        = bestWeight;
            CountCluster(1, s1) = CountCluster(1, s1) + 1;
            
            % Calculate CIM based on t-th and (t+1)-th cluster position as an Error state.
            Ecim = CIM(weight(s1,:), bestWeight, cimSig); % If CountCluster is large, Ecim goes to zero.
            
            % If an Error become small comparing with previous state, update ErrCIM.
            ErrCIM(1, s1) = min(ErrCIM(1, s1), Ecim);
            
            % Create an edge between s1 and s2 clusters.
            if EXnumClusters >= 2
                EXs2 = numSortedProb(2);
                s2 = orderCIM( EXs2 ); % Second Matching Cluster Index
                if CIM(pattern, weight(s2,:), cimSig) <= maxCIM % Vigilance Test for s2 cluster.
                    edge(s1,s2) = 1;
                    edge(s2,s1) = 1;
                    NewEdgedNode(1,s1) = 1;
                end
            end
            
        else
            % Match Fail
            % Add Cluster
            numClusters                  = numClusters + 1;
            weight(numClusters,:)        = pattern;
            CountCluster(1, numClusters) = 1;
            NewEdgedNode(1, numClusters) = 0;
            ErrCIM(1, numClusters)       = 1;
            edge(numClusters,:)          = 0;
            edge(:,numClusters)          = 0;
            
        end % end Vigilance Test

    end % NstateCIM test
        
    
    
    % Topology Reconstruction
    if mod(sampleNum, Lambda) == 0
        
        % -----------------------------------------------------------------
        % Delete Node based on number of neighbors
        nNeighbor = sum(edge);
        deleteNodeEdge = (nNeighbor == 0);
        
        if sum(deleteNodeEdge) ~= size(weight,1)
            % Delete process
            edge(deleteNodeEdge, :) = [];
            edge(:, deleteNodeEdge) = [];
            weight(deleteNodeEdge, :) = [];
            numClusters = numClusters - sum(deleteNodeEdge);
            CountCluster(:, deleteNodeEdge) = [];
            NewEdgedNode(:, deleteNodeEdge) = [];
            ErrCIM(:, deleteNodeEdge) = [];
        end
        
        % -----------------------------------------------------------------
        % Delete Node based on ErrCIM
        [stateEC, posEC] = sort(ErrCIM, 'ascend');
        highEC = ( stateEC > maxCIM*2);
        deleteNodeEC = posEC(highEC);
        
        if size(deleteNodeEC,2) ~= size(weight,1)
            % Delete process
            edge(deleteNodeEC, :) = [];
            edge(:, deleteNodeEC) = [];
            weight(deleteNodeEC, :) = [];
            numClusters = numClusters - size(deleteNodeEC,2);
            CountCluster(:, deleteNodeEC) = [];
            NewEdgedNode(:, deleteNodeEC) = [];
            ErrCIM(:, deleteNodeEC) = [];
        end
        
        % -----------------------------------------------------------------
        % Delete Intersections of edge
        [weight, edge, NewEdgedNode] = DeleteIntersection(weight, edge, NewEdgedNode, cimSig);
        
    end % end topology reconstruction
    
end % end numSample



% -------------------------------------------------------------------------
% Delete Node based on number of neighbors
nNeighbor = sum(edge);
deleteNodeEdge = (nNeighbor == 0);

if sum(deleteNodeEdge) ~= size(weight,1)
    % Delete process
    edge(deleteNodeEdge, :) = [];
    edge(:, deleteNodeEdge) = [];
    weight(deleteNodeEdge, :) = [];
    numClusters = numClusters - sum(deleteNodeEdge);
    CountCluster(:, deleteNodeEdge) = [];
    NewEdgedNode(:, deleteNodeEdge) = [];
    ErrCIM(:, deleteNodeEdge) = [];
end


% -------------------------------------------------------------------------
% Delete Node based on ErrCIM
[stateEC, posEC] = sort(ErrCIM, 'ascend');
highEC = ( stateEC > maxCIM*2 );
deleteNodeEC = posEC(highEC);

if size(deleteNodeEC,2) ~= size(weight,1)
    % Delete process
    edge(deleteNodeEC, :) = [];
    edge(:, deleteNodeEC) = [];
    weight(deleteNodeEC, :) = [];
    numClusters = numClusters - size(deleteNodeEC,2);
    CountCluster(:, deleteNodeEC) = [];
    NewEdgedNode(:, deleteNodeEC) = [];
    ErrCIM(:, deleteNodeEC) = [];
end


% -------------------------------------------------------------------------
% Delete intersections of edge
[weight, edge, NewEdgedNode] = DeleteIntersection(weight, edge, NewEdgedNode, cimSig);


% -------------------------------------------------------------------------
% Cluster Labeling based on edge
connection = graph(edge ~= 0);
LebelCluster = conncomp(connection);



net.numClusters = numClusters;
net.weight = weight;
net.CountCluster = CountCluster;
net.LebelCluster = LebelCluster;
net.NewEdgedNode = NewEdgedNode;
net.edge = edge;
net.ErrCIM = ErrCIM;

end



% Kernel Bayes Rule
function [Po, posteriorMean] = KernelBayesRule(pattern, weight, paramKBR)

% Xi : pattern
% Yj : weight
% Pr : prior probability of Y

meanU = mean(pattern,1);

% Parameters for Kernel Bayes Rule
Sig = paramKBR.Sig;           % Kernel bandwidth
numNode = paramKBR.numNode;   % Number of Nodes
Eps = paramKBR.Eps;           % Scaling Factor
Delta = paramKBR.Delta;       % Scaling Factor
gamma = paramKBR.gamma;       % Scaling Factor
Pr = paramKBR.prior;          % Prior Probability

% Calculate Gram Matrix
Gy = Gramian(weight, weight, Sig); % Gy
Gx = Gramian(Pr, Pr, Sig);         % Gx

m_hat = zeros(numNode,1);
tmp = zeros(size(Pr,1),size(pattern,1));
for i=1:size(Pr,1)
   for j=1:size(pattern,1)
       tmp(i,j) = gamma(j) * gaussian_kernel(pattern(j,:), Pr(i,:), Sig); % kx(.,Pr)
   end
   m_hat(i) = sum(tmp(i,:),2);
end

mu_hat = numNode \ (Gx + numNode * Eps * eye(numNode)) * m_hat;
Lambda = diag(mu_hat);
LG = Lambda * Gy;
R = LG \ (LG^2 + Delta * eye(numNode)) * Lambda;
ky = gaussian_kernel(weight, meanU, Sig);
tmp_m_hatQ = R * ky;
tmp_m_hatQ( tmp_m_hatQ < 0 ) = 0;

Po = tmp_m_hatQ / sum(tmp_m_hatQ); % Posterior Probability  m_hatQ
posteriorMean = weight'*Po; % Estimated Mean

end


% Gram Matrix
function gram = Gramian(X1, X2, sig)
a=X1'; b=X2';
if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))];  
  b = [b; zeros(1,size(b,2))];  
end 
aa=sum(a.*a); bb=sum(b.*b); ab=a'*b;  
D = sqrt(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

gram = exp(-(D.^2 / (2 * sig.^2)));
end


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
[n, att] = size(Y);
g_Kernel = zeros(n, att);

for i = 1:att
    g_Kernel(:,i) = GaussKernel(X(i)-Y(:,i), sig);
end

ret0 = GaussKernel(0, sig);
ret1 = mean(g_Kernel, 2);

cim = sqrt(ret0 - ret1)';
end

% Gaussian Kernel
function g_kernel = GaussKernel(sub, sig)
g_kernel = exp(-sub.^2/(2*sig^2));
% g_kernel = 1/(sqrt(2*pi)*sig) * exp(-sub.^2/(2*sig^2));
end


% Gaussian Kernel
function g_kernel = gaussian_kernel(X, W, sig)
nrm = sum(bsxfun(@minus, X, W).^2, 2);
g_kernel = exp(-nrm/(2*sig^2));
end



% Delete intersections of edge
function [weight, edge, NewEdgedNode] = DeleteIntersection(weight, edge, NewEdgedNode, cimSig)

% for d = 1:size(weight,1); % Search all nodes
for d = find(NewEdgedNode == 1) % Search only new edged nodes
    
    node1 = find(edge(d,:)); % Neighbors of d-th node
    if size(node1,1) >= 1
       posX1 = weight(d,:); % position of d-th node
        for m = 1:size(node1,2) % Search all neighbors of d-th nodes
            posY1 = weight(node1(m),:); % position of m-th neighbor node of d-th node
            for h = 1:size(node1,2)
                target2 = node1(h);
                node2 = find(edge(target2,:)); % Neighbors of m-th node
                posX2 = weight(target2,:); % position of h-th neighbor node of m-th node
                for k = 1:size(node2,2)
                    posY2 = weight(node2(k),:); % position of k-th neighbor node of h-th node
                    isConvex = findIntersection(posX1, posY1, posX2, posY2); % find intersections
                    if isConvex == 1 % If intersection is exist, delete edge which has larger CIM.
                        cim1 = CIM(weight(d,:), weight(node1(m),:), cimSig);
                        cim2 = CIM(weight(target2,:), weight(node2(k),:), cimSig);
                        if cim2 >= cim1
                            edge(target2, node2(k)) = 0;
                            edge(node2(k), target2) = 0;
                        else
                            edge(d, node1(m)) = 0;
                            edge(node1(m), d) = 0;
                        end
                    end % end isConvex
                end % end k
            end % end h
        end % end m  
    end

end % end d

NewEdgedNode = zeros(size(NewEdgedNode));

end

% Check intersection of edges
function [isConvex] = findIntersection(A, B, C, D)

F1  = B(:,1)-D(:,1);
F2  = B(:,2)-D(:,2);
M11 = B(:,1)-A(:,1);
M21 = B(:,2)-A(:,2);
M12 = C(:,1)-D(:,1);
M22 = C(:,2)-D(:,2);
deter = M11.*M22 - M12.*M21;
lambda = -(F2.*M12-F1.*M22)./deter;
gamma = (F2.*M11-F1.*M21)./deter;

% E = (lambda*[1 1]).*A + ((1-lambda)*[1 1]).*B;
% isConvex = (0 <= lambda & lambda <= 1)  & (0 <= gamma & gamma <= 1);

isConvex = (0 < lambda & lambda < 1)  & (0 < gamma & gamma < 1) ;
isConvex = isConvex';

end



