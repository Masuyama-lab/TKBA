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


MIter = 1;     % Number of iterations
NR = 0.1; % Noise Rate [0,1]

% Load Data
load 2D_ClusteringDATASET
DATA = [data(:,1) data(:,2)];

% scaling [0,1]
DATA = normalize(DATA,'range');

% Noise Setting [0,1]
if NR > 0
    noiseDATA = rand(size(DATA,1)*NR, size(DATA,2));
    DATA(1:size(noiseDATA,1),:) = noiseDATA;
end

% Parameters of TKBA ======================================================
TKBAnet.edge = zeros(2,2); % Initial connections (edges) matrix
TKBAnet.numClusters = 0;   % Number of clusters
TKBAnet.weight = [];       % Mean of cluster
TKBAnet.CountCluster = []; % Counter for each cluster
TKBAnet.NewEdgedNode = []; % Node which creates new edge.
TKBAnet.ErrCIM = [];       % CIM between clusters

TKBAnet.cimSig = 0.05;     % Kernel Bandwidth for CIM
TKBAnet.kbrSig = 1.0;      % Kernel Bandwidth for KBR
TKBAnet.maxCIM = 0.2;     % Vigilance Parameter by CIM [0~1]
TKBAnet.Lambda = 400;      % Interval for Node deletion and topology construction
% =========================================================================


for nitr = 1:MIter
    fprintf('Iterations: %d/%d\n',nitr,MIter);
    
    % Randamize data
    ran = randperm(size(DATA,1));
    DATA = DATA(ran,:);
    
    TKBAnet = TKBA(DATA, TKBAnet);
    plotTKBA(DATA, TKBAnet);
    
end



