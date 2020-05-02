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
function plotTKBA(DATA, net)


w = net.weight;
edge = net.edge;
[N,~] = size(w);
label = net.LebelCluster;

color = [
    [1 0 0]; 
    [0 1 0]; 
    [0 0 1]; 
    [1 0 1];
    [0.8500 0.3250 0.0980];
    [0.9290 0.6940 0.1250];
    [0.4940 0.1840 0.5560];
    [0.4660 0.6740 0.1880];
    [0.3010 0.7450 0.9330];
    [0.6350 0.0780 0.1840];
];
m = length(color);

whitebg('black')
plot(DATA(:,1),DATA(:,2),'cy.');
hold on;

for i=1:N-1
    for j=i:N
        if edge(i,j)==1
            plot([w(i,1) w(j,1)],[w(i,2) w(j,2)],'w','LineWidth',1.5);
        end
    end
end
for k = 1:N
    plot(w(k,1),w(k,2),'.','Color',color(mod(label(1,k)-1,m)+1,:),'MarkerSize',35);
end

axis equal
grid on
hold off
axis([0 1 0 1]);
pause(0.01);

end