%load & parse network

t=1e-20;
dt = .1*t;
tmax=1e-6;
T9 = 7;
rho = 1e8;
networkFile = fopen('~/Desktop/Research/FERN/fernPartialEqCPU/data/CUDAnet_3.inp','r');
reactionFile = fopen('~/Desktop/Research/FERN/fernPartialEqCPU/data/rateLibrary_3.data','r');

%parse Network File
y = [0 0 0 0]; %abundances
speciesID = 1;
numSpecies = 4;

while (~feof(networkFile))
    speciesBlock = textscan(networkFile,'%s',4,'Delimiter','\n');
    speciesHeader = textscan(speciesBlock{1}{1},'%s',5,'Delimiter',' ');
    y(speciesID) = str2double(speciesHeader{1}{5});
    speciesID = speciesID+1;
end

%parse Reaction file
%calculate rates
k = [0 0 0 0 0 0];
%memberID = [0 0 0 0 0 0 0 0];
p = [0 0 0 0 0 0 0]; 
reacID = 1;

while (~feof(reactionFile) && reacID < 9)
    reacBlock = textscan(reactionFile,'%s',8,'Delimiter','\n');
    reacHeader = textscan(reacBlock{1}{1},'%s',10,'Delimiter',' ');
    memberID = str2double(reacHeader{1}{3});
    statFac = str2double(reacHeader{1}{9});
    params = textscan(reacBlock{1}{2},'%s',7,'Delimiter',' ');
    for i = 1:7
        params{1}{i} = str2double(params{1}{i});
    end

    if(memberID ~= 0 && memberID ~= 2)
        k(reacID/2) = k(reacID/2) + exp(params{1}{1}+(params{1}{2}/T9)+(params{1}{3}/(T9^(1/3)))+params{1}{4}*(T9^(1/3))+params{1}{5}*T9+params{1}{6}*(T9^(5/3))+params{1}{7}*log(T9));
    else
        k((reacID+1)/2) = exp(params{1}{1}+(params{1}{2}/T9)+(params{1}{3}/(T9^(1/3)))+params{1}{4}*(T9^(1/3))+params{1}{5}*T9+params{1}{6}*(T9^(5/3))+params{1}{7}*log(T9));
        %^^ putting odd reacID into its rate slot
    end
    
    for i = 1:6
        k(i);
    end

    reacID = reacID+1;
end

fclose(networkFile);
fclose(reactionFile);




%eigenvalues of flux matrix
statFac(1)
kf1 = k(1)*rho*rho*statFac(1);
kr1 = k(2);
kf2 = k(3)*rho;
kr2 = k(4);
kf3 = k(5);
kr3 = k(6);

count = 1;
yplot1 = [];
yplot2 = [];
yplot3 = [];
tplot = [];
dtplot = [];
while t < tmax
    L=[-kf2*y(1)*y(1), kr2-kf1*y(1), kr1-kf3*y(1), kr3;
        kf2*y(1)*y(1), -kr2-kf1*y(1), kr1, 0;
        kf1*y(2)-kf3*y(3), 0, -kr1, kr3;
        kf3*y(3), 0, 0, -kr3]

    'eigenvalues'
    eigenL = eig(L)

    %get largest eigenvalue
    lambda = 0;
    absEigL = abs(eigenL);
    for i = 1:4
        if gt(absEigL(i),abs(lambda))
            lambda = eigenL(i);
        end
    end

    'largest eigenvalue'
    lambda

    'delta t'
    dt = abs(1/lambda);
    if dt > .1*t
        dt = .1*t;
    end
    'dydt'
    dydt = L*transpose(y)
    yplot1(count) = y(1);
    yplot2(count) = y(2);
    yplot3(count) = y(3);
    tplot(count) = t;
    dtplot(count) = dt;
    for i = 1:numSpecies
        y(i) = y(i)+dt*(dydt(i))
    end
    t=t+dt
    count = count + 1;
end
loglog(tplot,yplot1,tplot, yplot2,tplot, yplot3)
legend('He', 'C', 'O')
axis([1e-16,1e-6,1e-9,5e-1])

loglog(tplot,dtplot)
