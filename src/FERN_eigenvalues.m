%load & parse network

t=1e-20;
dt = .1*t;
tmax=1e-5;
T9 = 7;
rho = 1e8;
checkAsy = 0;
networkFile = fopen('~/Desktop/Research/FERN/fernPartialEqCPU/data/CUDAnet_3.inp','r');
reactionFile = fopen('~/Desktop/Research/FERN/fernPartialEqCPU/data/rateLibrary_3.data','r');

%parse Network File
speciesID = 0;

%parse speciesFile
while (~feof(networkFile))
    speciesID = speciesID+1;
    speciesBlock = textscan(networkFile,'%s',4,'Delimiter','\n');
    speciesHeader(speciesID) = textscan(speciesBlock{1}{1},'%s',5,'Delimiter',' ');
    
end

numSpecies = speciesID;
y = zeros(1,numSpecies);
for i = 1:numSpecies
    %get initial abundances
   y(i) = str2double(speciesHeader{i}{5});
end

%parse Reaction file
%calculate rates
reacID = 0;
rgID=0; %first rgID *does* start at 1. see line 50.

%parese reacFile, get stuff for reactions -- --
while (~feof(reactionFile))
    reacID = reacID+1;
    reacBlock(reacID) = textscan(reactionFile,'%s',8,'Delimiter','\n');
    reacHeader(reacID) = textscan(reacBlock{reacID}{1},'%s',10,'Delimiter',' ');
    
    %member of RG
    memberID(reacID) = str2double(reacHeader{reacID}{3});
    numReactants(reacID) = str2double(reacHeader{reacID}{5});
    numProducts(reacID) = str2double(reacHeader{reacID}{6});
    statFac(reacID) = str2double(reacHeader{reacID}{9});
    %get 7 parameters to calculate reaction rate
    paramsStr(reacID) = textscan(reacBlock{reacID}{2},'%s',7,'Delimiter',' ');
    reactantsStr(reacID) = textscan(reacBlock{reacID}{7},'%s',numReactants(reacID),'Delimiter',' ');
    productsStr(reacID) = textscan(reacBlock{reacID}{8},'%s',numProducts(reacID),'Delimiter',' ');
end
numReactions = reacID;

fclose(networkFile);
fclose(reactionFile);

%loop through reactions -- -- , 
%1) calculate rate parameters, 
%2) set up reaction groups,
%3) get reactants and products for each reaction
Rate = zeros(1,numReactions);

%create global reactants matrix. A=(a,b): Holds all reactants (b) in each 
%reaction (a). There are a maximum of 3 reactants (for 3-body reactions),
%so size(A) = (numReactions, 3).
%NOTE: He4 is SpeciesID = 1
reactantsMatrix = zeros(numReactions, 3);
%repeat above for products
productsMatrix = zeros(numReactions, 3);

%initiate array to point reactions to their reaction group
RGbyReac = zeros(1,numReactions);
'Rates';
for i = 1:numReactions
    
    %Update statistical factor to accommodate for multi-body reactions
    statFac(i) = statFac(i)*(rho.^(numReactants(i)-1));
        
    
    %Don't need to save the params for global, so just overwrite during
    %loop.
    paramsArray = zeros(1,7);

    for j = 1:7
       %build array holding parameters to calculate rate for this reac.
       paramsArray(1,j) = str2double(paramsStr{i}{j});
    end
    
    %get reactants in this reaction (will contrib to Fminus for each
    %reactant)
    reactants = zeros(1,numReactants(i));
    for j = 1:numReactants(i)
        reactants = reactantsStr(i);
        %NOTE: add (+) 1 to differentiate when reactantsMatrix has 0, which
        %will imply an empty entry, ie. no more reactants. In fact, we're
        %using He4 is SpeciesID = 1. 
        reactantsMatrix(i,j) = str2double(reactants{1}{j})+1;
    end

    %get products in this reaction (will contrib to Fplus for each
    %product)
    for j = 1:numProducts(i)
        products = productsStr(i);
        productsMatrix(i,j) = str2double(products{1}{j})+1;
    end

    %calculate rates -- -- 

    %calculate final rate for this reaction
    Rate(i) = statFac(i)*exp(paramsArray(1,1)+(paramsArray(1,2)/T9)+(paramsArray(1,3)/(T9^(1/3)))+paramsArray(1,4)*(T9^(1/3))+paramsArray(1,5)*T9+paramsArray(1,6)*(T9^(5/3))+paramsArray(1,7)*log(T9));

    %Set up reaction groups -- -- 
   
    if(memberID(i) == 0)
        %then, start new reaction group
        %!!! reaction group IDs start at 1, not 0.
        rgID = rgID + 1;
    end
    
    %reactions 0 and 1 are forward reactions
    %reactions 2 and 3 are reverse reactions
    %sum 0 and 1 rates (kf), then sum 2 and 3 rates (kr). 
    %NOTE: for this to work, we must make sure all reaction files are set
    %up in this way... First two together, second two together, etc.
    if(memberID(i) == 1)
        kf(rgID) = (Rate(i-1) + Rate(i));
        for q = 1:numReactants(i)
            %make reactant matrix by RG for forward reactions
            RGforwardReactantsMatrix(rgID, q) = reactantsMatrix(i, q);
            %number Reactants for forward RG
            numReactantsRGf(rgID) = numReactants(i);
        end
    elseif(memberID(i) == 3)
        kr(rgID) = (Rate(i-1) + Rate(i));
        for q = 1:numReactants(i)
            %make reactant matrix by RG for forward reactions
            RGreverseReactantsMatrix(rgID, q) = reactantsMatrix(i, q);
            %number Reactants for forward RG
            numReactantsRGr(rgID) = numReactants(i);
        end
    end
    
    %Array to point reactions to their reaction group
    RGbyReac(i) = rgID;
end
numRG = rgID;

RGforwardReactantsMatrix
RGreverseReactantsMatrix
numReactantsRGf
numReactantsRGr


%Number of Reactions that contribute to increasing/decreasing this species
numFplus = zeros(1,numSpecies);
numFminus = zeros(1,numSpecies);
totalFplus = 0;
totalFminus = 0;
%Array holding the reactions responsible for increasing each isotope,
%separated by numFplus(i) where i is this isotope. So if there are two
%reactions that increase He4, FplusReacs(1) and FplusReacs(2) are for He4.
%Then if there are two reactions that increase C12, FplusReacs(3) and
%FplusReacs(4) are for C12, etc. We'll access each by using the sum of
%numFplus(). This allows for a much smaller single vector array, rather 
%than a large matrix, which would have lots of empty slots as some isotopes 
%have many more associated reactions and reflects closer what is done in 
%the C code. Making FplusReacs at least as large as considering that **at 
%most half of all reactions increase each isotope. This is larger than we 
%need, but don't know exact dimesions yet. Will resize after next section.
tempFplusReacs = zeros(1,numSpecies*(numReactions/2));
tempFminusReacs = zeros(1,numSpecies*(numReactions/2));

%TODO Move some of the next part into the integration while loop
for i = 1:numSpecies
   for j = 1:numReactions
       for n = 1:numReactants(j)
          if (i == reactantsMatrix(j,n))
             %then this (j) is a reaction that depletes this isotope.
             %starting point for next FminusReacs entry
             totalFminus = totalFminus + 1;
             %keeping track of how many reactions deplete this isotope
             numFminus(i) = numFminus(i) + 1;
               
             %Set ID of reaction that is responsible for increasing this
             %isotope
             tempFminusReacs(totalFminus) = j;
             %Once we've identified that this reaction depletes, break out
             %of for to avoid overcounting if this species appears as more
             %than one reactant in reaction. ie. he4+he4+he4 --> c12 
             break;
          end
       end
       for n = 1:numProducts(j)
          if (i == productsMatrix(j,n))
             %then this (j) is a reaction that increases this isotope.
             %starting point for next FminusReacs entry
             totalFplus = totalFplus + 1;
             %keeping track of how many reactions deplete this isotope
             numFplus(i) = numFplus(i) + 1;
             %Set ID of reaction that is responsible for increasing this
             %isotope
             tempFplusReacs(totalFplus) = j;
             %Once we've identified that this reaction increases, break out
             %of for to avoid overcounting if this species appears as more
             %than one product in reaction. ie. c12 --> he4+he4+he4
             break;
          end
       end
   end
end

%resize FplusReacs and FminusReacs to save dat memory, tho.

FplusReacs = zeros(1, totalFplus);
FminusReacs = zeros(1, totalFminus);

for i = 1:totalFplus
   FplusReacs(i) = tempFplusReacs(i); 
end

for i = 1:totalFminus
   FminusReacs(i) = tempFminusReacs(i); 
end

clear tempFplusReacs tempFminusReacs

%Okay, now that I've freed up that space... really just a formalism... but
%still, it's cool. And now that I have arrays pointing isotopes to the
%reactions that increase and decrease them,
%Time to go into the while loop and calculate the fluxes
%using the current abundance values. 

%eigenvalues of flux matrix
% kf1 = k(1)*rho*rho*statFac(1);
% kr1 = k(2);
% kf2 = k(3)*rho;
% kr2 = k(4);
% kf3 = k(5);
% kr3 = k(6);

numTimesteps = 1;
yplot1 = [];
yplot2 = [];
yplot3 = [];
tplot = [];
dtplot = [];

FminusSum = zeros(1,numSpecies);
FplusSum = zeros(1,numSpecies);
L = zeros(numSpecies);
%for building the matrix L. If reaction used
%reacUsed = zeros(1, numReactions) TODO
Flux = zeros(1,numReactions);
setNextOut = 0;
alldtcount = 1;
alldt = [];
while t < tmax
    %Loop through all reactions and set up Flux arrays for this timestep
    for i = 1:numReactions
        %Give Flux a value so its product with y won't be zero. 
        Flux(i) = Rate(i);
        %go through all reactants that allow this reaction to occur
%         sprintf('reaction(%d) has these reactants:', i)
        for j = 1:numReactants(i)
            Flux(i) = Flux(i)*y(reactantsMatrix(i,j));
%             sprintf('reactant(%d) has y(%d): %e', j, reactantsMatrix(i,j), y(reactantsMatrix(i,j)))
        end
%         sprintf('Yielding Flux(%d): %e', i, Flux(i))
        %check if Fluxes we just calculated are same as what's in L matrix:
        if mod(i,2) == 0
            %This is the second reaction in a reaction group. i is even
            %TotalFlux holds all total Fplus and total Fminus for each RG
            %For example, since reacs 1 and 2 are forward triple-?, their
            %total Fplus is in TotalFlux(1). Similarly, since reacs 3 and 4
            %are reverse triple-?, their total Fminus is TotalFlux(2). I'll
            %parse into Fplus/Fminus by RG next, and compare with what I 
            %have in the L-matrix.
            TotalFlux(i/2) = Flux(i) + Flux(i-1);
        end
    end
    
    for i = 1:(numReactions/2)
       if mod(i,2) == 0
          %i is even. This is the second item for a specific reaction in 
          %TotalFlux, which means it is the total flux for a reverse 
          %reaction, Fminus(RGid). 
          RGfFlux(i/2) = TotalFlux(i);
       else
           %i is odd. This is the first item for a specific reaction in
           %Total Flux, which means it is the total flux for a forward
           %reaction, Fplus(RGid).
           RGrFlux((i/2)+.5) = TotalFlux(i);
       end
    end
    
%     TotalFlux
%     RGfFlux
%     sprintf('Fminus should be \nfor RG1: %e \nRG2: %e', (kr(1)*y(2)), (kr(2)*y(3)))
%     sprintf('Fplus should be \nfor RG1: %e \nRG2: %e', (kf(1)*y(1)*y(1)*y(1)), (kf(2)*y(1)*y(2)))
%     RGrFlux
    
    %Loop through species and set up FluxSum arrays
    %A counter to keep track of the starting index in FplusReacs array that
    %indicates set of reactions responsible for increasing this(i) species
    speciesFplusIndex = 1;
    speciesFminusIndex = 1;
    for i = 1:numSpecies 
        %populate FplusSum using reaction groups instead of Reacs (more
        %like literature does it).
        for j = 1:numRG
            %check if this species is a reactant is in forward RG
            %how many times does this reactant appear in this reaction?
            %This will be multiplied to the total flux from this RG
            %If it only appears once, speicesRepeat = 1. 
            speciesForwardRepeat = 0;
            speciesReverseRepeat = 0;
            %^^^NOTE: ACTUALLY WE DON'T NEED THIS: If we loop through,
            %every time the i = RGforwardReactantsMatrix(j,l), which is all
            %three times for triple-? in regards to i = he4, the Flux will
            %be added three times. So no need to multiply by 3.
            for l = 1:numReactantsRGf(j)
                 if i == RGforwardReactantsMatrix(j,l)
                     speciesForwardRepeat = speciesForwardRepeat + 1;
                 end
            end
            for l = 1:numReactantsRGr(j)
                 if i == RGreverseReactantsMatrix(j,l)
                     speciesReverseRepeat = speciesReverseRepeat + 1;
                 end
            end
            if speciesForwardRepeat > 0
                %then the species is being depleted by j.
                FminusSum(i) = FminusSum(i) + speciesForwardRepeat*RGfFlux(j);
            end
            
            if speciesReverseRepeat > 0
                %then the species is being depleted by j.
                FplusSum(i) = FplusSum(i) + speciesReverseRepeat*RGrFlux(j);
            end
            
            for l = 1:numReactantsRGr(j)
                 if i == RGreverseReactantsMatrix(
            %now do it for reverse reactions
            
        end
        
        
        
        
%         %populate FplusSum -- --
%         startingFplusIndex = speciesFplusIndex;
%         %subtract 1 to keep from spilling over into next species' reactions
%         endingFplusIndex = speciesFplusIndex+numFplus(i)-1;
%         for j = startingFplusIndex:endingFplusIndex
%             %Add to FplusSum for this species (i) with Flux(j) from this 
%             %reaction (j)
%             %These reactions, FplusReacs(j), increase species (i).
%             FplusSum(i) = FplusSum(i) + Flux(FplusReacs(j));
%             
%             %Build L matrix, all those NOT L(i,i)-- --
%             %Row is d(y_i)/dt, column 1 is flux/y_a, column
%             %2 is flux/y_b, etc. where flux is the flux contributes to
%             %fplus for (i), with the abundance for this column's species
%             %divided out so it can be multiplied by the vector y later...
%             %This is just to find the eigenvalues of L. The reactants for 
%             %Fplus are never species (i). so, L_1,1 will be built in the
%             %Fminus for loop coming up.
%             for k = 1:numSpecies
% %                 'species k in L(i,k)'
% %                 k
%                 %All Fplus reactions do not have species i as a reactant.
%                 %we will build L(i,i) in the Fminus loop below.
%                 % loop through all reactants of the reaction FplusReacs(j)
%                 for m = 1:numReactants(FplusReacs(j))
% %                     'the reaction that increases species (i)'
% %                     FplusReacs(j)
% %                     'the reactantID within this reaction (not species ID)'
% %                     m
%                     %if the species(k) is a reactant in the reaction we're
%                     %looking at, FplusReacs(j), and that reactant is not
%                     %the same as species(i) (as that will be taken care of
%                     %in Fminus loop below):
%                     if ((k == reactantsMatrix(FplusReacs(j),m)) && (i ~= k))
% %                         'species k is indeed a reactant, and is not species (i)'
% %                         'this reactions flux'
% %                         Flux(FplusReacs(j))
% %                         (Flux(FplusReacs(j))/y(k))
%                         %then this speices(k) is a reactant in this 
%                         %reaction that contributes to increasing species
%                         %(i). Add its flux/y(k) to L(i,k).
%                         %don't bother changing L(i,k) if the y vector
%                         %component, species(k), has zero abundance. It will
%                         %automatically be zero, as the Flux(i) should be
%                         %zero. (otherwise ends in a NaN in L)
%                         if(y(k) ~= 0)
%                             L(i,k) = L(i,k) + (Flux(FplusReacs(j))/y(k));
%                         end
%                     end
%                 end
%             end
%             %end Bulid Fplus parts of L Matrix, all those not L(i,i)
%         end
% 
%         %populate FminusSum -- --
%         startingFminusIndex = speciesFminusIndex;
%         %subtract 1 to keep from spilling over into next species' reactions
%         endingFminusIndex = speciesFminusIndex+numFminus(i)-1;
%         for j = startingFminusIndex:endingFminusIndex
% %             Output for testing -- -- TODO Remove
% %             'species'
% %             i
% %             'reaction'
% %             FminusReacs(j)
% %             'Flux from this reaction'
% %             Flux(FminusReacs(j))
%             %Add to FplusSum for this species (i) with Flux(j) from this 
%             %reaction (j)
%             %These reactions, FplusReacs(j), increase species (i).
%             FminusSum(i) = FminusSum(i) + Flux(FminusReacs(j));
%             
%             
%             %Build L(i,k) matrix where i == k -- --
%             
%             for k = 1:numSpecies
% %                 'species k in L(i,k)'
% %                 k
%                 %All Fminus reactions have species i as a reactant.
%                 %Here we build L(i,i). L(i,k) where i ~= k was built above.
%                 % loop through all reactants of the reaction FminusReacs(j)
%                 for m = 1:numReactants(FminusReacs(j))
% %                     'the reaction that decreases species (i)'
% %                     FminusReacs(j)
% %                     'the reactantID within this reaction (not species ID)'
% %                     m
%                     %if the species(k) is a reactant in the reaction we're
%                     %looking at, FminusReacs(j), and that reactant is
%                     %INDEED the same as species(i):
%                     if ((k == reactantsMatrix(FminusReacs(j),m)) && (i == k))
% %                         'species k is indeed a reactant, and is INDEED species (i)'
% %                         'this reactions flux'
% %                         Flux(FminusReacs(j))
% %                         (Flux(FminusReacs(j))/y(k))
%                         %then this speices(k) is a reactant in this 
%                         %reaction that contributes to depleting species
%                         %(i). Add its flux/y(k) to L(i,k).
%                         %don't bother changing L(i,k) if the y vector
%                         %component, species(k), has zero abundance. It will
%                         %automatically be zero, as the Flux(i) should be
%                         %zero. (otherwise ends in a NaN in L)
%                         if(y(k) ~= 0)
%                             L(i,k) = L(i,k) + (Flux(FminusReacs(j))/y(k));
%                         end
%                     end
%                 end
%             end
%             %end Bulid Fminus parts of L Matrix, all those INDEED L(i,i)
%         end
%         
%         
%         
%         
%         %Go to next species' reactions
%         speciesFplusIndex = speciesFplusIndex + numFplus(i);
%         speciesFminusIndex = speciesFminusIndex + numFminus(i);
    end
%     'Programmed L'
%     L

    
    %OLD Manually built L matrix -- --
    %populate flux matrix... it's a linearized matrix... so really "d/dt"
    L=[-3*kf(1)*y(1)*y(1)-kf(2)*y(2), kr(1), kr(2);
        3*kf(1)*y(1)*y(1), -kr(1)-kf(2)*y(1), kr(2);
        kf(2)*y(2), 0, -kr(2)];
    'Manual L';
    L;

    
    %'eigenvalues'
    eigenL = eig(L);

    %get largest eigenvalue
    lambda = 0;
    absEigL = abs(eigenL);
    for i = 1:numSpecies
        if gt(absEigL(i),abs(lambda))
            lambda = eigenL(i);
        end
    end

    %'largest eigenvalue'
    lambda;
    
    %'delta t'
    dt = abs(1/lambda);
    if dt > .1*t
        dt = .1*t;
    end
    alldt(alldtcount) = dt;
    alldtcount = alldtcount + 1;
   
    %'dydt'
    %calculate abundance rate. populates dydt vector, which contains each 
    %the change in abundances for each species. Instead of doing 
    %Y + (FplusSum - FminusSum) * dt; 
    %as in FERN (line 656 in kernels), dydt is the final change.
    dydt = L*transpose(y);
    sprintf('%e\n', transpose(y));
    %!!! TURN THIS INTO FUNCTION !!!
    %Update populations
    for i = 1:numSpecies
        %checkAsy     
        if ((FminusSum(i)*dt/y(i)) > 1.0 && checkAsy == 1)
            %do Asymptotic Update
            %This is exactly what's in FERN now... Change it such that
            %we're modifying the L-matrix... removing corresponding row an
            %column, corresponding to species (i). 
            y(i) = (y(i)+(FplusSum(i)*dt))/(1+FminusSum(i)*dt);
        else
            %Update abundances, Euler Update
            %This is what's in FERN:
            %y(i)+((FplusSum(i)-FminusSum(i))*dt)
            %This is what we'll use by using the L-matrix, which is
            %essentially what is in FERN. Just that FplusSum and FminusSum
            %are implicit in the components of L.
            
            y(i) = y(i)+dt*(dydt(i));
        end
    end
                %output 100 times during calculation
            if(log10(t) > -16)
                if(setNextOut == 0)
                    intervalLogt = (log10(tmax)-log10(t))/100;
                    nextOutput = log10(t);
                    setNextOut = 1;
                    OutCount = 0;
                end
                
                if(log10(t) >= nextOutput)
                   OutCount = OutCount + 1;
                   %print abundances
%                    sprintf('Output: %d/100\n abundances: \n He4: %e\n C12: %e\n O16: %e\n time: %f', OutCount, y, log10(t))
% %                    sprintf('kf: %e %e %e\nkr: %e %e %e \n', kf, kr)
%                    sprintf('Flux: \n %e\n %e\n %e\n %e\n %e\n %e\n %e\n %e\n', Flux)
                   %print time
                    nextOutput = nextOutput + intervalLogt;
                end
                
            end
    yplot1(numTimesteps) = y(1);
    yplot2(numTimesteps) = y(2);
    yplot3(numTimesteps) = y(3);
    tplot(numTimesteps) = t;
    dtplot(numTimesteps) = dt;
    t=t+dt;
    %number of timesteps
    numTimesteps = numTimesteps + 1;
end
sprintf('%e\n', alldt);
%plot abundances
loglog(tplot,yplot1,tplot, yplot2,tplot, yplot3, tplot, dtplot)
legend('He', 'C', 'O', 't')
axis([1e-16,1e-5,1e-9,5e-1])

%plot time vs timestep
%loglog(tplot,dtplot)

   