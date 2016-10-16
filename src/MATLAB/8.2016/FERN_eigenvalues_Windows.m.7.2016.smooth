%load & parse network
'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
t=1e-20;
dt = .1*t;
tmax=1e-8;
T9 = 7;
rho = 1e8;
checkAsy = 1;
pEquilOn = 1;
plotJavaCompare = 1;
networkFile = fopen('fern\data\CUDAnet_150.inp','r');
reactionFile = fopen('fern\data\rateLibrary_150.data','r');
if (plotJavaCompare == 1) 
    javaOutFile = fopen('fern\data\Y_150.out');
    C = textscan(javaOutFile, '%s','Delimiter','');
    fclose(javaOutFile);    

    %make each line in parse Y.out file components in an array.
    C = C{:};
    for i = 1:length(C)
        if(strcmp(C{i},'Plot_Times:'))
            javaPlotTimes = textscan(C{i+1},'%f','Delimiter',' ');
        end
        if(strcmp(C{i},'Abundances_Y(Z,N,time):'))
            javaStartAbundances = i
        end
        if(findstr(C{i},'JavaAsy'))
            javaEndAbundances = i
        end
    end
    counter = 1;
    for i = javaStartAbundances+1:javaEndAbundances-1
        if(mod(counter,2) == 0)
            holder = textscan(C{i},'%f','Delimiter',' ');
            for j = 1:length(holder{1})
                javaAbundances(counter/2,j) = holder{1}(j);
            end
        else
            holder = textscan(C{i},'%d','Delimiter',' ');
            for j = 1:length(holder{1})
                javaIsotope((counter/2)+.5,j) = holder{1}(j);
            end
        end
        counter = counter + 1;
    end
end
%parse Network File
speciesID = 0;

%parse speciesFile
while (~feof(networkFile))
    speciesID = speciesID+1;
    speciesBlock = textscan(networkFile,'%s',4,'Delimiter','\n');
    speciesHeader(speciesID) = textscan(speciesBlock{1}{1},'%s',5,'Delimiter',' ');
    
end

numSpecies = speciesID;
y = zeros(1, numSpecies);
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
%     ReacLabel(reacID) = reacHeader{reacID}{1};
    isReverse(reacID) = str2double(reacHeader{reacID}{8});
    reacClass(reacID) = str2double(reacHeader{reacID}{2});
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

%initialize reacVector which holds which isotopes are increasing and
%decreasing for each reaction. I will use this to build the Reaction Groups
%rather than isReverse, because isReverse is unreliable. 
reacVector = zeros(numReactions,numSpecies);

%create global reactants matrix. A=(a,b): Holds all reactants (b) in each 
%reaction (a). There are a maximum of 3 reactants (for 3-body reactions),
%so size(A) = (numReactions, 3).
%NOTE: He4 is SpeciesID = 1
reactantsMatrix = zeros(numReactions, 4);
%repeat above for products
productsMatrix = zeros(numReactions, 4);

%initiate array to point reactions to their reaction group
RGbyReac = zeros(1,numReactions);
tempRGclassbyRG = zeros(1,numReactions);
'Rates';
reacPlaced = zeros(1,numReactions);
reacParent = zeros(1,numReactions);
isRGmember = 0;
RGmemberCounter = 0;
for i = 1:numReactions
    
    %Update statistical factor to accommodate for multi-body reactions
    statFac(i) = statFac(i)*(rho.^(numReactants(i)-1));
    
    %get reactants in this reaction (will contrib to Fminus for each
    %reactant)
    reactants = zeros(1,numReactants(i));
    for j = 1:numReactants(i)
        reactants = reactantsStr(i);
        %NOTE: add (+) 1 to differentiate when reactantsMatrix has 0, which
        %will imply an empty entry, ie. no more reactants. In fact, we're
        %using He4 is SpeciesID = 1. 
        reactantsMatrix(i,j) = str2double(reactants{1}{j})+1;
        
        %subtract reactants to reaction vector for this reaction
        for x = 1:numSpecies
            if x == reactantsMatrix(i,j)
                reacVector(i,x) = reacVector(i,x) - 1; 
            end
        end
    end
    %get products in this reaction (will contrib to Fplus for each
    %product)
    for j = 1:numProducts(i)
        products = productsStr(i);
        productsMatrix(i,j) = str2double(products{1}{j})+1;
        
        %add products to reaction vector for this reaction
        for x = 1:numSpecies
            if x == productsMatrix(i,j)
                reacVector(i,x) = reacVector(i,x) + 1; 
            end
        end
    end
end
for i = 1:numReactions
    %Place Reactions into Reaction Groups:
    
    if(reacPlaced(i) == 0)
       %If this reaction has not yet been placed, stop reaction loop for a
       %second, then set up this RG, then continue loop.
       %first make this reaction the parent of the RG.
       RGmemberCounter = 0;
       memberID(i) = RGmemberCounter;
       rgID = rgID + 1;
       RGparent(rgID) = i;
       reacParent(i) = i;
       tempRGclassbyRG(rgID) = reacClass(i);
       RGbyReac(i) = rgID;
       
       sprintf('PARENT:\nReaction(%d), %s, has not yet been placed.\nIt is member(%d) of RG(%d)\nIt has RGclass(%d)\n', i, reacHeader{i}{1}, memberID(i), rgID, tempRGclassbyRG(rgID));
       %now loop through all reactions again, and check if each reaction's
       %reaction vector matches with the parent reaction. If so, add it to
       %the RG.
       
       for x = 1:numReactions
          for m = 1:numSpecies
             if(abs(reacVector(i,m)) ~= abs(reacVector(x,m)) || i == x)
                isRGmember = 0;
                %if any species is determined not to have been used the
                %same number of times in a reaction, it is not the same
                %reaction. Break out of the species loop:
                break;
             else
                %if a reaction vector component matches up, this might be a
                %member of the reaction group of reaction i. Keep looping
                %through all species to see if the reaction vectors match
                %entirely.
                isRGmember = 1;
             end
          end
          
          %if reaction x managed to show that all of its species
          %depletion/generation match that of reaction i, then add it to
          %the reaction group. 
          if(isRGmember == 1)
             for m = 1:numSpecies
                %first, check if this is a reverse reaction. Update that
                %which was read from the network file, as it is not to be
                %trusted.
                
                if(reacVector(i,m) ~= reacVector(x,m) && abs(reacVector(i,m)) == abs(reacVector(x,m)))
                   isReverse(x) = 1; 
                end
             end
             
             reacPlaced(x) = 1;
             RGmemberCounter = RGmemberCounter + 1;
             memberID(x) = RGmemberCounter;
             reacParent(x) = i;
             RGbyReac(x) = rgID;
          end
       end
    end
end
numRG = rgID
%print Reaction Group info
for i = 1:numRG
    sprintf('Reaction Group %d\nParent: reac(%d), %s', i, RGparent(i), reacHeader{RGparent(i)}{1});
    for j = 1:numReactions
        if RGparent(i) == reacParent(j)
            sprintf('reac(%d), %s, member(%d)', j, reacHeader{j}{1}, memberID(j));
        end
    end
end 
for i = 1:numReactions  
        
    %Don't need to save the params for global, so just overwrite during
    %loop.
    paramsArray = zeros(1,7);

    for j = 1:7
       %build array holding parameters to calculate rate for this reac.
       paramsArray(1,j) = str2double(paramsStr{i}{j});
    end
    
    %calculate rates -- -- 

    %calculate final rate for this reaction
    Rate(i) = statFac(i)*exp(paramsArray(1,1)+(paramsArray(1,2)/T9)+(paramsArray(1,3)/(T9^(1/3)))+paramsArray(1,4)*(T9^(1/3))+paramsArray(1,5)*T9+paramsArray(1,6)*(T9^(5/3))+paramsArray(1,7)*log(T9));
    sprintf('Reaction(%d) Rate(%e)', i, Rate(i));
    %Set up reaction groups -- -- 
   
%     if(memberID(i) == 0)
%         %then, start new reaction group
%         %!!! reaction group IDs start at 1, not 0.
%         rgID = rgID + 1;
%         RGparent(rgID) = i;
%         tempRGclassbyRG(rgID) = reacClass(i);
%     end
end
kr = zeros(1,numRG);
kf = zeros(1,numRG);
for i = 1:numReactions
 
    %logic to parse total reaction rates, check which reactions are forward
    %and reverse
    for j = 1:numRG
        if RGbyReac(i) == j
            sprintf('then reaction %d, %s, is in RG %d,', i, reacHeader{i}{1}, j);
            %sort its rate into forward or
            %reverse for this RG.
            if isReverse(i) == 0
                
                %Add its rate to kf(j)
                kf(j) = kf(j) + Rate(i);
                sprintf('reac(%d) is a forward reaction. Add Rate(%e) to kf(%d): %e', i, Rate(i), j, kf(j))
            else
               
                %this is a reverse reaction. Add its rate to kr(j)
                kr(j) = kr(j) + Rate(i);
                sprintf('reac(%d) is a reverse reaction. Add Rate(%e) to kr(%d): %e,', i, Rate(i), j, kr(j))
            end
        end
    end
end

RGclassbyRG = zeros(1,numRG);

for i = 1:numRG
    RGclassbyRG(i) = tempRGclassbyRG(i); 
   
    sprintf('(%d) kf: %e, kr: %e\n',i,kf(i),kr(i))

end

clear tempRGclassbyRG;


%Number of Reactions that contribute to increasing/decreasing this species
numFplus = zeros(1,numSpecies);
numFminus = zeros(1,numSpecies);
totalFplus = 1;
totalFminus = 1;
totalFplusRG = 0;
totalFminusRG = 0;
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
tempFplusFac = zeros(1,numSpecies*(numReactions/2));
tempFminusFac = zeros(1,numSpecies*(numReactions/2));
tempLrowFplus = zeros(1, numSpecies*(numReactions/2));
tempLcolFplus = zeros(1, numSpecies*(numReactions/2));
tempLrowFminus = zeros(1, numSpecies*(numReactions/2));
tempLcolFminus = zeros(1, numSpecies*(numReactions/2));

%TODO Move some of the next part into the integration while loop

for i = 1:numSpecies
   startingFplusRG = totalFplusRG + 1;
   startingFminusRG = totalFminusRG + 1;
   for j = 1:numReactions
       isReactant = 0;
       isProduct = 0;
       noReactantRepeat = 0;
       noProductRepeat = 0;
       for n = 1:numReactants(j)
           %check to see if this species appears as a reactant. If so,
           %count how many times it appears. ie for triple-?, he4 appears 3
           %times, so tempFminusFac(1) should be 3. 
          if (i == reactantsMatrix(j,n) && totalFminus<numel(tempFminusFac))
             %then this (j) is a reaction that depletes this isotope.
             isReactant = 1;
             tempFminusFac(totalFminus) = tempFminusFac(totalFminus) + 1;
          end
       end
       %If we've established that i is a reactant in j, add to the total
       %number of reactions that deplete every isotope (totalFminus), and
       %the total number of reactions that deplete this isotope
       %(numFminus(i)). tempFminusReacs(totalFminus) is the growing list of
       %reaction ids that we're mapping to the species. 
       if isReactant == 1 && noReactantRepeat == 0
             noReactantRepeat = 1;
             %added noReactantRepeat above to avoid overadding adding to 
             %totalFminus, etc if this species appears more than once as 
             %a reactant.
             %keeping track of how many reactions deplete this isotope
             numFminus(i) = numFminus(i) + 1;
               
             %Set ID of reaction that is responsible for increasing this
             %isotope
             tempFminusReacs(totalFminus) = j;
             %Also, This reaction's flux should be placed in L(i,i)
             tempLrowFminus(totalFminus) = i;
             tempLcolFminus(totalFminus) = i;
             %starting point for next FminusReacs entry
             totalFminus = totalFminus + 1;
       end
       for n = 1:numProducts(j)
          %follows similar logic as previous loops/if statements:
          if (i == productsMatrix(j,n) && totalFplus<numel(tempFplusFac))
             %then this (j) is a reaction that increases this isotope. 
             isProduct = 1;
             tempFplusFac(totalFplus) = tempFplusFac(totalFplus) + 1;
          end
       end
      if isProduct == 1 && noProductRepeat == 0
         noProductRepeat = 1;
         %added noProductRepeat above to avoid overadding adding to 
         %totalFplus, etc if this species appears more than once as 
         %a product.
         %keeping track of how many reactions increase this isotope
         numFplus(i) = numFplus(i) + 1;
         %Set ID of reaction that is responsible for increasing this
         %isotope
         tempFplusReacs(totalFplus) = j;
         %Also, this reaction's Flux should be placed in
         %L(i,first-reactantID)
         tempLrowFplus(totalFplus) = i;
         tempLcolFplus(totalFplus) = reactantsMatrix(j,1);
         %starting point for next FplusReacs entry
         totalFplus = totalFplus + 1;

      end
   end
end

%resize FplusReacs and FminusReacs to save dat memory, tho.

FplusReacs = zeros(1, totalFplus);
FminusReacs = zeros(1, totalFminus);
FplusFac = zeros(1, totalFplus);
FminusFac = zeros(1, totalFminus);
LrowFplus = zeros(1, totalFplus);
LcolFplus = zeros(1, totalFplus);
LrowFminus = zeros(1, totalFminus);
LcolFminus = zeros(1, totalFminus);

for i = 1:totalFplus
   FplusReacs(i) = tempFplusReacs(i);
   FplusFac(i) = tempFplusFac(i);
   LrowFplus(i) = tempLrowFplus(i);
   LcolFplus(i) = tempLcolFplus(i);
end

for i = 1:totalFminus
   FminusReacs(i) = tempFminusReacs(i);
   FminusFac(i) = tempFminusFac(i);
   LrowFminus(i) = tempLrowFminus(i);
   LcolFminus(i) = tempLrowFminus(i);
end

clear tempFplusReacs tempFminusReacs tempFplusFac tempFminusFac

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


setNextOut = 0;
alldtcount = 1;
alldt = [];



%%%%%% Variables for Partial Equilibrium %%%%%%%

y_a=0;
y_b=0;
y_c=0;
y_d=0;
y_e=0;
c1=0;
c2=0;
c3=0;
c4=0;
a=0;
b=0;
c=0;
alpha=0;
beta=0;
gamma=0;
q=0;
y_eq_a=0;
y_eq_b=0;
y_eq_c=0;
y_eq_d=0;
y_eq_e=0;
PE_val_a=0;
PE_val_b=0;
PE_val_c=0;
PE_val_d=0;
PE_val_e=0;
members=0;
pEquilbyRG = zeros(1,numRG);
pEquilFac = zeros(1,numReactions);
tolerance = 0.001;
dydt = zeros(numSpecies,1);
ydotLast = zeros(numSpecies,1);
ydoubledot = zeros(numSpecies,1);
ydoubledotLast = zeros(numSpecies,1);

tempyplot = zeros(numSpecies, 100000);
while t < tmax
    FminusSum = zeros(1,numSpecies);
    FplusSum = zeros(1,numSpecies);
    L = zeros(numSpecies);
    Flux = zeros(1,numReactions);
    %preserve last dt to make buffer so dt doesn't change so drastically:
    dtLast = dt;
    
    %%%%%%%       PARTIAL EQUILIBRIUM        %%%%%%%
    %Copied directly from FERN, from my work last summer
    %TODO Put this into function
    
    %final partial equilibrium loop for calculating equilibrium
    pEquilbyRG;
    for i = 1:numRG
        pEquilbyRG(i) = 0;
        %reset RG reactant and product populations
        y_a = 0;
        y_b = 0;
        y_c = 0;
        y_d = 0;
        y_e = 0;
        %Get current population for each reactant and product of this RG
        if RGclassbyRG(i) == 1 
          y_a = y(reactantsMatrix(RGparent(i),1));
          y_b = y(productsMatrix(RGparent(i),1));
          %set specific constraints and coefficients for RGclass 1
          c1 = y_a+y_b;
          a = 0;
          b = -kf(i);
          c = kr(i);
          q = 0;

          %theoretical equilibrium population of given species
          y_eq_a = -c/b;
          y_eq_b = c1-y_eq_a;

          %is each reactant and product in equilibrium?
          PE_val_a = abs(y_a-y_eq_a)/abs(y_eq_a);
          PE_val_b = abs(y_b-y_eq_b)/abs(y_eq_b);
          if (PE_val_a < tolerance && PE_val_b < tolerance) 
            pEquilbyRG(i)  = 1;
          end
        elseif RGclassbyRG(i) == 2
            if (reactantsMatrix(RGparent(i),2) > 0)
                y_a = y(reactantsMatrix(RGparent(i),1));
                y_b = y(reactantsMatrix(RGparent(i),2));
                y_c = y(productsMatrix(RGparent(i),1));
            else
                y_a = y(productsMatrix(RGparent(i),1));
                y_b = y(productsMatrix(RGparent(i),2));
                y_c = y(reactantsMatrix(RGparent(i),1));
            end
          c1 = y_b-y_a;
          c2 = y_b+y_c;
          a = -kf(i);
          b = -(c1*kf(i)+kr(i));
          c = kr(i)*(c2-c1);
          q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));
          y_eq_b = y_eq_a+c1;
          y_eq_c = c2-y_eq_b;
          PE_val_a = abs(y_a-y_eq_a)/abs(y_eq_a);
          PE_val_b = abs(y_b-y_eq_b)/abs(y_eq_b);
          PE_val_c = abs(y_c-y_eq_c)/abs(y_eq_c);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance) 
            pEquilbyRG(i) = 1;
          end
        elseif RGclassbyRG(i) == 3
          y_a = y(reactantsMatrix(RGparent(i),1));
          y_b = y(reactantsMatrix(RGparent(i),2));
          y_c = y(reactantsMatrix(RGparent(i),3));
          y_d = y(productsMatrix(RGparent(i),1));
          c1 = y_a-y_b;
          c2 = y_a-y_c;
          c3 = ((y_a+y_b+y_c)/3)+y_d;
          a = kf(i)*(c1+c2)-kf(i)*y_a;
          b = -((kf(i)*c1*c2)+kr(i));
          c = kr(i)*(c3+(c1/3)+(c2/3));
          q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));
          y_eq_b = y_eq_a-c1;
          y_eq_c = y_eq_a-c2;
          y_eq_d = c3-y_eq_a+((1/3)*(c1+c2));
          PE_val_a = abs(y_a-y_eq_a)/abs(y_eq_a);
          PE_val_b = abs(y_b-y_eq_b)/abs(y_eq_b);
          PE_val_c = abs(y_c-y_eq_c)/abs(y_eq_c);
          PE_val_d = abs(y_d-y_eq_d)/abs(y_eq_d);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) 
            pEquilbyRG(i) = 1;
          end
        elseif RGclassbyRG(i) == 4
          y_a = y(reactantsMatrix(RGparent(i),1));
          y_b = y(reactantsMatrix(RGparent(i),2));
          y_c = y(productsMatrix(RGparent(i),1));
          y_d = y(productsMatrix(RGparent(i),2));
          c1 = y_a-y_b;
          c2 = y_a+y_c;
          c3 = y_a+y_d;
          a = kr(i)-kf(i);
          b = -(kr(i)*(c2+c3))+(kf(i)*c1);
          c = kr(i)*c2*c3;
          q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));
          y_eq_b = y_eq_a-c1;
          y_eq_c = c2-y_eq_a;
          y_eq_d = c3-y_eq_a;
          PE_val_a = abs(y_a-y_eq_a)/abs(y_eq_a);
          PE_val_b = abs(y_b-y_eq_b)/abs(y_eq_b);
          PE_val_c = abs(y_c-y_eq_c)/abs(y_eq_c);
          PE_val_d = abs(y_d-y_eq_d)/abs(y_eq_d);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) 
            pEquilbyRG(i) = 1;
          end
        elseif RGclassbyRG(i) == 5
          if(productsMatrix(RGparent(i), 3) > 0)
            y_a = y(reactantsMatrix(RGparent(i),1));
            y_b = y(reactantsMatrix(RGparent(i),2));
            y_c = y(productsMatrix(RGparent(i),1));
            y_d = y(productsMatrix(RGparent(i),2));
            y_e = y(productsMatrix(RGparent(i),3));
          else
            y_a = y(productsMatrix(RGparent(i),1));
            y_b = y(productsMatrix(RGparent(i),2));
            y_c = y(reactantsMatrix(RGparent(i),1));
            y_d = y(reactantsMatrix(RGparent(i),2));
            y_e = y(reactantsMatrix(RGparent(i),3));
          end
%           i
%           RGclassbyRG(i)
%           RGparent(i)
%           productsMatrix(RGparent(i),3)
%           %RGbyReac(40)
%           reacHeader{RGparent(i)}{1}
%          
          c1 = y_a+((y_c+y_d+y_e)/3);
          c2 = y_a-y_b;
          c3 = y_c-y_d;
          c4 = y_c-y_e;
          a = (((3*c1)-y_a)*kr(i))-kf(i);
          alpha = c1+((c3+c4)/3);
          beta = c1-(2*c3/3)+(c4/3);
          gamma = c1+(c3/3)-(2*c4/3);
          b = (c2*kf(i))-(((alpha*beta)+(alpha*gamma)+(beta*gamma))*kr(i));
          c = kr(i)*alpha*beta*gamma;
          q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));
          y_eq_b = y_eq_a-c2;
          y_eq_c = alpha-y_eq_a;
          y_eq_d = beta-y_eq_a;
          y_eq_e = gamma-y_eq_a;
          PE_val_a = abs(y_a-y_eq_a)/abs(y_eq_a);
          PE_val_b = abs(y_b-y_eq_b)/abs(y_eq_b);
          PE_val_c = abs(y_c-y_eq_c)/abs(y_eq_c);
          PE_val_d = abs(y_d-y_eq_d)/abs(y_eq_d);
          PE_val_e = abs(y_e-y_eq_e)/abs(y_eq_e);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance && PE_val_e < tolerance) 
            pEquilbyRG(i) = 1;
          end
        end
    end
    %%%%%%%        END PARTIAL EQUILIBRIUM      %%%%%%%
    
    
        %Loop through all reactions and set up Flux arrays for this timestep
    for i = 1:numReactions
        
        %%% set up partial equilibrium factor to remove flux elements from
        %%% L matrix if a reaction is in partial equilibrium.
        if pEquilOn == 1
            if pEquilbyRG(RGbyReac(i)) == 1
                pEquilFac(i) = 0;
            elseif pEquilbyRG(RGbyReac(i)) == 0
                pEquilFac(i) = 1;
            end
        end
        
        %Give Flux a value so its product with y won't be zero. 
        Flux(i) = Rate(i);
        %go through all reactants that allow this reaction to occur
%          sprintf('reaction(%d) has these reactants:', i)
        for j = 1:numReactants(i)
            Flux(i) = Flux(i)*y(reactantsMatrix(i,j));
%              sprintf('reactant(%d) has y(%d): %e', j, reactantsMatrix(i,j), y(reactantsMatrix(i,j)))
        end
%          sprintf('Yielding Flux(%d): %e', i, Flux(i))
        %check if Fluxes we just calculated are same as what's in L matrix:
%         if mod(i,2) == 0
%             %This is the second reaction in a reaction group. i is even
%             %TotalFlux holds all total Fplus and total Fminus for each RG
%             %For example, since reacs 1 and 2 are forward triple-?, their
%             %total Fplus is in TotalFlux(1). Similarly, since reacs 3 and 4
%             %are reverse triple-?, their total Fminus is TotalFlux(2). I'll
%             %parse into Fplus/Fminus by RG next, and compare with what I 
%             %have in the L-matrix.
%             TotalFlux(i/2) = Flux(i) + Flux(i-1);
%         end
    end
    
%     for i = 1:(numReactions/2)
%        if mod(i,2) == 0
%           %i is even. This is the second item for a specific reaction in 
%           %TotalFlux, which means it is the total flux for a reverse 
%           %reaction, Fminus(RGid). 
%           RGrFlux(i/2) = TotalFlux(i)
%        else
%            %i is odd. This is the first item for a specific reaction in
%            %Total Flux, which means it is the total flux for a forward
%            %reaction, Fplus(RGid).
%            RGfFlux((i/2)+.5) = TotalFlux(i);
%        end
%     end
    
%     sprintf('RGfFlux(1) should be: %e\nRGfFlux(2) should be: %e\nRGrFlux(1) should be: %e\nRGrFlux(2) should be: %e\n', kf(1)*y(1)*y(1)*y(1), kf(2)*y(1)*y(2), kr(1)*y(2), kr(2)*y(3)) 
    
%     TotalFlux
%     RGfFlux
%     sprintf('Fminus should be \nfor RG1: %e \nRG2: %e', (kr(1)*y(2)), (kr(2)*y(3)))
%     sprintf('Fplus should be \nfor RG1: %e \nRG2: %e', (kf(1)*y(1)*y(1)*y(1)), (kf(2)*y(1)*y(2)))
%     RGrFlux
    
    %Loop through species and set up FluxSum arrays
    %A counter to keep track of the starting index in FplusReacs array that
    %indicates set of reactions responsible for increasing this species(i)
    speciesFplusIndex = 1;
    speciesFminusIndex = 1;
    for i = 1:numSpecies
        %populate FplusSum and Fplus Lrow for this species -- --
        startingFplusIndex = speciesFplusIndex;
        %subtract 1 to keep from spilling over into next species' reactions
        endingFplusIndex = speciesFplusIndex+numFplus(i)-1;
        for j = startingFplusIndex:endingFplusIndex
            %Add to FplusSum for this species (i) with Flux(j) from this 
            %reaction (j), AND, build Fplus portion of Lrow for this species.
            
            %These reactions, FplusReacs(j), increase species (i).
            
            FplusSum(i) = FplusSum(i) + FplusFac(j)*Flux(FplusReacs(j));
            
            %Build Fplus portion of Lrow for species i, 
            %all those NOT L(i,i)-- --
            %Row is d(y_i)/dt, column 1 is flux/y_a, column
            %2 is flux/y_b, etc. where flux is the flux contributes to
            %fplus for (i), with the abundance for this column's species
            %divided out so it can be multiplied by the vector y later...
            %Remember, L is used just for its eigenvalues. The reactants for 
            %Fplus are never species (i). so, L_1,1 will be built in the
            %Fminus for loop coming up.

            % of all of the reactants in this reaction(FplusReacs(j)) 
            % increasing isotope (i), Let's just take the first one,
            % reactantsMatrix(FplusReacs(j),1),
            % divide out its abundance from the flux of this reaction
            % FplusReacs(j), and add it to L(i,
            % reactantsMatrix(FplusReacs(j),1)). We choose the first one,
            % because the choice for which reactant to divide out is
            % arbitrary. (Or so we think). The eigenvalues should be
            % invariant under transformations, right? 
            
            %so, according to the above^^^, I don't need to do this:
% % %             for m = 1:numReactants(FplusReacs(j))
% % % %              'the reaction that increases species (i)'
% % % %              FplusReacs(j)
% % % %              'the reactantID within this reaction (not species ID)'
% % % %              m
% % % 
% % %                %this speices(m) is a reactant in the reaction
% % %                %FplusReacs(j), and contributes to increasing species(i).
% % %                %Add its flux/y(m) to L(i,m).

               %don't bother changing L(i,reactantsMatrix(FplusReacs(j),1)) 
               %if the y vector component,
               %species(reactantsMatrix(FplusReacs(j),1)),
               %has zero abundance. It's contribution 
               %will automatically be zero, as the Flux(FplusReacs(j)) 
               %should be zero. (otherwise ends in a NaN in L)
             if(y(reactantsMatrix(FplusReacs(j),1)) ~= 0)
                 %sprintf('Inside Fplus Building L:\nj: %d\nSpecies: %d\nReaction increasing species: %d\nFplusFac: %d\nFlux from Reaction: %e\nFirst Reactant of Reaction: %d\nAbundance of reactant 1:%e\n', j, i, FplusReacs(j), FplusFac(j), Flux(FplusReacs(j)), reactantsMatrix(FplusReacs(j),1), y(reactantsMatrix(FplusReacs(j),1)))
                 if (pEquilOn == 1)
                    L(i,reactantsMatrix(FplusReacs(j),1)) = L(i,reactantsMatrix(FplusReacs(j),1)) + pEquilFac(FplusReacs(j))*FplusFac(j)*(Flux(FplusReacs(j))/y(reactantsMatrix(FplusReacs(j),1)));
                 else
                    L(i,reactantsMatrix(FplusReacs(j),1)) = L(i,reactantsMatrix(FplusReacs(j),1)) + FplusFac(j)*(Flux(FplusReacs(j))/y(reactantsMatrix(FplusReacs(j),1)));
                 end
             end
        end
            %end Bulid Fplus parts of L Matrix, all those not L(i,i)


        %populate FminusSum and Fminus Lrow for this species -- --
        startingFminusIndex = speciesFminusIndex;
        %subtract 1 to keep from spilling over into next species' reactions
        endingFminusIndex = speciesFminusIndex+numFminus(i)-1;

        for j = startingFminusIndex:endingFminusIndex

            %Add to FminusSum for this species(i) with Flux(FminusReacs(j)) from this 
            %reaction(FminusReacs(j))
            %These reactions, FminusReacs(j), decrease species(i).
            FminusSum(i) = FminusSum(i) + FminusFac(j)*Flux(FminusReacs(j));
            
%             sprintf('Species: %d\nFminusIndex: %d\nReaction: %d\nFminusFac: %d\nFlux from Reac: %e\nFminusSum: %e\n', i, j, FminusReacs(j), FminusFac(j), Flux(FminusReacs(j)), FminusSum(i))
%             sprintf('FminusSum(1) should be: %e\n', -(((3*(Flux(1)+Flux(2)))+Flux(5)+Flux(6))))
            %All Fminus reactions have species i as a reactant.
            %Here we build L(i,i).
            %this speices(i) is a reactant in this 
            %reaction(FminusReacs(j)) that contributes to depleting species
            %(i). Add its flux/y(i) to L(i,i).
            
            %don't bother changing L(i,i) if the y vector
            %component, species(i), has zero abundance. It will
            %automatically be zero, as the Flux(FminusReacs(j)) should be
            %zero. (otherwise ends in a NaN in L)
            if(y(i) ~= 0)
                %sprintf('Inside Fminus Building L:\nSpecies: %d\nReaction: %d\nFminusFac: %d\nFlux from Reaction: %e\nAbundance of species:%e\n', i, FminusReacs(j), FminusFac(j), Flux(FminusReacs(j)), y(i))
                if(pEquilOn == 1)
                    %multiply by a factor corresponding to the 
                    %partialEquilibrium state for this reaction. 
                    %If a reaction is in partial equilibrium, its 
                    %flux will be multiplied by 0 to remove it from L.
                    L(i,i) = L(i,i) - pEquilFac(FminusReacs(j))*FminusFac(j)*(Flux(FminusReacs(j))/y(i));
                else
                    L(i,i) = L(i,i) - FminusFac(j)*(Flux(FminusReacs(j))/y(i));
                end
            end
            %end Bulid Fminus parts of L Matrix, all those INDEED L(i,i)
        end

        %Go to next species' reactions
        speciesFplusIndex = speciesFplusIndex + numFplus(i);
        speciesFminusIndex = speciesFminusIndex + numFminus(i);
    end
    
    %'Programmed L'
    L;
    %FminusSum
    %sprintf('L1,1 PROGRAMMED: %e \n', L(1,1))
    
    %This reflects identically what is in the manual L. This is what we aim
    %for, programatically. 
    %sprintf('FULL RATES*ABUNDANCES:\nL(1,1) should be: %e\nL(1,2) should be: %e\nL(1,3) should be: %e\n', -3*Rate(1)*y(1)*y(1)-3*Rate(2)*y(1)*y(1)-Rate(5)*y(2)-Rate(6)*y(2), 3*Rate(3)+3*Rate(4), Rate(7)+Rate(8))
    %This will confirm if Fluxes are being built correctly. So far they
    %are. 
    
    %sprintf('Checking Flux against manual calculation:\nFlux(1): %e, Should be %e\nFlux(2):%e, Should be %e\nFlux(3): %e, Should be %e\nFlux(4): %e, Should be %e\nFlux(5): %e, Should be %e\nFlux(6): %e, Should be %e\nFlux(7): %e, Should be %e\nFlux(8): %e, Should be %e\n', Flux(1), Rate(1)*y(1)*y(1)*y(1), Flux(2), Rate(2)*y(1)*y(1)*y(1), Flux(3), Rate(3)*y(2), Flux(4), Rate(4)*y(2), Flux(5), Rate(5)*y(1)*y(2), Flux(6), Rate(6)*y(1)*y(2), Flux(7), Rate(7)*y(3), Flux(8), Rate(8)*y(3))
    %sprintf('Checking Rates against kf/kr:\nkf(1): %e Should be: %e\nkr(1): %e Should be: %e\nkf(2): %e Should be: %e\nkr(2): %e Should be: %e\n', kf(1), Rate(1)+Rate(2), kr(1), Rate(3)+Rate(4), kf(2), Rate(5)+Rate(6), kr(2), Rate(7)+Rate(8))
    %sprintf('WITH FLUXES:\nL(1,1) should be: %e\nL(1,2) should be: %e\nL(1,3) should be: %e\nL(2,1) should be: %e\nL(2,2) should be: %e\nL(2,3) should be: %e\nL(3,1) should be: %e\nL(3,2) should be: %e\nL(3,3) should be: %e\n', -(((3*(Flux(1)+Flux(2)))+Flux(5)+Flux(6))/y(1)), 3*(Flux(3)+Flux(4)/y(2)), ((Flux(7)+Flux(8))/y(3)), (Flux(1)+Flux(2)/y(1)), -((Flux(3)+Flux(4)+Flux(5)+Flux(6))/y(2)), (Flux(7)+Flux(8))/y(3), (Flux(5)+Flux(6))/y(1), 0, -(Flux(7)+Flux(8))/y(3))
    %sprintf('L(1,1) is: %e\n', -3*kf(1)*y(1)*y(1)-kf(2)*y(2))

    
    %OLD Manually built L matrix FOR 3-isotope network-- --
    %TODO: When I move up to 16 species alpha network, I'll need to try
    %making the 16x16 manual matrix to make sure all is well here... Well I
    %might not need to that because that would take forever, first of all,
    %and second of all, as long as the results are correct from the 
    %programmed L, it's all good!
    %populate flux matrix... it's a linearized matrix... so really "d/dt"
%     
%     L=[-3*kf(1)*y(1)*y(1)-kf(2)*y(2), 3*kr(1), kr(2);
%         kf(1)*y(1)*y(1), -kr(1)-kf(2)*y(1), kr(2);
%         kf(2)*y(2), 0, -kr(2)];
%     'Manual L'
%     L
   %sprintf('L1,1 MANUAL: %e \n', L(1,1))
   
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

%     'largest eigenvalue'
%     lambda
    
    %'delta t'
    %changed to 2/lambda because that is the explicit theoretical maximum.
    dt = abs(1/lambda);
    if dt > .01*t
        dt = .01*t;
    end
%     if dt > .1*t && t < 1e-13
%         dt = .1*t;
%     elseif dt > .01*t && t > 1e-13
%         dt = .01*t;
%     end
    
    
    %%%%%%%     ASYMPTOTIC APPROXIMATION      %%%%%%%
    
    %checkAsy with current dt, and adjust L for column and row for species 
    %that satisfy, and update dt by finding the eigen values again:
    AsySatisfied = 0;
    AsySpecies = zeros(1,numSpecies);
    R = L;
    for i = 1:numSpecies
        %if ((FplusSum(i)/FminusSum(i)) > .999999 && (FplusSum(i)/FminusSum(i)) < 1.000001 && checkAsy == 1)
        if ((FplusSum(i)/FminusSum(i) > 0.99) && (FplusSum(i)/FminusSum(i) < 1.01) && checkAsy == 1) %This seems to work better!
%             if(FminusSum(i) * dt / y(i) > 1.0) //From FERN
            %If any isotopes pass the above, switch on that Asymptotic was
            %satisfied at least once. Thus we'll proceed to the next if
            %statement.
            AsySatisfied = 1;
            AsySpecies(i) = 1;
            %do Asymptotic Update
            %This is exactly what's in FERN now... Change it such that
            %we're modifying the L-matrix... removing corresponding row an
            %column, corresponding to species (i).
            %y(i) = (y(i)+(FplusSum(i)*dt))/(1+FminusSum(i)*dt);
            
            %%%%UPDATE%%%%
            %I've introduced an R matrix, which is the altered L matrix,
            %since I need the original L matrix to properly calculate the
            %dydt vector. NOTE: I *can* indeed remove L(i,j) if Asy is
            %satisfied, as dydt(i) for species(i) should be zero anyway.
            
            for j = 1:numSpecies
                %L(i,j) = 0;
                R(i,j) = 0;
                R(j,i) = 0;
            end
        end
    end
    
    if (AsySatisfied == 1 && checkAsy == 1)
       %find the new eigenvalues of modified L, and recalculate dt 
       %'eigenvalues'
        eigenR = eig(R);

        %get largest eigenvalue
        lambda = 0;
        absEigR = abs(eigenR);
        for i = 1:numSpecies
            if gt(absEigR(i),abs(lambda))
                lambda = eigenR(i);
            end
        end

        %'largest eigenvalue'
        lambda;
        %'delta t'
        dt = abs(1/lambda);
            
        if dt > .01*t
            dt = .01*t;
        end
% 
%         if dt > .1*t && t < 1e-13
%             dt = .1*t;
%         elseif dt > .01*t && t > 1e-13
%             dt = .01*t;
%         end
    end
    
    alldt(alldtcount) = dt;
    alldtcount = alldtcount + 1;
   
    %'dydt'
    %calculate abundance rate. populates dydt vector, which contains each 
    %the change in abundances for each species. Instead of doing 
    %Y + (FplusSum - FminusSum) * dt; 
    %as in FERN (line 656 in kernels), dydt is the final change.
    %sprintf('y: %e %e %e\n', y)
    %sprintf('compareFplusSum-FminusSum: %e %e %e\n', FplusSum-FminusSum)
    %sprintf('FplusSum for He4 Should be: %e\nand FminusSum: %e\n', 3*kr(1)*y(2)+kr(2)*y(3), -3*kf(1)*y(1)*y(1)*y(1)-kf(2)*y(1)*y(2))
    
    %Check the jerk of the abundance curve... If it's larger than some
    %tolerance, reduce dt.
    
    ydotLast = dydt;
    dydt = L*transpose(y);
    
    ydoubledotLast = ydoubledot;
    ydoubledot = (dydt - ydotLast)/dt;
    
    ytripledot = (ydoubledot - ydoubledotLast)/dt;
%     check1 = 0;
    check2 = 0;
    maxytripledot = 0;
    for i = 1:numel(ytripledot)
        if i == 1
           maxytripledot = ytripledot(i);
        elseif ytripledot(i)>maxytripledot
            maxytripledot = ytripledot(i);
        end
        
%         if abs(ytripledot(i)) > 1 && abs(ytripledot(i)) < 1e5 && t > 1e-16
%             sprintf('t: %e, y(%d): %e\n', t, i, ytripledot(i))
%             check1 = 1;
%         end

    end
    
    if maxytripledot > 1e30 && t > 1e-16
        %sprintf('t: %e, y(%d): %e\n', t, i, ytripledot(i))
        check2 = 1;
    end
    
%     if check1 == 1
%        dt = .9*dt; 
%     end
% 
%     if check2 == 1
%         dt = .9*dt;
%        %dt = 1/(log10(maxytripledot))*dt 
%     end
    
    %pushes timestep, losing a little accuracy
%     if (abs(dt-dtLast)/dtLast) > 10     
%         if dt > dtLast
%            dt = dtLast/0.01; 
%         elseif dt < dtLast
%            dt = dtLast;
%         end
%     end

%   constrains change in timestep to specified amount, preserves accuracy
% 
%     if (abs(dt-dtLast)/dtLast) > 100
%         if dt > dtLast
%            dt = dtLast/0.01;
%         elseif dt < dtLast
%            dt = dtLast/100;
%         end
%     end
%     
    

    
%     if ((dt-dtLast)/dtLast) > .05
%            dt = dtLast/0.95; 
%     end
%     
%         %ensures that dt doesn't change by more than 5% between timesteps
%     if (dt-dtLast/dtLast) < (-0.05)
%        dt = dtLast/1.05;
%     end

    R;
    L;
    pEquilbyRG;
    dt;
    dtLast;
    %for output. list RGs in eq
    numeq = 1;
    ineq = [];
    numAsy = 0;
    for a = 1:numRG
        if pEquilbyRG(a) == 1
            ineq(numeq) = a;
            numeq = numeq + 1;
        end
    end
    if numel(ineq) > 0
        sprintf('%d ', ineq);
    end
    
    %check number species asymptotic
    for a = 1:numSpecies
        if AsySpecies(a) == 1
            numAsy = numAsy + 1;
        end
    end

    %sprintf('dydt: %e %e %e\n', dydt)
    %!!! TURN THIS INTO FUNCTION !!!
    %Update populations
    for i = 1:numSpecies
        if AsySpecies(i) == 1
            y(i) = (y(i) + FplusSum(i) * dt) / (1.0 + FminusSum(i) * dt / y(i));
        else
            %Update abundances, Euler Update
            %This is what's in FERN:
            %y(i)+((FplusSum(i)-FminusSum(i))*dt)
            %This is what we'll use by using the L-matrix, which is
            %essentially what is in FERN. Just that FplusSum and FminusSum
            %are implicit in the components of L.  
            dydt;
            dt;
            y(i) = y(i)+(dt*(dydt(i)));
        end
        tempyplot(i,numTimesteps) = y(i);
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
                    sprintf('Output: %d/100\ntime: %e, dt: %e\n', OutCount, t, dt)
                    sprintf('abundances: \n')
                    sprintf('%e\n', y)
                    sprintf('kf: %e \n', kf)
                    sprintf('kr: %e \n', kr)
                    sprintf('Flux:')
                    sprintf('%e\n', Flux)
                    sprintf('FplusSum, FminusSum')
                    sprintf('%e, %e\n',FplusSum, FminusSum)
                    sprintf('FracRGPE: %d/%d\nFracAsy: %d/%d, dt: %e\n, t: %e\nOutput: %d/100\n',numeq-1,numRG,numAsy,numSpecies, dt, t, OutCount)

                   %print time
                    nextOutput = nextOutput + intervalLogt;
                end 
            end
            
%             if numAsy > 0
%                 
%                 'GOT SOME ASSY'
%                 
%                     sprintf('Output: %d/100\ntime: %e, dt: %e\n', OutCount, t, dt)
%                     sprintf('abundances: \n')
%                     sprintf('%e\n', y)
%                     sprintf('kf: %e %e %e\nkr: %e %e %e \n', kf, kr)
%                     sprintf('Flux:')
%                     sprintf('%e\n', Flux)
%                     sprintf('FplusSum, FminusSum')
%                     sprintf('%e, %e\n',FplusSum, FminusSum)
%                     sprintf('FracRGPE: %d/%d\nFracAsy: %d/%d',numeq-1,numRG,numAsy,numSpecies)
%                 
% %                 return
%             end
    

    %plot time & timestepping
    tplot(numTimesteps) = t;
    dtplot(numTimesteps) = dt;
    
    t=t+dt;
    dt;
    %number of timesteps
    numTimesteps = numTimesteps + 1;
end
min_y = 0;
yplot = zeros(numSpecies, numTimesteps-1);
javaYPlot = zeros(numSpecies, length(javaPlotTimes{1}));
javaTPlot = zeros(1, length(javaPlotTimes{1}));
for i = 1:numSpecies
    counter = 1;
    for j = 1:numTimesteps-1
        if(plotJavaCompare == 1 && ((tplot(j)/javaPlotTimes{1}(counter)) < 1.01 && (tplot(j)/javaPlotTimes{1}(counter)) > .99) && counter < length(javaPlotTimes{1}))
            tplot(j);
            javaPlotTimes{1}(counter);
            javaYPlot(i,counter) = tempyplot(i,j);
            javaYPlot(i,counter);
            tempyplot(i,j);
            javaTPlot(counter) = javaPlotTimes{1}(counter);
            counter = counter + 1;
        else
            yplot(i,j) = tempyplot(i,j);
        end
        if yplot(i,j) < min_y %get minimum y for lowest limit on plot
           min_y = yplot(i,j); 
        end
    end
end

clear tempyplot

sprintf('%e\n', alldt);
%plot abundances
% N = 1:numSpecies+1;
if(plotJavaCompare == 1)
    %sprintf('Len: javaTplot: %d, javaYPlot: %d, javaAbundances: %d',length(javaTPlot), length(javaYPlot), length(javaAbundances))
    lh = loglog(javaTPlot,javaYPlot,'-',javaTPlot,javaAbundances,'--');
else
    lh = loglog(tplot,yplot,tplot,dtplot);
    set(lh(numSpecies+1),'LineWidth',2);
end
%make t v. dt line fat'
lh
%legendCell = cellstr(num2str(N', '%-d'));
%legend(legendCell,'Location','NorthEastOutside')
% if numel(y) == 3
%     loglog(tplot,yplot1,tplot, yplot2,tplot, yplot3, tplot, dtplot)
%     legend('He', 'C', 'O', 't')
% elseif numel(y) >= 16
%     loglog(tplot, yplot1, tplot, yplot2, tplot, yplot3, tplot, yplot4, tplot, yplot5, tplot, yplot6, tplot, yplot7, tplot, yplot8, tplot, yplot9, tplot, yplot10, tplot, yplot11, tplot, yplot12, tplot, yplot13, tplot, yplot14, tplot, yplot15, tplot, yplot16, tplot, dtplot)
%     legend('He', 'C', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Zn', 'Ge', 'Se', 't')
% end

axis([1e-16,tmax,1e-15,5e-1])

%plot time vs timestep
%loglog(tplot,dtplot)

   