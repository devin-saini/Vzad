clear;
allflashes = readmatrix('testing.csv');
dim = length(allflashes);
x = round(dim/40);
allFos = zeros(x,1);
allFms = zeros(x,1);
allFvFms = zeros(x,1);
for i = 1:(dim/40)
    allFos(i) = allflashes(((40*i)-31),2);
    allFms(i) = allflashes(((40*i)+5),2);
    allFvFms(i) = (allFms(i)-allFos(i))/allFms(i);
end
numTrains = round(x/50);
organizedFvFm = zeros(50,numTrains);
for j = 1:numTrains
    for l = 1:50
        organizedFvFm(l,j) = allFvFms((50*(j-1))+(l));
    end
end
writeout = table(organizedFvFm);
writetable(writeout, 'organizedflashtrains.csv');

organizedFm = zeros(50,numTrains);
for j = 1:numTrains
    for l = 1:50
        organizedFm(l,j) = allFms((50*(j-1))+(l));
    end
end
writeout2 = table(organizedFm);
writetable(writeout2, 'organizedFms.csv');