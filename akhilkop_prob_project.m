clear all
close all
format long
load('data.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  X = F1 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% calculating the mean for each class %%%%%%%%%%

m_f1_c1 = mean(F1(1:100,1));
m_f1_c2 = mean(F1(1:100,2));
m_f1_c3 = mean(F1(1:100,3));
m_f1_c4 = mean(F1(1:100,4));
m_f1_c5 = mean(F1(1:100,5));

%%%%% calculating the standard deviation for each class %%%%%%%%%%

std_f1_c1 = std(F1(1:100,1));
std_f1_c2 = std(F1(1:100,2));
std_f1_c3 = std(F1(1:100,3));
std_f1_c4 = std(F1(1:100,4));
std_f1_c5 = std(F1(1:100,5));

%%%%%%%%%% creating normal PDF for each class %%%%%%%%%%%%%%%%

pdf_f1_c1 = makedist('normal','mu',m_f1_c1,'sigma',std_f1_c1);
pdf_f1_c2 = makedist('normal','mu',m_f1_c2,'sigma',std_f1_c2);
pdf_f1_c3 = makedist('normal','mu',m_f1_c3,'sigma',std_f1_c3);
pdf_f1_c4 = makedist('normal','mu',m_f1_c4,'sigma',std_f1_c4);
pdf_f1_c5 = makedist('normal','mu',m_f1_c5,'sigma',std_f1_c5);

%%%%%% creating dummy class to store count of each observation falling into a class %%%%%%%%%%% 
class_f1 = [0,0,0,0,0];

%%%%% classifieng each observation into a class based on probalities %%%%%% 

for i = 1:5 
    for j = 101:1000
        [m,ind] = max([pdf(pdf_f1_c1,F1(j,i)),pdf(pdf_f1_c2,F1(j,i)),pdf(pdf_f1_c3,F1(j,i)),pdf(pdf_f1_c4,F1(j,i)),pdf(pdf_f1_c5,F1(j,i))]);
        if ind == i
            class_f1(i) = class_f1(i)+1;
        end
    end
end

%%%%%%%%% calculating the classification accuracy and error rate of f1 %%%%%%%%%%

Accuracy_F1 = ((class_f1(1)+class_f1(2)+class_f1(3)+class_f1(4)+class_f1(5))/4500)*100
Error_F1 = 100-Accuracy_F1                 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  X = F2 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% calculating the mean for each class %%%%%%%%%%
m_f2_c1 = mean(F2(1:100,1));
m_f2_c2 = mean(F2(1:100,2));
m_f2_c3 = mean(F2(1:100,3));
m_f2_c4 = mean(F2(1:100,4));
m_f2_c5 = mean(F2(1:100,5));

%%%%% calculating the standard deviation for each class %%%%%%%%%%

std_f2_c1 = std(F2(1:100,1));
std_f2_c2 = std(F2(1:100,2));
std_f2_c3 = std(F2(1:100,3));
std_f2_c4 = std(F2(1:100,4));
std_f2_c5 = std(F2(1:100,5));

%%%%%%%%%% creating normal PDF for each class %%%%%%%%%%%%%%%%

pdf_f2_c1 = makedist('normal','mu',m_f2_c1,'sigma',std_f2_c1);
pdf_f2_c2 = makedist('normal','mu',m_f2_c2,'sigma',std_f2_c2);
pdf_f2_c3 = makedist('normal','mu',m_f2_c3,'sigma',std_f2_c3);
pdf_f2_c4 = makedist('normal','mu',m_f2_c4,'sigma',std_f2_c4);
pdf_f2_c5 = makedist('normal','mu',m_f2_c5,'sigma',std_f2_c5);

%%%%%% creating dummy class to store count of each observation falling into a class %%%%%%%%%%%

class_f2 = [0,0,0,0,0];

%%%%% classifieng each observation into a class based on probalities %%%%%%

for i = 1:5
    for j = 101:1000
        [m,ind] = max([pdf(pdf_f2_c1,F2(j,i)),pdf(pdf_f2_c2,F2(j,i)),pdf(pdf_f2_c3,F2(j,i)),pdf(pdf_f2_c4,F2(j,i)),pdf(pdf_f2_c5,F2(j,i))]);
        if ind == i
            class_f2(i) = class_f2(i)+1;
        end
    end
end

%%%%%%%%% calculating the classification accuracy and error rate of f1 %%%%%%%%%%

Accuracy_F2 = ((class_f2(1)+class_f2(2)+class_f2(3)+class_f2(4)+class_f2(5))/4500)*100
Error_F2 = 100-Accuracy_F2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  X = Z1 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% normalising the observations made by each person %%%%%%%%%%%%

for i = 1:1000
    m = mean(F1(i,:));
    s = std(F1(i,:));
    for j=1:5
        Z1(i,j) = (F1(i,j)-m)/s;
    end
end

%%%%%% calculating the mean for each class %%%%%%%%%%

m_z1_c1 = mean(Z1(1:100,1));
m_z1_c2 = mean(Z1(1:100,2));
m_z1_c3 = mean(Z1(1:100,3));
m_z1_c4 = mean(Z1(1:100,4));
m_z1_c5 = mean(Z1(1:100,5));

%%%%% calculating the standard deviation for each class %%%%%%%%%%

std_z1_c1 = std(Z1(1:100,1));
std_z1_c2 = std(Z1(1:100,2));
std_z1_c3 = std(Z1(1:100,3));
std_z1_c4 = std(Z1(1:100,4));
std_z1_c5 = std(Z1(1:100,5));

%%%%%%%%%% creating normal PDF for each class %%%%%%%%%%%%%%%%

pdf_z1_c1 = makedist('normal','mu',m_z1_c1,'sigma',std_z1_c1);
pdf_z1_c2 = makedist('normal','mu',m_z1_c2,'sigma',std_z1_c2);
pdf_z1_c3 = makedist('normal','mu',m_z1_c3,'sigma',std_z1_c3);
pdf_z1_c4 = makedist('normal','mu',m_z1_c4,'sigma',std_z1_c4);
pdf_z1_c5 = makedist('normal','mu',m_z1_c5,'sigma',std_z1_c5);

%%%%%% creating dummy class to store count of each observation falling into a class %%%%%%%%%%% 

class_z1 = [0,0,0,0,0];

%%%%% classifieng each observation into a class based on probalities %%%%%%

for i = 1:5
    for j = 101:1000
        [m,ind] = max([pdf(pdf_z1_c1,Z1(j,i)),pdf(pdf_z1_c2,Z1(j,i)),pdf(pdf_z1_c3,Z1(j,i)),pdf(pdf_z1_c4,Z1(j,i)),pdf(pdf_z1_c5,Z1(j,i))]);
        if ind == i
            class_z1(i) = class_z1(i)+1;
        end
    end
end

%%%%%%%%% calculating the classification accuracy and error rate of f1 %%%%%%%%%%

Accuracy_Z1 = ((class_z1(1)+class_z1(2)+class_z1(3)+class_z1(4)+class_z1(5))/4500)*100
Error_Z1 = 100-Accuracy_Z1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% X = [Z1,F2] %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class_z1f2 = [0,0,0,0,0];

for i = 1:5
    for j = 101:1000
        [m,ind] = max([pdf(pdf_f2_c1,F2(j,i))*pdf(pdf_z1_c1,Z1(j,i)),pdf(pdf_f2_c2,F2(j,i))*pdf(pdf_z1_c2,Z1(j,i)),pdf(pdf_f2_c3,F2(j,i))*pdf(pdf_z1_c3,Z1(j,i)),pdf(pdf_f2_c4,F2(j,i))*pdf(pdf_z1_c4,Z1(j,i)),pdf(pdf_f2_c5,F2(j,i))*pdf(pdf_z1_c5,Z1(j,i))]);
        if ind == i
            class_z1f2(i) = class_z1f2(i)+1;
        end
    end
end

%%%%%%%%% calculating the classification accuracy and error rate of f1 %%%%%%%%%%

Accuracy_Z1F2 = ((class_z1f2(1)+class_z1f2(2)+class_z1f2(3)+class_z1f2(4)+class_z1f2(5))/4500)*100
Error_Z1F2 = 100-Accuracy_Z1F2

%%%%%%%% creating plot for z1 vs f2 %%%%%%%%%%%%%%%%%%

figure()
plot(F1,'o')
xlabel('Feature(F1)')
ylabel('Performance Measurement')
legend('C1','C2','C3','C4','C5','location','northeastoutside')
title('Plot for X=F1')

figure()
plot(F2,'o')
xlabel('Feature(F2)')
ylabel('Performance Measurement')
legend('C1','C2','C3','C4','C5','location','northeastoutside')
title('Plot for X=F2')

figure()
plot(Z1,'o')
xlabel('Feature(Z1)')
ylabel('Performance Measurement')
legend('C1','C2','C3','C4','C5','location','northeastoutside')
title('Plot for X=Z1')

figure()
plot(F1,F2,'o')
xlabel('1st feature(F1)')
ylabel('2nd feature(F2)')
legend('C1','C2','C3','C4','C5')
title('Plot for X=[F1 F2]')

figure()
plot(Z1,F2,'o')
xlabel('1st feature(Z1)')
ylabel('2nd feature(F2)')
legend('C1','C2','C3','C4','C5')
title('Plot for X=[Z1 F2]')





