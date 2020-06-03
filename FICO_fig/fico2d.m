% [X,Y] = meshgrid(-2:0.25:2,-1:0.2:1);
% Z = X.* exp(-X.^2 - Y.^2);
% [U,V,W] = surfnorm(X,Y,Z);
% quiver3(X,Y,Z,U,V,W,0.5);
% hold on
% surf(X,Y,Z);
% colormap hsv
% view(-35,45)
% axis ([-2 2 -1 1 -.6 .6])
% hold off
% 
clc;
clear all;
close all;
fil = 'invariant/'
data0 = csvread(join([fil,'df_un_0.csv']),1,0);  % skips the first three rows of data
data1 = csvread(join([fil,'df_un_1.csv']),1,0);  % skips the first three rows of data


datae0 = csvread(join([fil,'df_eqopt_0.csv']),1,0);  % skips the first three rows of data
datae1 = csvread(join([fil,'df_eqopt_1.csv']),1,0);  % skips the first three rows of data

datad0 = csvread(join([fil,'df_dp_0.csv']),1,0);  % skips the first three rows of data
datad1 = csvread(join([fil,'df_dp_1.csv']),1,0);  % skips the first three rows of data

data0_ = data0(2:5,:);
data0_(5,:)= data0(5,:);
data1_ = data1(2:5,:);
data1_(5,:)= data1(5,:);

u = (data0_-data0);
v = (data1_-data1);
u = 0.9 * u;
v = 0.9 * v;

datae0_ = datae0(2:5,:);
datae0_(5,:)= datae0(5,:);
datae1_ = datae1(2:5,:);
datae1_(5,:)= datae1(5,:);
ue = datae0_-datae0;
ve = datae1_-datae1;
ue = 0.9 * ue;
ve = 0.9 * ve;

datad0_ = datad0(2:5,:);
datad0_(5,:)= datad0(5,:);
datad1_ = datad1(2:5,:);
datad1_(5,:)= datad1(5,:);
ud = datad0_-datad0;
vd = datad1_-datad1;

ud = 0.9 * ud;
vd = 0.9 * vd;

figure1 = figure('InvertHardcopy','off','Color',[1 1 1]);

axes1 = axes('Parent',figure1);
hold(axes1,'on');
scale1 = [0.01,0.33,0.4,0.7,0.9];
n = [1,3,5];
n_=[5,3,1]
% for i = 1:5
%     pbaspect([1 1 1])
%     scatter(data0(:,i),data1(:,i),50,'filled','MarkerFaceAlpha',scale1(i),'MarkerFaceColor','b','MarkerEdgeColor','b');
%     hold on
%     scatter(datae0(:,i),datae1(:,i),50,'filled','MarkerFaceAlpha',scale1(i),'MarkerFaceColor','r','MarkerEdgeColor','r');
%     hold on    
%     scatter(datad0(:,i),datad1(:,i),50,'filled','MarkerFaceAlpha',scale1(i),'MarkerFaceColor','g','MarkerEdgeColor','g');
%     hold on
%     if i ~= 5
%         quiver(data0(i,:),data1(i,:),u(i,:),v(i,:),'Color','b','AutoScale','off');
%         hold on
%         quiver(datae0(i,:),datae1(i,:),ue(i,:),ve(i,:),'Color','r','AutoScale','off');
%         hold on
%         quiver(datad0(i,:),datad1(i,:),ud(i,:),vd(i,:),'Color','g','AutoScale','off');
%         hold on
%     end
%     
% end  

for i = n_
    pbaspect([1 1 1])
    scatter(data0(:,i),data1(:,i),50,'filled','MarkerFaceAlpha',scale1(i),'MarkerFaceColor','b','MarkerEdgeColor','b');
    hold on
    scatter(datae0(:,i),datae1(:,i),50,'filled','MarkerFaceAlpha',scale1(i),'MarkerFaceColor','r','MarkerEdgeColor','r');
    hold on    
    scatter(datad0(:,i),datad1(:,i),50,'filled','MarkerFaceAlpha',scale1(i),'MarkerFaceColor','g','MarkerEdgeColor','g');
    hold on 
end  
for i = 1:5
    if i ~= 5
        quiver(data0(i,n),data1(i,n),u(i,n),v(i,n),'Color','b','AutoScale','off');
        hold on
        quiver(datae0(i,n),datae1(i,n),ue(i,n),ve(i,n),'Color','r','AutoScale','off');
        hold on
        quiver(datad0(i,n),datad1(i,n),ud(i,n),vd(i,n),'Color','g','AutoScale','off');
        hold on
    end
    
end  
grid on
pbaspect([1 1 1])

xlim([0.15 0.9])
ylim([0.15 0.9])
% 
% xlim([0.1 0.9])
% ylim([0.7 0.9])


% legend('Unconstrained','Equal of opportunity','Demographic Parity');
set(axes1,'FontName','times new roman','FontSize',12);
set(axes1,'FontName','times new roman','FontSize',12);
% Create ylabel
ylabel('Caucasian','FontSize',12,'FontName','times new roman');

% Create xlabel
xlabel('African American','FontSize',12,'FontName','times new roman');

