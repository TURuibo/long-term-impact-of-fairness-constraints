clc;
clear all;
close all;

data0 = csvread('df_un_0.csv',1,0);  % skips the first three rows of data
data1 = csvread('df_un_1.csv',1,0);  % skips the first three rows of data


datae0 = csvread('df_eqopt_0.csv',1,0);  % skips the first three rows of data
datae1 = csvread('df_eqopt_1.csv',1,0);  % skips the first three rows of data

datad0 = csvread('df_dp_0.csv',1,0);  % skips the first three rows of data
datad1 = csvread('df_dp_1.csv',1,0);  % skips the first three rows of data

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
t = 0.1:0.2:0.9;
az = zeros(1,5);
color1 =['#D9FFFF';'#D9FFFE';'#D9FFFD';'#D9FFFC';'#D9FFFB'];
for i = 1:5
    hold on;
    if i == 1
        plot3(data0(i,:),data1(i,:),t,'o','Color','b','MarkerSize',6,'MarkerFaceColor','#D9FFFF','DisplayName','Unconstrained');
        plot3(datae0(i,:),datae1(i,:),t,'o','Color','r','MarkerSize',6,'MarkerFaceColor','#FFCC00','DisplayName','Equal of opportunity');
        plot3(datad0(i,:),datad1(i,:),t,'o','Color','g','MarkerSize',6,'MarkerFaceColor','#99FF33','DisplayName','Demographic Parity');
    end
    
    plot3(data0(i,:),data1(i,:),t,'o','Color','b','MarkerSize',6,'MarkerFaceColor','#D9FFFF');
    plot3(datae0(i,:),datae1(i,:),t,'o','Color','r','MarkerSize',6,'MarkerFaceColor','#FFCC00');
    plot3(datad0(i,:),datad1(i,:),t,'o','Color','g','MarkerSize',6,'MarkerFaceColor','#99FF33');
    if i ~= 5
        quiver3(data0(i,:),data1(i,:),t,u(i,:),v(i,:),az,'Color','b','AutoScale','off','MaxHeadSize',0.3);
        quiver3(datae0(i,:),datae1(i,:),t,ue(i,:),ve(i,:),az,'Color','r','AutoScale','off','MaxHeadSize',0.3);
        quiver3(datad0(i,:),datad1(i,:),t,ud(i,:),vd(i,:),az,'Color','g','AutoScale','off','MaxHeadSize',0.3);
    end
    
    z=t(i);
    X = [0,0,1,0];
    Y = [0,1,1,0];
    Z = [z,z,z,z];
    h2=fill3(X,Y,Z,[0 0 1]);
    h2.FaceColor=[0 0 z];
    h2.FaceAlpha=0.2;
end
z=t(i-1);
X = [0,0,1,1];
Y = [0,0,1,1];
Z = [0,1,1,0];
h2=fill3(X,Y,Z,[0 0 1]);
h2.FaceColor=[0 0 z];
h2.FaceAlpha=0.3;

grid on
pbaspect([1 1 1])
xlim([0.1 0.85])
ylim([0.1 0.85])
zlim([0 1])
xlabel('AA') 
ylabel('C') 
zlabel('T10') 
view([-54.9172017446225 10.6843818019898]);
legend('Unconstrained','Equal of opportunity','Demographic Parity');
