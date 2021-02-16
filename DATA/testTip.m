

clear all, close all, clc


%% Load data Original

addpath('../../CircleFitByPratt/');

% ALWAYS check path !!!!!!!!!!!!!!!!!!!!!!!!
pathh = './results/sym06/'; % <-----
% dataTipGradRaw = importdata([pathh 'dataTip.dat']);

fid = fopen([pathh 'dataparamcsv.csv']);
parameters = textscan(fid,'%s%s','delimiter',',');

parameters{1,:}
tablehight = size(parameters{1,1},1);
table = zeros(tablehight,1);

for i = 3:tablehight
    table(i) = str2num(cell2mat(parameters{:,2}(i)));
end
 
%% Rescale to physical coordinates

start = 17800;
endd = 20000;

p.Lx = table(17);
p.Ly = table(18);
p.Nx = table(15);
p.Ny = table(16);
p.dx = table(19);
p.dy = table(20);

% dataTipGradRaw = dataTipGradRaw(start:20000,:);

xog = (dataTipGradRaw(10000:20000,1) - 1)*p.dx;
yog = (dataTipGradRaw(10000:20000,2) - 1)*p.dy;

XY = [xog yog];
Par = CircleFitByPratt(XY);
cx = Par(1);
cy = Par(2);

dataTipGradRaw = dataTipGradRaw(start:endd,:);

xog = (dataTipGradRaw(:,1) - 1)*p.dx;
yog = (dataTipGradRaw(:,2) - 1)*p.dy;

% Plot
figure;
plot(xog,yog,'b-')
title('Original')
grid on
daspect([1 1 1])

%% Symmetry reduced

% Load data
dataTipGradRaw = importdata([pathh 'dataTip_sym.dat']);

dataTipGradRaw = dataTipGradRaw(start:endd,:);

xtip = (dataTipGradRaw(:,1) - 1)*p.dx;
ytip = (dataTipGradRaw(:,2) - 1)*p.dy;

figure;
plot(xtip,ytip,'b-')
title('Symmetry reduced')
grid on
daspect([1 1 1])

%% Reconstruction

dataSym = importdata([pathh 'c_phi_list_sym.dat']);

% dataSym = dataSym(floor(3*size(dataSym,1)/4):end,:);

dataSym = dataSym(start:endd,:);

phix = dataSym(:,4);
phiy = dataSym(:,5);
phit = dataSym(:,6);

xsym = (-phix - ytip.*sin(-phit) + xtip.*cos(-phit));
ysym = (-phiy + xtip.*sin(-phit) + ytip.*cos(-phit));

figure;
plot(xsym,ysym,'b-')
title('Recosntruction')
grid on
daspect([1 1 1])

XY = [xsym ysym];
Par = CircleFitByPratt(XY);
scx = Par(1);
scy = Par(2);

% plot
figure;
plot(xog,yog,'b-')
hold on
% plot(xsym,ysym,'r-')
% hold on
plot(xsym+(cx-scx),ysym+(cy-scy),'m-')
% plot(xsym,ysym,'m-')
grid on
title('Comparison')
lgd = legend('Original','Recon-Sym','centered-Recon-Sym','Location','southeast');
legend('boxoff')
set(lgd,'color','none');
daspect([1 1 1])

%%

figure;
plot(xog(1:size(xsym,1)))
hold on
plot(xsym+(cx-scx))
grid on
title('x-component of sym and non-sym')

figure;
plot(yog(1:size(ysym,1)))
hold on
plot(ysym+(cy-scy))
grid on
title('y-component of sym and non-sym')

figure;
plot(xog(1:size(xsym,1))-(xsym+(cx-scx)),'b-.')
hold on
plot(yog(1:size(ysym,1))-(ysym+(cy-scy)),'r-.')
grid on
title('Component error')

%%

figure;
plot(phix,'b-o')
hold on
plot(phiy,'r-o')
grid on
title('Phix & Phiy')

%% Velocity field
 
xx = 0:32:(p.Nx-1);
yy = 0:32:(p.Ny-1);
% xx = -(p.Nx-1)/2:32:(p.Nx-1)/2;
% yy = -(p.Ny-1)/2:32:(p.Ny-1)/2;
[X,Y] = meshgrid(xx,fliplr(yy));
cx = dataSym(:,1);
cy = dataSym(:,2);
ct = dataSym(:,3);
phit = dataSym(:,6);

figure;
for i = 20000%:size(dataSym,1)
    t = i;
    convx = -( p.dy*Y*ct(t) - cx(t)*cos(phit(t)) + cy(t)*sin(phit(t)) );
    convy = -( -p.dx*X*ct(t) - cx(t)*sin(phit(t)) - cy(t)*cos(phit(t)) );
    quiver(X,Y,convx,convy)
    axis tight
    daspect([1 1 1])
    hold off
    pause(0.01)
end

