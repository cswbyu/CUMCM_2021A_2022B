clear;clc;
load('data_adj.mat');
R=300;
%求解方向向量
[r,c]=size(data_adj);
%ans矩阵存储每个主索节点对应的新坐标和与原位置的偏移�?
ans=[];
loc=-data_adj;
avg=[];
for h=0.4:0.01:0.6
    disp(['h=',num2str(h)]);
    ans=[ans;[h,0,0,0]];
    count=0;
    sum=0;
   for i=1:r
       syms x y z;
       eq1=loc(i,2)*x-loc(i,1)*y;
       eq2=loc(i,3)*y-loc(i,2)*z;
       eq3=z+R+h-(x^2+y^2)/(4*h+1.864*R);
       [x,y,z]=solve(eq1,eq2,eq3,x,y,z);
       x=double(x);
       y=double(y);
       z=double(z);
       %确定解的个数
       [r0,c0]=size(x);
       %对每一个解判断其是否满足约束条�?
       for j=1:r0
           flag=1;
           dis=sqrt((x(j,1)-data_adj(i,1))^2+(y(j,1)-data_adj(i,2))^2+(z(j,1)-data_adj(i,3))^2);
           %约束条件1：到主对称轴的距离是否小�?300
           if sqrt(x(j,1)^2+y(j,1)^2)>150
               flag=0;
           endif
           %约束条件2�?
           if dis>0.6
               flag=0;
           endif
           %表示该点可以取到
           if flag==1
               ans=[ans;[x(j,1),y(j,1),z(j,1),dis]];
               count=count+1;
               sum=sum+dis;
           endif
       endfor
    endfor
    ave0=sum/count;
    avg=[avg;[h,ave0]];

endfor