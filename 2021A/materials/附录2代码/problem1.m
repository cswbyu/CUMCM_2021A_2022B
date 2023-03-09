clear;clc;
load('data_p1.mat');
R=300;
[r,c]=size(data_p1);
ans=[];
loc=-data_p1;
avg=[];
for h=0.4:0.01:0.6
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
       [r0,c0]=size(x);
       for j=1:r0
           flag=1;
           dis=sqrt((x(j,1)-data_p1(i,1))^2+(y(j,1)-data_p1(i,2))^2+(z(j,1)-data_p1(i,3))^2);
           if sqrt(x(j,1)^2+y(j,1)^2)>150
               flag=0;
           endif
           if dis>0.6
               flag=0;
           endif
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