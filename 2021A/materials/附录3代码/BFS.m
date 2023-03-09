clc;clear;
global  len_edges all_edges 
load('data_A.mat')
node2idx = containers.Map(node_name,1:size(node_name,1));
adj = zeros(2226,2226);
for i=1:size(node_conn,1)
    temp1 = node2idx(node_conn(i,1));
    temp2 = node2idx(node_conn(i,2));
    temp3 = node2idx(node_conn(i,3));
    adj(temp1,temp2) = adj(temp1,temp2) + 1;
    adj(temp2,temp3) = adj(temp2,temp3) + 1;
    adj(temp1,temp3) = adj(temp1,temp3) + 1;
endfor
adj = (adj+adj')>0;
G = graph(adj);
all_edges = table2array(G.Edges);
rel_nodes = node_pos(all_edges(:,1),:) - node_pos(all_edges(:,2),:);
len_edges = sqrt(rel_nodes(:,1).^2 + rel_nodes(:,2).^2 + rel_nodes(:,3).^2);
vflag=zeros(size(node_name,1),1);
for i=1:size(all_edges)
    if vflag(all_edges(i,2))==0
        a=-len_edges(i)*0.0007;
        b=len_edges(i)*0.0007;
        t=a+(b-a)*rand(1);
        A=node_pos(all_edges(i,1),1)-node_pos(all_edges(i,2),1);
        B=node_pos(all_edges(i,1),2)-node_pos(all_edges(i,2),2);
        C=node_pos(all_edges(i,1),3)-node_pos(all_edges(i,2),3);
        A1=A/sqrt(A^2+B^2+C^2);
        B1=B/sqrt(A^2+B^2+C^2);
        C1=C/sqrt(A^2+B^2+C^2);
      node_pos(all_edges(i,2),1)=node_pos(all_edges(i,2),1)+t*A1;
      node_pos(all_edges(i,2),2)=node_pos(all_edges(i,2),2)+t*B1;
      node_pos(all_edges(i,2),3)=node_pos(all_edges(i,2),3)+t*C1;
      vflag(all_edges(i,2))=1;
    endif
endfor
    