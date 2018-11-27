clc
clear
load('/home/winston/workSpace/PycharmProjects/VDSR_rebuild/OriginalCodeMatconvnet/VDSR model/VDSR.mat');


for i = 1:length(net.layers)
    if i == 39
       t = 40; 
    end
    typeName = net.layers{i}.type;
    fprintf('-----------Layer %d------------------\n',i);
    fprintf('layer type : %s\n',typeName);
    if strcmp(typeName, 'conv')
       % get the kernelsize 
       sz =  size(net.layers{i}.filters);
       if length(sz) == 3
          sh = sz(1);
          in = sz(3);
          out = 1;
       else 
          sh = sz(1);
          in = sz(3);
          out = sz(4);
       end
       
       fprintf('Size of the kernel:%d,\nIn channel:%d,\nOut channel:%d.\n',sh,in,out);
    end
    fprintf('\n\n');
end 

