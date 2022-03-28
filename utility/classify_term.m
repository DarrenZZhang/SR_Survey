function predict = classify_term(train_data,testVec,class_num,coeff,train_num)

feature = size(train_data,2);
contrVir=zeros(feature,class_num);
for kk=1:class_num
    for hh=1:train_num
        contrVir(:,kk)=contrVir(:,kk)+coeff((kk-1)*train_num+hh)*train_data((kk-1)*train_num+hh,:)';
    end
end
%º∆À„±Ì æ≤–≤Ó
for i=1:class_num   
    resiVir(i)=norm(testVec'-contrVir(:,i));     
end

[val predict]=min(resiVir);
