function [cm]= svmconfmat(g,ghat)
// Confusion matrix for classification algorithms.
// Calling Sequence
//   cm = svmconfmat(g, ghat)
//  Parameters
//  g: known group labels
//  ghat: predicted group labels
//  cm : confusion matrix determined   by the known group labels g and the predicted group labels ghat
//  Description
//    The confusion matrix CM(I,J) represents the count of instances
//    whose known group labels are group J and whose predicted group labels
//    are group I.
//   Examples
//    [label,instance]=libsvmread("demos/heart_scale");
//    cc = svmtrain(label,instance);
//    [predicted_label]=svmpredict(label,instance,cc);
//    cm = svmconfmat(label,  predicted_label);
//    


        [nargout,nargin]=argn(0);
	//rand('state',0); // reset random seed


if length(g)~=length(ghat) 
   error('g and ghat have different lengths')
end

 c = unique(g);
// cm=zeros(length(c),length(c));
// 
// for i = 1:length(c)
//    for j = 1:length(c)
//       cm(i,j) = (sum ((g==j).*(ghat==i))/sum(ghat == i))*100;
//    end
// end

iemod=ieee();
ieee(2);

L = length(c);
cm = zeros(L,L);
for ii = 1:L
   Tii = ghat==c(ii);
   S = sum(Tii);
   for jj = 1:L
      cm(ii,jj) = sum ((g==c(jj)) & Tii);
   end
end
ieee(iemod);
endfunction
