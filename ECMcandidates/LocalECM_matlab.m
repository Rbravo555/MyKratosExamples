
clc;clear;


for j=1:1
    step_string = string(j-1);
    loasing_file = strcat('Basis', step_string);
    loasing_file = strcat(loasing_file, '.mat');
    load(loasing_file)
    % single set of selected elements and positive weights
    DATA = [] ;
    if j>1
        DATA.IND_POINTS_CANDIDATES = global_elements;
    end
    [e,w] = EmpiricalCubatureMethod_CANDcompl(SnapshotMatrix,DATA);%EmpiricalCubatureMethod_CANDcompl or  EmpiricalCubatureMethod
    elements{j} = e;
    weights{j} = w;
    if j==1
       global_elements = e;
    else
       global_elements = unique([global_elements;e]);
    end
    
    
    
    
end

global_elements
% W_svd = ones(size(ResidualProjected,1),1);
% % Empirical Cubature Method
% DATA = [] ; 
% DATA.IncludeSingularValuesF  = 0 ; % Singular Values are not included in the minimization norm
% DATA.TOLFilterCandidatePoints = 1e-10;
% [elements,weights]= EmpiricalCubatureMethod(U_svd,S_svd,W_svd,DATA);



