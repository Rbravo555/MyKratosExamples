function [z,w,ERROR_GLO,DATAOUT]= EmpiricalCubatureMethod_CANDcompl(BasisF,DATA)
% Version created 23-FEB-2021 to cope with clustering 


G = BasisF' ;
W = ones(size(G,2),1);
b = G*W;  % b Vector (exact integral)
bEXACT =b ;
nbEXACT = norm(bEXACT) ;




Gnorm =sqrt(sum(G.*G,1)) ; % Norm of Columns of G
M = size(G,2) ;  % Number of FE points
DATA = DefaultField(DATA,'TOL',1e-6) ; % Default tolerance for convergence
TOL = DATA.TOL ;
% INITIALIZATIONS
% ------------------------
z = [] ; % Set of integration points
% Set of candidate points (those whose associated column has low norm are removed)

%  PointsWithZero =  find(sum(G(1:end-1,:),1)==0) ;

y=1:M ;
%y(PointsWithZero) = []  ;
DATA = DefaultField(DATA,'TOLFilterCandidatePoints',1e-16) ;
GnormNOONE =sqrt(sum(G(1:end-1,:).*G(1:end-1,:),1)) ; % Norm of Columns of G

if DATA.TOLFilterCandidatePoints >0
    TOL_REMOVE = DATA.TOLFilterCandidatePoints*norm(b) ;
    rmvpin = find(GnormNOONE(y)<TOL_REMOVE) ;
    y(rmvpin) = [] ;
end

DATA = DefaultField(DATA,'RemoveColumnsWithNegativeProjection',0); % 4-Dec-2019



DATA = DefaultField(DATA,'IND_POINTS_CANDIDATES',[]) ;

if ~isempty(DATA.IND_POINTS_CANDIDATES)
    y = intersect(y,DATA.IND_POINTS_CANDIDATES) ;
end
yORIG = y ; 

DATA  = DefaultField(DATA,'USE_SINGULAR_VALUES_INTF_FOR_MEASURING_ECM_ERROR',1) ;  % 27th-april-2020
USEsingvERR = DATA.USE_SINGULAR_VALUES_INTF_FOR_MEASURING_ECM_ERROR ; 


alpha = [] ; % Vector of weights
mPOS = 0 ; % Number of nonzero weights
r = b ; % Residual vector
k = 1;  % Number of iterations
errorGLO = [] ; %(for storing error)
% Default number of points
DATA = DefaultField(DATA,'npoints',length(b)) ;
m = min(DATA.npoints,length(b)) ;
% END INITIALIZATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
normB = norm(b) ;
nerror = norm(r)/normB  ;
H = [] ; % Inverse of (Gz'*Gz)
ERROR_GLO = [] ;
NPOINTS =[] ;
nerrorACTUAL = nerror;
y = y(:) ;
NITERACIONES = 10*m ; 
 iOLD = [] ; 
 yCOMPL = (1:length(W))'; 
 yCOMPL(y) = [] ; % Complementary set of the candidate set (in case this set is not sufficiently rich)
 
 
NITERATIONS_NO_MORE_POINTS = 10 ; %    NUMBER OF ITERATIONS IN WHICH THE CODE IS ALLOWED TO "ITERATE" IN ORDER TO FIND
                                  %  THE MINIMUM 
 
ITER_MINIMUM = 0;  
MAX_NUMBER_POINTS = 0 ; 
while  nerrorACTUAL >TOL && mPOS <m   && ~isempty(y) && k<NITERACIONES
    % TIMELOC_k = tic ;
    % STEP 1. Compute new point
    ObjFun = G(:,y)'*r ;
    ObjFun = ObjFun./Gnorm(y)';
    [maxLOC, indSORT] = max(ObjFun)  ;
    i = y(indSORT(1)) ;
    % STEP 2.  Update alpha and H  (unrestricted least-squares)
    if k==1
        alpha =  G(:,i)\b ;
        H = 1/(G(:,i)'*G(:,i)) ;
    else
        [H alpha] = UpdateWeightsInverse(G(:,z),H,G(:,i),alpha,r) ;
    end
    % STEP 3. Move i from set y to set z
    z = [z;i] ;     y(indSORT(1)) = [] ; 
    % STEP 4. Find possible negative weights
    n = find(alpha<=0) ;
    if  ~isempty(n)
        % STEP 5
        y = [y; z(n)];  z(n)=[] ;
        H = MultiUpdateInverseHermitian(H,n) ;
        % Recomputing alpha
        alpha = H*(G(:,z)'*b );
        if ITER_MINIMUM > NITERATIONS_NO_MORE_POINTS
            disp('--------------------------------------------------------------')
            disp(['The algorithm cannot proceed with the current set of points'])
            disp(['We enlarge the set of candidates with the complementary set'])
            y = [y;yCOMPL] ; 
        end
       % ITER_MINIMUM = ITER_MINIMUM + 1; 
    else
       % ITER_MINIMUM = 0 ; 
    end
    
    if length(z) > MAX_NUMBER_POINTS 
        ITER_MINIMUM = 0 ; 
    else
        ITER_MINIMUM = ITER_MINIMUM + 1; 
    end
    
     iOLD = i ; 
    % STEP 6
    r = b-G(:,z)*alpha ;
    nerrorACTUAL = nerror ;
    % STEP 7
    disp(['k = ',num2str(k),', m=',num2str(length(z)),' ,','; error n(res)/n(b) (%) = ',...
        num2str(nerror*100),';  Actual error % =',num2str(nerrorACTUAL*100)]) ;
    ERROR_GLO(k) = nerrorACTUAL ;
    NPOINTS(k) =  length(z) ;
    
    mPOS = length(z) ;
    MAX_NUMBER_POINTS = max(mPOS,MAX_NUMBER_POINTS) ; 
    k = k + 1 ;
    
    %     if length(z) == m
    %         dbstop('88')
    %         disp('')
    %     end
    
end


if  k>= NITERACIONES
    PROPORpoints = length(yORIG)/length(W)*100; 
    error(['NO CONVERGENCE. ENLARGE THE SET OF CANDIDATE POINTS  (NOW IT CONTAINS ',num2str(PROPORpoints),' % of the total number of Gauss points)'])
end 


w = alpha.*sqrt(W(z)) ;

disp(['Total number of iterations =',num2str(k)])

figure(500)
hold on
xlabel('Number of points')
ylabel('Error (%)')
plot(NPOINTS,ERROR_GLO*100,'k')

DATAOUT.kiteraciones = k ;

% figure(501)
% hold on
% xlabel('Number of iterations')
% ylabel('Number of points')
% plot(NPOINTS,'k')

end



function  DATAGEN = DefaultField(DATAGEN,FIELDVAR,dEFval) ; 
% DefaultField adds the field FIELDVAR to structure array DATAGEN and
% assigns the specified value dEFval. 
if ~isfield(DATAGEN,FIELDVAR)
    DATAGEN.(FIELDVAR) = dEFval ; 
end

end

function [Bast x] = UpdateWeightsInverse(A,Aast,a,xold,r)

if nargin == 0
    format long g
    P = 100 ; 
    k = 49 ; 
    A = randn(P,k) ; 
    a = randn(P,1) ;
    Aast = inv(A'*A) ; 
    B = [A a] ; 
    BastReal = inv(B'*B) ;
    b = randn(P,1) ; 
    xold = Aast*(A'*b) ; 
    r = b - A*xold ; 
    xREAL = BastReal*(B'*b) ;
    
 
end
c = A'*a ; 
d = Aast*c ; 
s = a'*a-c'*d ; 
Bast = [Aast + (d*d')/s  -d/s; -d'/s  1/s] ; 
v = a'*r/s ; 
x = [(xold -d*v ); v] ;

 
end


function Ahinv = MultiUpdateInverseHermitian(Binv,jrowMAT)
% Recursive application of UpdateInverseHermitian
%  J.A. Hdez, 24 March 2017
if nargin == 0
    m = 10; n =7; 
    jrowMAT =[ 8] ; 
    A = randn(m,n) ; a = randn(m,length(jrowMAT)) ;
    Bor = zeros(m,n+length(jrowMAT)) ; 
    indOR = 1:size(Bor,2) ; 
    indOR(jrowMAT) = [] ; 
    Bor(:,indOR) = A ; 
    Bor(:,jrowMAT) = a;     
    B = [Bor'*Bor] ;  
    Binv = inv(B);     
    AhinvREAL = inv(A'*A)      ;
    
end
 
jrowMAT = sort(jrowMAT) ; 
BinvOLD = Binv ;


for i = 1:length(jrowMAT)
    jrow = jrowMAT(i)-i+1 ; 
    Ahinv = UpdateInverseHermitian(BinvOLD,jrow) ; 
    BinvOLD = Ahinv ; 
end
end


function Ahinv = UpdateInverseHermitian(Binv,jrow)
% Let B = ([A a]^T*[A a]), where A is a m-by-n full rank  matrix, and "a" a vector of
% m entries. Suppose that we are given the inverse of B (Binv). Function
% UpdateInverseHermitian(Binv,j) returns the value of inv(A^T A) based on the block
% decomposition of % matrix Binv
%
% If the second argument jrow is passed, then it is assumed that B is given
% by
%
%  B = [A(:,1:jrow-1) a A(:,jrow:end)] ;
%
%  J.A. Hdez, 24 March 2017
if nargin == 0
    m = 10; n =4;
    jrow =1 ;
    A = randn(m,n) ; a = randn(m,1) ;
    Bor = [A(:,1:jrow-1) a A(:,jrow:end)] ;
    B = [Bor'*Bor] ;
    Binv = inv(B);
    AhinvREAL = inv(A'*A) 
    
    
end

if nargin ==1
    jrow = size(Binv,2) ;
end

if jrow == size(Binv,2)
    Ahinv = Binv(1:end-1,1:end-1) -  (Binv(1:end-1,end)*Binv(end,1:end-1))/Binv(end,end) ;
else
    AUX = [Binv(:,1:jrow-1)   Binv(:,jrow+1:end)   Binv(:,jrow)] ;
    AUX = [AUX(1:jrow-1,:) ;  AUX(jrow+1:end,:)  ; AUX(jrow,:)] ;
    
    Ahinv = AUX(1:end-1,1:end-1) -  (AUX(1:end-1,end)*AUX(end,1:end-1))/AUX(end,end) ;
    
end

end


