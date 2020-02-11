% Original Source: http://www.biometrics.tibs.org/datasets/070308_Code_Oscar.zip
% Including here to have everything in one place

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is example code to perform the OSCAR penalized regression from the
% paper, "Simultaneous regression shrinkage, variable selection, and
% supervised clustering of predictors with OSCAR", by Howard D. Bondell and
% Brian J. Reich
%
% Contact: bondell@stat.ncsu.edu
%
% To use this code, the files: OscarReg.m, OscarSeqReg.m, OscarSeqOpt.m,
% and OscarSelect.m need to be in the working directory.
%
% This is an example problem with p=20 covariates and sample size n=25. 
% The first 6 variables each have true coefficient of 1 and are normally
% distributed with a strong AR(1)-type correlation among them.  
% The remaining 14 variables are independent normals with no effect. 
%
% 
%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Minimizes sum of squared error subject to
%
% (1 - c) L_1 + c (pairwise L_infty) <= t
%
% where 0 <= c <= 1
%
% t = prop * T_0
%
% T_0 represents the value of the constraint at an initial solution (either
% least squares or user specified)
%
% and 0 < prop < 1
%
%
% Note that c = 0 is the LASSO
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  
% A function call performs the optimization for each pair over a 2-d grid of (c, prop)
% values specified by the user. The function takes the following inputs and
% gives the following outputs.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
%
% X: The n by p design matrix. Do not include a column for the intercept.
%    The design matrix will be standardized by the function if not already
%    done. The coefficients returned will be for the standardized
%    predictors.
%
% y: The n by 1 response vector. It will be centered by the function, if 
%    not already done.
%
% cvalues: A vector (or scalar) representing the values of c to use in the penalty
% term. These values are parameterized to be between 0 and 1 as above. 
% Should be ordered from smallest to largest, if not, the function will reorder.
%
% propvalues: A vector (or scalar) representing the values of the proportion of the
% norm of the initial estimate to use as the bound. Should be ordered from
% smallest to largest, if not, the function will reorder.
%
% initcoef: An optional p dimensional vector of an initial estimate. The constraint 
% bound is given as a proportion of the achieved value for this initial estimate. 
% Least Squares is used if no value for initcoef is specified. If p > n, one may wish to
% do a ridge regression and specify the solution as the initial estimate to Oscar.
% If doing cross-validation, this initial estimate should be specified as
% the same for each split of the data set, otherwise the bound will not
% have the same value for each split.
% 
% method: Either 1 or 2. The default solver in Matlab's optimization package cannot directly
% perform the full optimization. So method = 2 chooses the sequential method
% as given in the Web Appendix. If the TOMLAB package is available choose
% method = 1 to perform the full optimization in one step using the SQOPT solver (much faster). If the user has
% an alternative solver, lines 112-113 of OscarReg.m can be modified to
% call that solver and then use method = 1 (better), or lines 45-46 of OscarSeqOpt.m can be 
% modified and then use method = 2.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUT:
%
% CoefMatrix: A 3-dimensional array. For each value of c requested, each column 
% represents the p-dimensional solution for each of the chosen proportions.
% 
% dfMatrix: A 3-dimensional array arranged as above representing the 
% corresponding degree of freedom estimate for each solution. This allows
% for computation of GCV, AIC, or BIC if desired.
%
% SSMatrix: A 3-dimensional array arranged as above representing the 
% residual sum of squares for each solution. This allows for computation of 
% GCV, AIC, or BIC if desired.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Generate data
n=25;
p=20;
p0=14;
p1=6;

MuVec=zeros(1,p);
SigmaMat=eye(p);
for i=1:p1
    for j=1:p1
        SigmaMat(i,j)=.9^(abs(i-j));
    end;  
end;
beta=[ones(p1,1);zeros(p0,1)];
X=mvnrnd(MuVec,SigmaMat,n);
for i=1:p
  X(:,i)=(X(:,i)-mean(X(:,i)))/std(X(:,i));
end;

y=normrnd(X*beta,1);
y=(y-mean(y))/std(y);


% Choose grid of parameter values to use
cvalues = [0; .01; .05; .1;.25;.5;.75;.9;1];
propvalues = [.0001; .001; .002; .00225; .0025; .00275; .0028; .0029; .003; .004; .005; .0075; .01;.025; .05; .1; .15;.2;.3;.4;.5;.6];

%%%% method = 2, chooses the sequential algorithm.

method = 2;     
initcoef = [];
[CoefMatrix dfMatrix SSMatrix] = OscarSelect(X, y, cvalues, propvalues, initcoef, method); % Calls function to perform optimization on the grid.

