kernels = cell(10,1);
scalar_kers = cell(10,1);

% polynomial kernel
% p = 2;
kernel_poly = @(x,y)((y'*x+1).^p);
scalar_ker_poly = @(x)((x+1).^p);

kernels{6} = kernel_poly;
scalar_kers{6} = scalar_ker_poly;

% % sigmoidal function
% kernel_tanh = @(x,y)(tanh(2*y'*x+1));
% scalar_ker_tanh = @(x,sd)(tanh(2*x+1));

% rbf function
kernel_rbf = @(x,y)(exp(-sum((x-y).^2)/(2*sd^2)));
scalar_ker_rbf = @(x)(exp(-x/(2*sd^2)));
kernels{7} = kernel_rbf;
scalar_kers{7} = scalar_ker_rbf;

% laplacian function
kernel_lap = @(x,y)(exp(-sum(abs(x-y))/sd));
scalar_ker_lap = @(x)(exp(-x/sd));
kernels{8} = kernel_lap;
scalar_kers{8} = scalar_ker_lap;

% exponential function
kernel_exp = @(x,y)(exp(-sum(abs(x-y))/(2*sd.^2)));
scalar_ker_exp = @(x)(exp(-x/(2*sd.^2)));
kernels{9} = kernel_exp;
scalar_kers{9} = scalar_ker_exp;

% relu function
kernel_relu = @(x,y)(max(x'*y,0));
scalar_ker_relu = @(x)(max(x,0));
kernels{10} = kernel_relu;
scalar_kers{10} = scalar_ker_relu;
