# model.py
import torch
import gpytorch

class SpatioTemporalGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel_cfg=None, input_dim=3):
        # kernel_cfg: dict, e.g. {"spatial_kernel":"Matern", "temporal_kernel":"Periodic", "matern_nu":1.5, "period":24.0}
        kernel_cfg = kernel_cfg or {}
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=kernel_cfg.get("learn_inducing", True)
        )
        super().__init__(variational_strategy)

        # Spatial kernel (active_dims 0,1)
        spatial_kind = kernel_cfg.get("spatial_kernel", "Matern")
        if spatial_kind == "RBF":
            base_spatial = gpytorch.kernels.RBFKernel(ard_num_dims=2, active_dims=[0, 1])
        else:  # Matern
            matern_nu = kernel_cfg.get("matern_nu", 1.5)
            base_spatial = gpytorch.kernels.MaternKernel(nu=matern_nu, ard_num_dims=2, active_dims=[0, 1])

        spatial_kernel = gpytorch.kernels.ScaleKernel(base_spatial)

        # Temporal kernel (active_dim 2)
        temporal_kind = kernel_cfg.get("temporal_kernel", "Periodic")
        if temporal_kind == "Periodic":
            period = kernel_cfg.get("period_length", 24.0)
            base_temp = gpytorch.kernels.PeriodicKernel(period_length=period, active_dims=[2])
        else:
            base_temp = gpytorch.kernels.RBFKernel(active_dims=[2])

        temporal_kernel = gpytorch.kernels.ScaleKernel(base_temp)

        # Combine kernels (sum, or product if desired)
        combine = kernel_cfg.get("combine", "sum")  # "sum" or "product"
        if combine == "product":
            covar = spatial_kernel * temporal_kernel
        else:
            covar = spatial_kernel + temporal_kernel

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
