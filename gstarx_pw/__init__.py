"""GSTARX-PW: penalized estimation and Wald testing of the spatial weight matrix
in generalized space-time autoregressive models with exogenous variables (GSTARX)."""
from .core import (SETTINGS, gc_km, distance_matrix, inverse_distance, uniform_matrix, knn_matrix, cross_correlation_matrix,
                   sparse_matrix, row_normalize, harmonic_design, design_generic, design_eid_weekly, annihilator,
                   exogenous_stage, simulate, wald_stats, wald_stats_explicit, restriction_matrices,
                   bootstrap_test, proj_simplex_rows, fixed_fit, var_fit, var_implied, pw_fit, select_kappa,
                   kappa_path, mae_offdiag, ljung_box, ljung_box_reject)
from .forecasting import forecast_path, dm_test, rolling_origin
from .datasets import (example_coordinates, weekly_calendar_design, simulate_price_panel, load_irish_wind,
                       EID_AL_FITR)

__version__ = "1.0.0"
