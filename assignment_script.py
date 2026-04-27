# Import all at once
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

# Data imports and exploring
df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv")
df['treatment'] = (df['version'] == 'gate_40').astype(int)
df['gamerounds_std'] = (df['sum_gamerounds'] - df['sum_gamerounds'].mean()) / df['sum_gamerounds'].std() # gamerounds will be used as a control variable; need normalization before being used
df.head(6)

# Subsampling
# Original data consist of 90,189 raws; Too much running time
df = df.sample(10000, random_state=42).reset_index(drop=True)

# Modeling Retention_1
coords = {"observation": df.index.values}
with pm.Model(coords=coords) as bayesian_model:
    treatment  = pm.Data("treatment",  df['treatment'].values,  dims="observation")
    gamerounds = pm.Data("gamerounds", df['gamerounds_std'].values,   dims="observation")
    retention1 = pm.Data("retention1", df['retention_1'].values, dims="observation")
    # Priors (normal distribution)
    β0         = pm.Normal("β0",         mu=0, sigma=1)
    β_treatment = pm.Normal("β_treatment", mu=0, sigma=1)
    β_gamerounds  = pm.Normal("β_gamerounds",  mu=0, sigma=1)
    # Linear model
    μ = β0 + β_treatment * treatment + β_gamerounds * gamerounds #gamerounds as a control variable
    p = pm.math.sigmoid(μ)
    likelihood = pm.Bernoulli("y", p=p, observed=retention1, dims="observation")

# Fit & sample
with bayesian_model:
    trace = pm.sample(5000, return_inferencedata=True, target_accept=0.9) #NUTS instead of Metropolis
burned_trace = trace.sel(draw=slice(500, None))

# Results & Plots
print(az.summary(burned_trace, var_names=["β0", "β_treatment", "β_gamerounds"]))
az.plot_posterior(burned_trace, var_names=["β0", "β_treatment", "β_gamerounds"])


# Modeling Retention_7
coords = {"observation": df.index.values}
with pm.Model(coords=coords) as bayesian_model:
    treatment  = pm.Data("treatment",  df['treatment'].values,  dims="observation")
    gamerounds = pm.Data("gamerounds", df['gamerounds_std'].values,   dims="observation")
    retention7 = pm.Data("retention7", df['retention_7'].values, dims="observation")
    # Priors (normal distribution)
    β0         = pm.Normal("β0",         mu=0, sigma=1)
    β_treatment = pm.Normal("β_treatment", mu=0, sigma=1)
    β_gamerounds  = pm.Normal("β_gamerounds",  mu=0, sigma=1)
    # Linear model
    μ = β0 + β_treatment * treatment + β_gamerounds * gamerounds #gamerounds as a control variable
    p = pm.math.sigmoid(μ)
    likelihood = pm.Bernoulli("y", p=p, observed=retention7, dims="observation")

# Fit & sample
with bayesian_model:
    trace = pm.sample(5000, return_inferencedata=True, target_accept=0.9) #NUTS instead of Metropolis
burned_trace = trace.sel(draw=slice(500, None))

# Results & Plots
print(az.summary(burned_trace, var_names=["β0", "β_treatment", "β_gamerounds"]))
az.plot_posterior(burned_trace, var_names=["β0", "β_treatment", "β_gamerounds"])