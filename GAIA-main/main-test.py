from api import SyntheticEnv, Gaia, infer_gaia

env = SyntheticEnv()
model = Gaia(env.f_soil, env.f_water, env.f_weather, env.action_dim)
# ... load trained weights if needed
result = infer_gaia(model, soil=[0.1]*16, water=[-0.2]*16, weather=[0.05]*8)
print(result)
