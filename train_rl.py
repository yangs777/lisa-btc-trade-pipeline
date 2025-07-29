import os, gdown, pandas as pd, numpy as np
from stable_baselines3 import PPO
from rl_env import OnePosEnv
from feature_engineering import build_feature_matrix

SAMPLE_ID="1Gq5ZTfrl5PX5iY2l4A8W4I1MxSN5jsbR"
CSV="BTCUSDT_20250624.csv"
if not os.path.exists(CSV):
    gdown.download(f"https://drive.google.com/uc?id={SAMPLE_ID}", CSV, quiet=False)
df=pd.read_csv(CSV)
feat=build_feature_matrix(df)
price=df['price'].values if 'price' in df.columns else np.linspace(1,1.1,len(feat))
env=OnePosEnv(feat.to_numpy(dtype=np.float32), price)
model=PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=256)
model.learn(total_timesteps=5000)
model.save("ppo_singlepos_v1")
print("✅ demo training complete")