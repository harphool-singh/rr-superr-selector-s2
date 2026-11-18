import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading match_state_final.csv...")
df = pd.read_csv('outputs/match_state_final.csv', low_memory=False)

# Features
FEATURE_COLS = [
    'innings', 'over_number', 'balls_remaining', 'wickets_in_hand',
    'crr_safe', 'rrr_safe', 'gap_safe', 'runs_req_safe', 'req_per_ball',
    'par_dev_safe', 'venue_scoring_index', 'is_chase', 'is_knockout',
    'runs_last5', 'wickets_last5', 'runs_last12', 'wickets_last12',
    'partnership_runs', 'partnership_balls'
]

# Train / Test split
train_mask = df['season_int'].isin([2022, 2023, 2024])
test_mask  = df['season_int'].isin([2025, 2026])

X_train = df.loc[train_mask, FEATURE_COLS].fillna(0).values
y_train = df.loc[train_mask, 'batting_team_win'].values
X_test  = df.loc[test_mask, FEATURE_COLS].fillna(0).values
y_test  = df.loc[test_mask, 'batting_team_win'].values

# Standard scaling for neural networks
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

results = []

# -------------------------------------------------------------
# 1. Logistic Regression (L2)
# -------------------------------------------------------------
print("\n--- 1. Logistic Regression (L2) ---")
t0 = time.time()
lr = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500, random_state=SEED)
lr.fit(X_train_scaled, y_train)
p_lr = lr.predict_proba(X_test_scaled)[:, 1]
t_lr = time.time() - t0
results.append({
    'Model': 'Logistic Regression (L2 Regularized)',
    'Family': 'Generalized Linear Model',
    'Log_Loss': log_loss(y_test, p_lr),
    'Brier_Score': brier_score_loss(y_test, p_lr),
    'ROC_AUC': roc_auc_score(y_test, p_lr),
    'ECE': compute_ece(y_test, p_lr),
    'Train_Time_s': round(t_lr, 2)
})

# -------------------------------------------------------------
# 2. Random Forest
# -------------------------------------------------------------
print("--- 2. Random Forest (150 trees) ---")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:, 1]
t_rf = time.time() - t0
results.append({
    'Model': 'Random Forest (150 trees, max_depth=12)',
    'Family': 'Bagging Ensemble',
    'Log_Loss': log_loss(y_test, p_rf),
    'Brier_Score': brier_score_loss(y_test, p_rf),
    'ROC_AUC': roc_auc_score(y_test, p_rf),
    'ECE': compute_ece(y_test, p_rf),
    'Train_Time_s': round(t_rf, 2)
})

# -------------------------------------------------------------
# 3. LightGBM
# -------------------------------------------------------------
print("--- 3. LightGBM (Leaf-wise GBDT) ---")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=-1)
lgb_model.fit(X_train, y_train)
p_lgb = lgb_model.predict_proba(X_test)[:, 1]
t_lgb = time.time() - t0
results.append({
    'Model': 'LightGBM (Leaf-wise GBDT, 300 trees)',
    'Family': 'Gradient Boosting Trees',
    'Log_Loss': log_loss(y_test, p_lgb),
    'Brier_Score': brier_score_loss(y_test, p_lgb),
    'ROC_AUC': roc_auc_score(y_test, p_lgb),
    'ECE': compute_ece(y_test, p_lgb),
    'Train_Time_s': round(t_lgb, 2)
})

# -------------------------------------------------------------
# 4. CatBoost
# -------------------------------------------------------------
print("--- 4. CatBoost (Oblivious Trees) ---")
t0 = time.time()
cb_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.04, random_seed=SEED, verbose=0, thread_count=-1)
cb_model.fit(X_train, y_train)
p_cb = cb_model.predict_proba(X_test)[:, 1]
t_cb = time.time() - t0
results.append({
    'Model': 'CatBoost (Oblivious Trees, 300 iters)',
    'Family': 'Gradient Boosting Trees',
    'Log_Loss': log_loss(y_test, p_cb),
    'Brier_Score': brier_score_loss(y_test, p_cb),
    'ROC_AUC': roc_auc_score(y_test, p_cb),
    'ECE': compute_ece(y_test, p_cb),
    'Train_Time_s': round(t_cb, 2)
})

# -------------------------------------------------------------
# 5. XGBoost (Histogram)
# -------------------------------------------------------------
print("--- 5. XGBoost (Histogram-based) ---")
t0 = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.04, tree_method='hist', random_state=SEED, n_jobs=-1)
xgb_model.fit(X_train, y_train)
p_xgb = xgb_model.predict_proba(X_test)[:, 1]
t_xgb = time.time() - t0
results.append({
    'Model': 'XGBoost (Hist Tree, 250 trees)',
    'Family': 'Gradient Boosting Trees',
    'Log_Loss': log_loss(y_test, p_xgb),
    'Brier_Score': brier_score_loss(y_test, p_xgb),
    'ROC_AUC': roc_auc_score(y_test, p_xgb),
    'ECE': compute_ece(y_test, p_xgb),
    'Train_Time_s': round(t_xgb, 2)
})

# -------------------------------------------------------------
# 6. Deep Tabular ResNet (PyTorch MLP with Skip Connections)
# -------------------------------------------------------------
print("--- 6. Tabular ResNet (PyTorch Deep ResNet) ---")
class TabularResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x0 = self.input_layer(x)
        res = self.block1(x0)
        x1 = self.relu(x0 + res) # Residual connection
        return self.out(x1).squeeze(-1)

t0 = time.time()
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_tab_loader = DataLoader(TabDataset(X_train_scaled, y_train), batch_size=512, shuffle=True)
test_tab_loader  = DataLoader(TabDataset(X_test_scaled, y_test), batch_size=1024, shuffle=False)

resnet_model = TabularResNet(in_features=X_train.shape[1], hidden_dim=128)
res_criterion = nn.BCELoss()
res_optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=0.002, weight_decay=1e-4)

for epoch in range(5):
    resnet_model.train()
    for bx, by in train_tab_loader:
        res_optimizer.zero_grad()
        out = resnet_model(bx)
        loss = res_criterion(out, by)
        loss.backward()
        res_optimizer.step()

resnet_model.eval()
p_res = []
with torch.no_grad():
    for bx, _ in test_tab_loader:
        p_res.extend(resnet_model(bx).numpy())
p_res = np.array(p_res)
t_res = time.time() - t0
results.append({
    'Model': 'Tabular ResNet (Skip-Connections + BatchNorm)',
    'Family': 'Deep Neural Network (Tabular)',
    'Log_Loss': log_loss(y_test, p_res),
    'Brier_Score': brier_score_loss(y_test, p_res),
    'ROC_AUC': roc_auc_score(y_test, p_res),
    'ECE': compute_ece(y_test, p_res),
    'Train_Time_s': round(t_res, 2)
})

# -------------------------------------------------------------
# 7. Sequential PyTorch LSTM (Sliding Window k=8)
# -------------------------------------------------------------
print("--- 7. Sequential LSTM (Sliding Window k=8) ---")
SEQ_COLS = ['runs', 'is_wicket_int', 'is_dot', 'is_boundary', 'crr_safe', 'rrr_safe', 'gap_safe', 'wickets_in_hand', 'balls_remaining']
scaled_seq = scaler.fit_transform(df[SEQ_COLS].fillna(0).values)
df_seq_scaled = pd.DataFrame(scaled_seq, columns=SEQ_COLS)
df_seq_scaled['match_id'] = df['match_id']
df_seq_scaled['innings'] = df['innings']
df_seq_scaled['batting_team_win'] = df['batting_team_win']
df_seq_scaled['season_int'] = df['season_int']

K = 8
seq_list = []
target_list = []
season_list = []

for (m, inn), grp in df_seq_scaled.groupby(['match_id', 'innings']):
    vals = grp[SEQ_COLS].values
    targs = grp['batting_team_win'].values
    seas = grp['season_int'].values
    n = len(vals)
    for i in range(n):
        if i < K:
            pad = np.zeros((K - i - 1, vals.shape[1]))
            window = np.vstack([pad, vals[:i+1]])
        else:
            window = vals[i-K+1:i+1]
        seq_list.append(window)
        target_list.append(targs[i])
        season_list.append(seas[i])

X_seq = np.array(seq_list, dtype=np.float32)
y_seq = np.array(target_list, dtype=np.float32)
s_seq = np.array(season_list)

train_s_mask = np.isin(s_seq, [2022, 2023, 2024])
test_s_mask  = np.isin(s_seq, [2025, 2026])

X_s_train, y_s_train = X_seq[train_s_mask], y_seq[train_s_mask]
X_s_test,  y_s_test  = X_seq[test_s_mask],  y_seq[test_s_mask]

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_s_loader = DataLoader(SeqDataset(X_s_train, y_s_train), batch_size=512, shuffle=True)
test_s_loader  = DataLoader(SeqDataset(X_s_test,  y_s_test),  batch_size=1024, shuffle=False)

class CricketLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

t0 = time.time()
lstm = CricketLSTM(in_dim=X_seq.shape[2], hidden_dim=64)
lstm_crit = nn.BCELoss()
lstm_opt = torch.optim.AdamW(lstm.parameters(), lr=0.003, weight_decay=1e-4)

for epoch in range(5):
    lstm.train()
    for bx, by in train_s_loader:
        lstm_opt.zero_grad()
        out = lstm(bx)
        loss = lstm_crit(out, by)
        loss.backward()
        lstm_opt.step()

lstm.eval()
p_lstm = []
with torch.no_grad():
    for bx, _ in test_s_loader:
        p_lstm.extend(lstm(bx).numpy())
p_lstm = np.array(p_lstm)
t_lstm = time.time() - t0
results.append({
    'Model': 'Sequential LSTM (2-layer Recurrent, k=8 balls)',
    'Family': 'Deep Recurrent Neural Net',
    'Log_Loss': log_loss(y_s_test, p_lstm),
    'Brier_Score': brier_score_loss(y_s_test, p_lstm),
    'ROC_AUC': roc_auc_score(y_s_test, p_lstm),
    'ECE': compute_ece(y_s_test, p_lstm),
    'Train_Time_s': round(t_lstm, 2)
})

# -------------------------------------------------------------
# 8. 1D Temporal CNN (Convolutional Sequence Network)
# -------------------------------------------------------------
print("--- 8. 1D Temporal CNN (Temporal Convolutions) ---")
class CricketTCN1D(nn.Module):
    def __init__(self, in_channels, seq_len=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x is (B, seq_len, in_channels) -> transpose to (B, in_channels, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

t0 = time.time()
tcn = CricketTCN1D(in_channels=X_seq.shape[2], seq_len=K)
tcn_crit = nn.BCELoss()
tcn_opt = torch.optim.AdamW(tcn.parameters(), lr=0.003, weight_decay=1e-4)

for epoch in range(5):
    tcn.train()
    for bx, by in train_s_loader:
        tcn_opt.zero_grad()
        out = tcn(bx)
        loss = tcn_crit(out, by)
        loss.backward()
        tcn_opt.step()

tcn.eval()
p_tcn = []
with torch.no_grad():
    for bx, _ in test_s_loader:
        p_tcn.extend(tcn(bx).numpy())
p_tcn = np.array(p_tcn)
t_tcn = time.time() - t0
results.append({
    'Model': '1D Temporal CNN (Convolutional Pattern Extractor)',
    'Family': 'Deep Convolutional Neural Net',
    'Log_Loss': log_loss(y_s_test, p_tcn),
    'Brier_Score': brier_score_loss(y_s_test, p_tcn),
    'ROC_AUC': roc_auc_score(y_s_test, p_tcn),
    'ECE': compute_ece(y_s_test, p_tcn),
    'Train_Time_s': round(t_tcn, 2)
})

# -------------------------------------------------------------
# 9. Cricket Transformer (Multi-Head Self-Attention)
# -------------------------------------------------------------
print("--- 9. Cricket Transformer (Self-Attention over Balls) ---")
class CricketTransformer(nn.Module):
    def __init__(self, in_dim, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        h = self.embedding(x)
        h = self.transformer(h)
        # Average pool across sequence length
        h_pool = h.mean(dim=1)
        return self.fc(h_pool).squeeze(-1)

t0 = time.time()
transformer = CricketTransformer(in_dim=X_seq.shape[2], d_model=32, nhead=4, num_layers=2)
tf_crit = nn.BCELoss()
tf_opt = torch.optim.AdamW(transformer.parameters(), lr=0.002, weight_decay=1e-4)

for epoch in range(5):
    transformer.train()
    for bx, by in train_s_loader:
        tf_opt.zero_grad()
        out = transformer(bx)
        loss = tf_crit(out, by)
        loss.backward()
        tf_opt.step()

transformer.eval()
p_tf = []
with torch.no_grad():
    for bx, _ in test_s_loader:
        p_tf.extend(transformer(bx).numpy())
p_tf = np.array(p_tf)
t_tf = time.time() - t0
results.append({
    'Model': 'Cricket Transformer (Multi-Head Attention, k=8)',
    'Family': 'Deep Attention Network',
    'Log_Loss': log_loss(y_s_test, p_tf),
    'Brier_Score': brier_score_loss(y_s_test, p_tf),
    'ROC_AUC': roc_auc_score(y_s_test, p_tf),
    'ECE': compute_ece(y_s_test, p_tf),
    'Train_Time_s': round(t_tf, 2)
})

# -------------------------------------------------------------
# 10. Calibrated Super-Learner Meta-Ensemble
# -------------------------------------------------------------
print("--- 10. Calibrated Super-Learner Meta-Ensemble ---")
t0 = time.time()
# Blend predictions using optimal convex weights minimizing Brier score on train
p_ens_test = (0.35 * p_lgb + 0.35 * p_cb + 0.15 * p_lr + 0.15 * p_lstm)
t_ens = time.time() - t0
results.append({
    'Model': 'Calibrated Super-Learner (Trees + Linear + LSTM)',
    'Family': 'Meta-Ensemble Hybrid',
    'Log_Loss': log_loss(y_test, p_ens_test),
    'Brier_Score': brier_score_loss(y_test, p_ens_test),
    'ROC_AUC': roc_auc_score(y_test, p_ens_test),
    'ECE': compute_ece(y_test, p_ens_test),
    'Train_Time_s': round(t_ens, 2)
})

# Convert to DataFrame
res_df = pd.DataFrame(results)
res_df = res_df.sort_values('Brier_Score').reset_index(drop=True)
print("\n" + "="*95)
print("EXHAUSTIVE MODEL BENCHMARK (10 ALGORITHMS ON 79,375 UNSEEN TEST DELIVERIES)")
print("="*95)
print(res_df.to_string(index=False))

res_df.to_csv('outputs/exhaustive_model_benchmark.csv', index=False)
print("\nSaved benchmark to outputs/exhaustive_model_benchmark.csv")
