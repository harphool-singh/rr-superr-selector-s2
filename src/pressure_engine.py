import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

print("Loading match_state_deep.csv...")
df = pd.read_csv('outputs/match_state_deep.csv', low_memory=False)

# Match level team runs
df['innings_runs_total'] = df.groupby(['match_id', 'innings'])['total_ball_runs'].transform('sum')

# Legal balls
legal_df = df[df['is_legal'] == 1].copy()

# Pressure is top 25% leverage index
legal_p  = legal_df[legal_df['pressure_ens_75'] == 1].copy()
legal_np = legal_df[legal_df['pressure_ens_75'] == 0].copy()

# Roles
roles = pd.read_csv('outputs/player_roles_classified.csv')

# Global IPL Pressure Benchmarks
ipl_p = legal_p[legal_p['is_ipl']].copy()
IPL_PAR_ECON = 6 * ipl_p['total_ball_runs'].sum() / len(ipl_p) # ~10.92
IPL_PAR_SR   = 100 * ipl_p['runs'].sum() / len(ipl_p)          # ~175.2
IPL_PAR_BND  = 100 * ipl_p['is_boundary'].sum() / len(ipl_p)    # ~25.4%
IPL_PAR_DOT  = 100 * ipl_p['is_dot'].sum() / len(ipl_p)         # ~32.8%
IPL_PAR_DIS  = 100 * ipl_p['is_wicket_int'].sum() / len(ipl_p)  # ~7.8%

print(f"IPL Pressure Par: Economy={IPL_PAR_ECON:.2f}, SR={IPL_PAR_SR:.1f}, Bnd%={IPL_PAR_BND:.1f}%, Dot%={IPL_PAR_DOT:.1f}%")

# ==============================================================================
# PART 1: 10-FACTOR BATTER RATINGS
# ==============================================================================
print("\n--- Engineering 10 Dimensions for Batters ---")

# D1 & D2: WPA per ball and RAE per ball
b_p_agg = legal_p.groupby(['striker_id', 'striker_name']).agg(
    balls_p=('runs', 'count'),
    runs_p=('runs', 'sum'),
    wkts_p=('is_wicket_int', 'sum'),
    dots_p=('is_dot', 'sum'),
    bnd_p=('is_boundary', 'sum'),
    wpa_sum_p=('wpa_bat_ens', 'sum'),
    rae_sum_p=('runs_above_expected', 'sum'),
    xw_sum_p=('wicket_above_expected', 'sum'),
    matches_p=('match_id', 'nunique'),
    ipl_balls_p=('is_ipl', 'sum')
).reset_index()

b_np_agg = legal_np.groupby(['striker_id', 'striker_name']).agg(
    balls_np=('runs', 'count'),
    runs_np=('runs', 'sum'),
    wkts_np=('is_wicket_int', 'sum'),
    dots_np=('is_dot', 'sum'),
    bnd_np=('is_boundary', 'sum')
).reset_index()

# D3: Lone Warrior Index: Striker run share during innings
striker_inn_runs = legal_df.groupby(['match_id', 'innings', 'striker_id', 'striker_name'])['runs'].sum().reset_index()
striker_inn_runs = striker_inn_runs.merge(legal_df[['match_id', 'innings', 'innings_runs_total']].drop_duplicates(), on=['match_id', 'innings'])
striker_inn_runs['run_share'] = striker_inn_runs['runs'] / striker_inn_runs['innings_runs_total'].clip(1)
lone_warrior = striker_inn_runs.groupby('striker_id')['run_share'].mean().reset_index().rename(columns={'run_share': 'lone_warrior_share'})

# D4: Steep Chase Dominance: RRR >= 10.0 in chase
chase_steep = legal_p[(legal_p['is_chase'] == 1) & (legal_p['rrr'] >= 10.0)].groupby('striker_id').agg(
    balls_steep=('runs', 'count'),
    runs_steep=('runs', 'sum'),
    wpa_steep=('wpa_bat_ens', 'sum')
).reset_index()
chase_steep['sr_steep'] = 100 * chase_steep['runs_steep'] / chase_steep['balls_steep'].clip(1)
chase_steep['wpa_pb_steep'] = chase_steep['wpa_steep'] / chase_steep['balls_steep'].clip(1)

# D8: Era-Adjusted Scoring (2024-2026 inflation era)
era_recent = legal_p[legal_p['season_int'] >= 2024].groupby('striker_id').agg(
    balls_recent=('runs', 'count'),
    wpa_recent=('wpa_bat_ens', 'sum'),
    rae_recent=('runs_above_expected', 'sum')
).reset_index()
era_recent['rae_pb_recent'] = era_recent['rae_recent'] / era_recent['balls_recent'].clip(1)

# D10: Positive Innings Consistency
inn_wpa_b = legal_p.groupby(['striker_id', 'match_id'])['wpa_bat_ens'].sum().reset_index()
b_cons = inn_wpa_b.groupby('striker_id').agg(
    pos_inn_rate=('wpa_bat_ens', lambda x: (x > 0).mean()),
    wpa_std=('wpa_bat_ens', 'std')
).reset_index()

# Merge all batter features
b_all = b_p_agg.merge(b_np_agg, on=['striker_id', 'striker_name'], how='left').fillna(0)
b_all = b_all.merge(lone_warrior, on='striker_id', how='left').fillna(0)
b_all = b_all.merge(chase_steep, on='striker_id', how='left').fillna(0)
b_all = b_all.merge(era_recent, on='striker_id', how='left').fillna(0)
b_all = b_all.merge(b_cons, on='striker_id', how='left').fillna(0)
b_all = b_all.merge(roles[['striker_id', 'primary_role', 'mean_pos']], on='striker_id', how='left')

# Rates
b_all['sr_p'] = 100 * b_all['runs_p'] / b_all['balls_p'].clip(1)
b_all['sr_np'] = 100 * b_all['runs_np'] / b_all['balls_np'].clip(1)
b_all['bnd_rate_p'] = 100 * b_all['bnd_p'] / b_all['balls_p'].clip(1)
b_all['bnd_rate_np'] = 100 * b_all['bnd_np'] / b_all['balls_np'].clip(1)
b_all['dot_rate_p'] = 100 * b_all['dots_p'] / b_all['balls_p'].clip(1)
b_all['dis_rate_p'] = 100 * b_all['wkts_p'] / b_all['balls_p'].clip(1)

# The 10 Metrics for Batters:
# 1. WPA per ball in pressure
b_all['M1_wpa_pb'] = b_all['wpa_sum_p'] / b_all['balls_p'].clip(1)
# 2. Contextual RAE per ball
b_all['M2_rae_pb'] = b_all['rae_sum_p'] / b_all['balls_p'].clip(1)
# 3. Lone Warrior run share
b_all['M3_lone_warrior'] = b_all['lone_warrior_share']
# 4. Steep Chase Dominance (WPA per ball in steep chases, with sample shrink)
b_all['M4_chase_steep'] = b_all['wpa_pb_steep']
# 5. Boundary Surge: Pressure Bnd% minus Non-Pressure Bnd%
b_all['M5_bnd_surge'] = b_all['bnd_rate_p'] - b_all['bnd_rate_np']
# 6. Dot Ball Resistance: 100 - Dot% in pressure (ability to keep strike moving)
b_all['M6_dot_resist'] = 100 - b_all['dot_rate_p']
# 7. Wicket Preservation: Expected wicket rate minus actual dismissal rate (higher = preserved wicket better)
b_all['M7_wkt_preserve'] = (b_all['xw_sum_p'] / b_all['balls_p'].clip(1)) * 100 - b_all['dis_rate_p']
# 8. Era-Adjusted Recent RAE per ball (2024-2026 inflation outperformance)
b_all['M8_era_rae'] = b_all['rae_pb_recent']
# 9. Pressure Strike Rate vs IPL Par
b_all['M9_sr_vs_par'] = b_all['sr_p'] - IPL_PAR_SR
# 10. Innings Consistency: positive impact rate
b_all['M10_pos_rate'] = b_all['pos_inn_rate']

# Eligibility: min 50 pressure balls, min 30 in IPL, non-pure bowler
b_elig = b_all[(b_all['balls_p'] >= 50) & (b_all['ipl_balls_p'] >= 30) & (b_all['primary_role'] != 'BOWLER')].copy()

# Standardize each metric
b_metrics = ['M1_wpa_pb', 'M2_rae_pb', 'M3_lone_warrior', 'M4_chase_steep', 'M5_bnd_surge', 
             'M6_dot_resist', 'M7_wkt_preserve', 'M8_era_rae', 'M9_sr_vs_par', 'M10_pos_rate']

# Weights: balanced across high leverage, contextual run creation, lone warrior, chase, and consistency
b_weights = {
    'M1_wpa_pb':       0.20,
    'M2_rae_pb':       0.15,
    'M3_lone_warrior': 0.10,
    'M4_chase_steep':  0.10,
    'M5_bnd_surge':    0.10,
    'M6_dot_resist':   0.08,
    'M7_wkt_preserve': 0.07,
    'M8_era_rae':      0.08,
    'M9_sr_vs_par':    0.07,
    'M10_pos_rate':    0.05
}

b_scaler = StandardScaler()
b_z = pd.DataFrame(b_scaler.fit_transform(b_elig[b_metrics]), columns=[f'{m}_z' for m in b_metrics], index=b_elig.index)
b_elig = pd.concat([b_elig, b_z], axis=1)

b_elig['composite_score'] = sum(b_elig[f'{m}_z'] * w for m, w in b_weights.items())

# Empirical Bayes Shrinkage
kappa_bat = b_elig['balls_p'].median()
b_elig['eb_weight'] = b_elig['balls_p'] / (b_elig['balls_p'] + kappa_bat)
b_elig['shrunk_score'] = b_elig['eb_weight'] * b_elig['composite_score'] + (1 - b_elig['eb_weight']) * b_elig['composite_score'].mean()

# 1000 Bootstrap iterations for Batters
print("Running 1,000 bootstrap iterations for Batters...")
b_boot_ranks = {pid: [] for pid in b_elig['striker_id']}
for _ in range(1000):
    sample = b_elig.sample(frac=1.0, replace=True)
    sample_sorted = sample.sort_values('shrunk_score', ascending=False)
    for rk, row in enumerate(sample_sorted.itertuples(), 1):
        if row.striker_id in b_boot_ranks:
            b_boot_ranks[row.striker_id].append(rk)

b_elig['boot_rank_mean'] = b_elig['striker_id'].map(lambda x: np.mean(b_boot_ranks.get(x, [99])))
b_elig['boot_rank_5th']  = b_elig['striker_id'].map(lambda x: np.percentile(b_boot_ranks.get(x, [99]), 5))
b_elig['boot_rank_95th'] = b_elig['striker_id'].map(lambda x: np.percentile(b_boot_ranks.get(x, [99]), 95))
b_elig['boot_top10_pct'] = b_elig['striker_id'].map(lambda x: 100 * np.mean(np.array(b_boot_ranks.get(x, [99])) <= 10))

b_final = b_elig.sort_values('shrunk_score', ascending=False).reset_index(drop=True)
b_final['rank'] = b_final.index + 1

# ==============================================================================
# PART 2: 10-FACTOR BOWLER RATINGS
# ==============================================================================
print("\n--- Engineering 10 Dimensions for Bowlers ---")

bw_p_agg = legal_p.groupby(['bowler_id', 'bowler_name']).agg(
    balls_p=('runs', 'count'),
    runs_given_p=('total_ball_runs', 'sum'),
    wkts_p=('is_wicket_int', 'sum'),
    dots_p=('is_dot', 'sum'),
    bnd_p=('is_boundary', 'sum'),
    wpa_sum_p=('wpa_bowl_ens', 'sum'),
    wae_sum_p=('wicket_above_expected', 'sum'),
    matches_p=('match_id', 'nunique'),
    ipl_balls_p=('is_ipl', 'sum')
).reset_index()

bw_np_agg = legal_np.groupby(['bowler_id', 'bowler_name']).agg(
    balls_np=('runs', 'count'),
    runs_given_np=('total_ball_runs', 'sum'),
    wkts_np=('is_wicket_int', 'sum'),
    dots_np=('is_dot', 'sum'),
    bnd_np=('is_boundary', 'sum')
).reset_index()

# D3: Lone Enforcer Index (Bowler economy vs rest of own bowling attack in the match)
bowler_match = legal_df.groupby(['match_id', 'innings', 'bowler_id', 'bowler_name']).agg(
    b_runs=('total_ball_runs', 'sum'),
    b_balls=('runs', 'count')
).reset_index()
match_bowl = legal_df.groupby(['match_id', 'innings']).agg(
    m_runs=('total_ball_runs', 'sum'),
    m_balls=('runs', 'count')
).reset_index()
bowler_match = bowler_match.merge(match_bowl, on=['match_id', 'innings'])
bowler_match['rest_runs'] = bowler_match['m_runs'] - bowler_match['b_runs']
bowler_match['rest_balls'] = bowler_match['m_balls'] - bowler_match['b_balls']
bowler_match['b_econ'] = 6 * bowler_match['b_runs'] / bowler_match['b_balls'].clip(1)
bowler_match['rest_econ'] = 6 * bowler_match['rest_runs'] / bowler_match['rest_balls'].clip(1)
bowler_match['econ_vs_rest'] = bowler_match['rest_econ'] - bowler_match['b_econ']
lone_enforcer = bowler_match.groupby('bowler_id')['econ_vs_rest'].mean().reset_index().rename(columns={'econ_vs_rest': 'lone_enforcer_savings'})

# D7: Death Overs Execution (Overs 16-20 under pressure)
death_p = legal_p[legal_p['over_number'] >= 16].groupby('bowler_id').agg(
    balls_death=('runs', 'count'),
    runs_death=('total_ball_runs', 'sum'),
    wkts_death=('is_wicket_int', 'sum'),
    wpa_death=('wpa_bowl_ens', 'sum')
).reset_index()
death_p['econ_death'] = 6 * death_p['runs_death'] / death_p['balls_death'].clip(1)
death_p['wpa_pb_death'] = death_p['wpa_death'] / death_p['balls_death'].clip(1)

# D8: Era-Adjusted Survival (2024-2026 inflation era)
era_bowl = legal_p[legal_p['season_int'] >= 2024].groupby('bowler_id').agg(
    balls_recent=('runs', 'count'),
    runs_recent=('total_ball_runs', 'sum'),
    wpa_recent=('wpa_bowl_ens', 'sum')
).reset_index()
era_bowl['econ_recent'] = 6 * era_bowl['runs_recent'] / era_bowl['balls_recent'].clip(1)
era_bowl['wpa_pb_recent'] = era_bowl['wpa_recent'] / era_bowl['balls_recent'].clip(1)

# D9: Narrow Target Defense (Chase, runs required <= 30 in balls remaining <= 18)
narrow_def = legal_p[(legal_p['is_chase'] == 1) & (legal_p['runs_required'] <= 30) & (legal_p['balls_remaining'] <= 18)].groupby('bowler_id').agg(
    balls_narrow=('runs', 'count'),
    wpa_narrow=('wpa_bowl_ens', 'sum')
).reset_index()
narrow_def['wpa_pb_narrow'] = narrow_def['wpa_narrow'] / narrow_def['balls_narrow'].clip(1)

# D10: Positive Spell Consistency
inn_wpa_bw = legal_p.groupby(['bowler_id', 'match_id'])['wpa_bowl_ens'].sum().reset_index()
bw_cons = inn_wpa_bw.groupby('bowler_id').agg(
    pos_spell_rate=('wpa_bowl_ens', lambda x: (x > 0).mean()),
    wpa_std=('wpa_bowl_ens', 'std')
).reset_index()

# Merge all bowler features
bw_all = bw_p_agg.merge(bw_np_agg, on=['bowler_id', 'bowler_name'], how='left').fillna(0)
bw_all = bw_all.merge(lone_enforcer, on='bowler_id', how='left').fillna(0)
bw_all = bw_all.merge(death_p, on='bowler_id', how='left').fillna(0)
bw_all = bw_all.merge(era_bowl, on='bowler_id', how='left').fillna(0)
bw_all = bw_all.merge(narrow_def, on='bowler_id', how='left').fillna(0)
bw_all = bw_all.merge(bw_cons, on='bowler_id', how='left').fillna(0)
bw_all = bw_all.merge(roles[['bowler_id', 'primary_role']], on='bowler_id', how='left')

# Rates
bw_all['econ_p'] = 6 * bw_all['runs_given_p'] / bw_all['balls_p'].clip(1)
bw_all['econ_np'] = 6 * bw_all['runs_given_np'] / bw_all['balls_np'].clip(1)
bw_all['dot_rate_p'] = 100 * bw_all['dots_p'] / bw_all['balls_p'].clip(1)
bw_all['bnd_rate_p'] = 100 * bw_all['bnd_p'] / bw_all['balls_p'].clip(1)
bw_all['wktr_p'] = 100 * bw_all['wkts_p'] / bw_all['balls_p'].clip(1)

# The 10 Metrics for Bowlers:
# 1. Defensive WPA per ball in pressure
bw_all['M1_wpa_pb'] = bw_all['wpa_sum_p'] / bw_all['balls_p'].clip(1)
# 2. Contextual Pressure Economy Savings vs League Par (10.92 - econ_p)
bw_all['M2_econ_savings_par'] = IPL_PAR_ECON - bw_all['econ_p']
# 3. Lone Enforcer Index (Economy savings vs own teammates in match)
bw_all['M3_lone_enforcer'] = bw_all['lone_enforcer_savings']
# 4. Contextual Wickets Above Expected per ball
bw_all['M4_wae_pb'] = bw_all['wae_sum_p'] / bw_all['balls_p'].clip(1)
# 5. Dot Ball Stranglehold Rate in Pressure (Dot%)
bw_all['M5_dot_rate'] = bw_all['dot_rate_p']
# 6. Boundary Containment Rate (IPL_PAR_BND - bnd_rate_p)
bw_all['M6_bnd_containment'] = IPL_PAR_BND - bw_all['bnd_rate_p']
# 7. Death Overs Dominance (WPA per ball in overs 16-20)
bw_all['M7_death_wpa'] = bw_all['wpa_pb_death']
# 8. Era-Adjusted Survival (Outperformance in 2024-2026 inflation seasons)
bw_all['M8_era_econ_savings'] = 11.20 - bw_all['econ_recent']
# 9. Narrow Target Defense (WPA per ball defending tight totals)
bw_all['M9_narrow_defense'] = bw_all['wpa_pb_narrow']
# 10. Spell Consistency: positive impact rate
bw_all['M10_pos_rate'] = bw_all['pos_spell_rate']

# Eligibility: min 50 pressure balls, min 30 in IPL, non-pure batter
bw_elig = bw_all[(bw_all['balls_p'] >= 50) & (bw_all['ipl_balls_p'] >= 30) & (bw_all['primary_role'] != 'BATTER')].copy()

# Standardize each metric
bw_metrics = ['M1_wpa_pb', 'M2_econ_savings_par', 'M3_lone_enforcer', 'M4_wae_pb', 'M5_dot_rate',
              'M6_bnd_containment', 'M7_death_wpa', 'M8_era_econ_savings', 'M9_narrow_defense', 'M10_pos_rate']

# Weights: high leverage defense, economy savings vs par, lone enforcer, wickets above expected, dot balls, death overs
bw_weights = {
    'M1_wpa_pb':             0.20,
    'M2_econ_savings_par':   0.15,
    'M3_lone_enforcer':      0.12,
    'M4_wae_pb':             0.12,
    'M5_dot_rate':           0.10,
    'M6_bnd_containment':    0.08,
    'M7_death_wpa':          0.08,
    'M8_era_econ_savings':   0.05,
    'M9_narrow_defense':     0.05,
    'M10_pos_rate':          0.05
}

bw_scaler = StandardScaler()
bw_z = pd.DataFrame(bw_scaler.fit_transform(bw_elig[bw_metrics]), columns=[f'{m}_z' for m in bw_metrics], index=bw_elig.index)
bw_elig = pd.concat([bw_elig, bw_z], axis=1)

bw_elig['composite_score'] = sum(bw_elig[f'{m}_z'] * w for m, w in bw_weights.items())

# Empirical Bayes Shrinkage
kappa_bow = bw_elig['balls_p'].median()
bw_elig['eb_weight'] = bw_elig['balls_p'] / (bw_elig['balls_p'] + kappa_bow)
bw_elig['shrunk_score'] = bw_elig['eb_weight'] * bw_elig['composite_score'] + (1 - bw_elig['eb_weight']) * bw_elig['composite_score'].mean()

# 1000 Bootstrap iterations for Bowlers
print("Running 1,000 bootstrap iterations for Bowlers...")
bw_boot_ranks = {pid: [] for pid in bw_elig['bowler_id']}
for _ in range(1000):
    sample = bw_elig.sample(frac=1.0, replace=True)
    sample_sorted = sample.sort_values('shrunk_score', ascending=False)
    for rk, row in enumerate(sample_sorted.itertuples(), 1):
        if row.bowler_id in bw_boot_ranks:
            bw_boot_ranks[row.bowler_id].append(rk)

bw_elig['boot_rank_mean'] = bw_elig['bowler_id'].map(lambda x: np.mean(bw_boot_ranks.get(x, [99])))
bw_elig['boot_rank_5th']  = bw_elig['bowler_id'].map(lambda x: np.percentile(bw_boot_ranks.get(x, [99]), 5))
bw_elig['boot_rank_95th'] = bw_elig['bowler_id'].map(lambda x: np.percentile(bw_boot_ranks.get(x, [99]), 95))
bw_elig['boot_top10_pct'] = bw_elig['bowler_id'].map(lambda x: 100 * np.mean(np.array(bw_boot_ranks.get(x, [99])) <= 10))

bw_final = bw_elig.sort_values('shrunk_score', ascending=False).reset_index(drop=True)
bw_final['rank'] = bw_final.index + 1

# ==============================================================================
# DISPLAY AND EXPORT
# ==============================================================================
print("\n" + "="*80)
print("TOP 15 BATTERS UNDER 10-FACTOR CRICKET PRESSURE MODEL")
print("="*80)
cols_show_b = ['rank', 'striker_name', 'primary_role', 'balls_p', 'ipl_balls_p', 'sr_p', 'M1_wpa_pb', 'M2_rae_pb', 'M3_lone_warrior', 'M10_pos_rate', 'shrunk_score', 'boot_top10_pct']
print(b_final[cols_show_b].head(15).to_string(index=False))

print("\n" + "="*80)
print("TOP 15 BOWLERS UNDER 10-FACTOR CRICKET PRESSURE MODEL")
print("="*80)
cols_show_bw = ['rank', 'bowler_name', 'primary_role', 'balls_p', 'ipl_balls_p', 'econ_p', 'M2_econ_savings_par', 'M3_lone_enforcer', 'M1_wpa_pb', 'M10_pos_rate', 'shrunk_score', 'boot_top10_pct']
print(bw_final[cols_show_bw].head(15).to_string(index=False))

# Save deep ratings
b_final.to_csv('outputs/batter_10factor_ratings.csv', index=False)
bw_final.to_csv('outputs/bowler_10factor_ratings.csv', index=False)

b_final.head(10).to_csv('outputs/top10_batters_10factor.csv', index=False)
bw_final.head(10).to_csv('outputs/top10_bowlers_10factor.csv', index=False)
print("\nSaved 10-factor ratings to outputs successfully.")
