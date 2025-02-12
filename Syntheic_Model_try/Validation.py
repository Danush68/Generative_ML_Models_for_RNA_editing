from scipy.stats import spearmanr
from Training_loop import model
# Compute Spearman's correlation
pred_efficiency = model(gRNA_test, target_test)[:, 0].detach().numpy()
true_efficiency = df_test['editing_efficiency'].values
rho, _ = spearmanr(pred_efficiency, true_efficiency)
print(f"Spearman's rho: {rho:.2f}")