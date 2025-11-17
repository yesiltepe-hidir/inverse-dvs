import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
# Set CUDA visible devices to 1 to avoid out of memory errors
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise = torch.load("noise.pt").to(device)
original_samples = torch.load("original_samples.pt").to(device)
sqrt_alpha_prod = torch.load("sqrt_alpha_prod.pt").to(device)
sqrt_one_minus_alpha_prod = torch.load("sqrt_one_minus_alpha_prod.pt").to(device)
x_mean = torch.load("original_samples_mean.pt")
x_std = torch.load("original_samples_std.pt")

random_noise = torch.randn_like(original_samples)
expected_l2_norm = torch.norm(random_noise, p=2)

n_samples = 50
random_noise = torch.randn_like(original_samples)

def k_coef(a, b, c):
    m = torch.zeros_like(a).to(a.device, dtype=a.dtype)
    for i in range(c):
        m += a * b ** i
    n = b ** c
    return m, n
n_dim = 1
font_size = 20

coef_list = [k_coef(sqrt_alpha_prod.clone(), sqrt_one_minus_alpha_prod.clone(), i) for i in range(n_samples)]
sqrt_alpha_prod_list = [coef_list[i][0] for i in range(n_samples)]
sqrt_one_minus_alpha_prod_list = [coef_list[i][1] for i in range(n_samples)]
k_rnr = [(sqrt_alpha_prod_list[i] * original_samples + sqrt_one_minus_alpha_prod_list[i] * noise) for i in range(n_samples)]
expected = [sqrt_alpha_prod_list[0] * original_samples + sqrt_one_minus_alpha_prod_list[0] * random_noise] * n_samples
expected_mean = [sqrt_alpha_prod[0] * x_mean + sqrt_one_minus_alpha_prod_list[0] * random_noise] * n_samples
expected_std =  [((sqrt_alpha_prod * x_std)**2 + sqrt_one_minus_alpha_prod**2)**0.5] * n_samples


def mean_std_k(k, x_mean, x_std, coefs, eps):
    coef_x0, coef_eps = coefs[k]
    mean = coef_x0 * x_mean + coef_eps * eps
    std = ((coef_x0 * x_std)**2 + (coef_eps)**2)**0.5
    return mean, std

n_samples = 1000

coef_list = [k_coef(sqrt_alpha_prod.clone(), sqrt_one_minus_alpha_prod.clone(), i) for i in range(n_samples)]
mean_std_list = [mean_std_k(i, x_mean, x_std, coef_list, random_noise) for i in range(n_samples)]
mean_list = [mean_std_list[i][0] for i in range(n_samples)]
std_list = [mean_std_list[i][1] for i in range(n_samples)]

# Create a figure for the mean values
every_n_sample = 15
plt.figure(figsize=(9, 6))

# Plot the mean values - using L2 norm instead of direct values
colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
expected_mean_norm  = [torch.norm(sample.float(), p=2).cpu().numpy() / n_dim for sample in expected_mean]

# Calculate L2 norms of mean values
mean_l2_norms = [torch.norm(mean, p=2).item() / n_dim for mean in mean_list]

# Make sure x_values start from 1
x_values = list(range(1, n_samples + 1))

# Create indices for every 50th element
scatter_indices = list(range(0, n_samples, every_n_sample))

# Plot only the selected points and lines between them (single call)
plt.plot([x_values[i] for i in scatter_indices], 
         [mean_l2_norms[i] for i in scatter_indices], 
         '-', linewidth=2, color='black')

# Add scatter plot for the same points
plt.scatter([x_values[i] for i in scatter_indices], 
            [mean_l2_norms[i] for i in scatter_indices], 
            c=[colors[i] for i in scatter_indices], 
            s=50, zorder=5, label='k-RNR Mean', edgecolors='black', linewidths=1)

plt.plot([x_values[i] for i in scatter_indices] , [expected_mean_norm[0]] * len(scatter_indices) , '--', color='black', label='Expected', linewidth=2)
# Set labels and title
plt.xlabel('k order', fontsize=font_size)
plt.ylabel('L2 Norm', fontsize=font_size)
plt.legend(frameon=True, edgecolor='black', fontsize=font_size, loc='upper left')  # Moved legend to upper left
# plt.title('Mean Values (L2 Norm) for Different k Orders', fontsize=16)

# Ensure x-axis includes all points from 1 to n_samples
plt.xlim(1, n_samples)
# Set x-ticks
x_ticks = [1] + [i for i in range(100, n_samples, 100)]
plt.xticks(x_ticks)

# Set all spines visible to create a rectangular box around the plot
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_edgecolor('black')
ax.spines['right'].set_edgecolor('black')
ax.spines['left'].set_edgecolor('black')
ax.spines['bottom'].set_edgecolor('black')
# Make all spines thicker
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Add grid
plt.grid(True, linestyle='--', alpha=0.2)
plt.savefig('k_rnr_mean_values_l2norm.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig('k_rnr_mean_values_l2norm.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()