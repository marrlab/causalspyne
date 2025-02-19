import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from causalspyne.implicit_gen_Sigma import gen_spectrum


# Generate sample data

# samples = np.random.normal(size=10000)
samples = [gen_spectrum() for _ in range(10000)]
# Create histogram
fig, ax = plt.subplots(figsize=(8, 6))

hist, bins, _ = ax.hist(samples, bins=30, density=True, alpha=0.7,
                        label="Histogram")

# Calculate bin centers for PDF plotting
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Compute PDF
pdf = stats.norm.pdf(bin_centers)

# Plot PDF
ax.plot(bin_centers, pdf, 'r-', label="PDF")

# Customize plot
ax.set_xlabel('eigenvalue')
ax.set_ylabel('Density')
ax.set_title('Histogram of maximum eigenvalue distribution')
ax.legend()

plt.show()
