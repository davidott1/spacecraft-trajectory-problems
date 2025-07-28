import matplotlib.pyplot as plt
import numpy as np

# --- Plot 1: Default Matplotlib Style ---
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Default Matplotlib Style')
plt.show()

# --- Plot 2: Using the Seaborn Whitegrid Style ---
# Apply the style
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(x, y)
plt.title('seaborn-v0_8-whitegrid Style')
plt.show()