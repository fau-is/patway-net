import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Ratio of training set" : [.0, .2, .4, .6, .8, 1],
    "Avg" : [0.532047788, 0.612610353, 0.634283209, 0.663306835, 0.707324699, 0.734117295]
})

plt.plot(df["Ratio of training set"], df["Avg"], marker = "o", linestyle="--")

x_labels = ["10%", "20%", "40%", "60%", "80%", "100%"]
plt.xticks(df["Ratio of training set"], x_labels)

y = [.5, .55, .6, .65, .7, .75]
plt.yticks(y)

plt.xlabel("Training sample size")
plt.ylabel("Test $AUC_{ROC}$")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'../plots/sepsis/sample_sizes.pdf', dpi=100, bbox_inches="tight")
plt.close(fig1)
