import matplotlib.pyplot as plt
import pandas as pd

# In this file a plot showing some instability during training and validation is created. Two models and their training
# curves are compared in a lineplot.

# load training and validation accuracies of desired models
train_acc = pd.read_csv('/Users/IrmavandenBrandt/Downloads/Internship/trainacc_pcam-small_isic_lr=0.000001.csv',
                        names=['Epochs', 'Accuracy'], header=None)
val_acc = pd.read_csv('/Users/IrmavandenBrandt/Downloads/Internship/valacc_pcam-small_isic_lr=0.000001.csv',
                      names=['Epochs', 'Accuracy'], header=None)
train_acc_high = pd.read_csv('/Users/IrmavandenBrandt/Downloads/Internship/trainacc_pcam-small_isic_lr=0.00001.csv',
                             names=['Epochs', 'Accuracy'], header=None)
val_acc_high = pd.read_csv('/Users/IrmavandenBrandt/Downloads/Internship/valacc_pcam-small_isic_lr=0.00001.csv',
                           names=['Epochs', 'Accuracy'], header=None)

# create figure by creating two subplots that are located below each other
fig, (ax1, ax2) = plt.subplots(2, 1)

# add lines where accuracies are shown on the y-axis and epochs on the x-axis and label them
ax1.plot(train_acc['Epochs'], train_acc['Accuracy'], label='Training')
ax1.plot(val_acc['Epochs'], val_acc['Accuracy'], label='Validation')
ax2.plot(train_acc_high['Epochs'], train_acc_high['Accuracy'], label='Training')
ax2.plot(val_acc_high['Epochs'], val_acc_high['Accuracy'], label='Validation')

# set axis labels, the x-axis label is shared
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Accuracy')
plt.xlabel('Epochs (5 * 50 epochs)')

# set titles
ax1.set_title("Accuracy scores using learning rate of 1.0e-6")
ax2.set_title("Accuracy scores using learning rate of 1.0e-5")

# add legends
ax1.legend(loc='lower right', prop={'size': 8})
ax2.legend(loc='lower right', prop={'size': 8})

# create tight layout so that sides are not cut off
plt.tight_layout()

# save plot with extra high dpi to avoid low resolution PNG
plt.savefig('outputs/stability_issues', dpi=1000)
plt.show()
