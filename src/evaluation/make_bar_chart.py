#%% Import package
import matplotlib.pyplot as plt

def make_bars(datalist, target_dataset, auc_mat, stat_mat, exp_mat, name, output_path):
    """Combines the AUC scores and the statistical as well as experts similarity measure and plots them in bar charts."""

    # Select different values depending on the target dataset

    if target_dataset == 'chest_xray':
        target_auc = auc_mat[0][1:]
        target_stat = stat_mat[0][1:]
        target_exp = exp_mat[0][1:]
        data_list = datalist[1:]
    elif target_dataset == 'ISIC2018':
        target_auc = []
        for i in range(len(auc_mat[1])):
            if i != 2:
                target_auc.append(auc_mat[1][i])

        target_stat = []
        for i in range(len(stat_mat[2])):
            if i != 2:
                target_stat.append(stat_mat[2][i])

        target_exp = []
        for i in range(len(exp_mat[2])):
            if i != 2:
                target_exp.append(exp_mat[2][i])

        data_list = []
        for i in range(len(datalist)):
            if i != 2:
                data_list.append(datalist[i])
    elif target_dataset == 'pcam':
        target_auc = auc_mat[2][:-1]
        target_stat = stat_mat[4][:-1]
        target_exp = exp_mat[4][:-1]
        data_list = datalist[:-1]
    else:
        return 'No target dataset'

    # Make bar chart

    values = ['AUC score', 'Statistical similarity', 'Experts similarity']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x=data_list, height=target_auc, width=0.8, align='center', color='gray')
    ax.bar(x=data_list, height=target_stat, width=0.5, align='center', color='blue')
    ax.bar(x=data_list, height=target_exp, width=0.2, align='center', color='orange')

    plt.legend(values, loc=2)

    plt.show()

    figure = ax.get_figure()
    figure.savefig(output_path + '/' + name)

    return
