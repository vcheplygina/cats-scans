def get_path(home, dataset):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param dataset: dataset name that needs to be collected. Note that this can either be a target or source dataset
    :return: complete path to storage location of target dataset
    """

    if dataset == 'isic':
        img_dir = f'{home}/ISIC2018/ISIC2018_Task3_Training_Input'
        label_dir = f'{home}/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

        return img_dir, label_dir

    if dataset == 'chest':
        data_dir = f'{home}/chest_xray'

        return data_dir

    if dataset == 'stl10':
        data_dir = f'{home}/stl10_binary/'

        return data_dir

    if dataset == 'textures':
        data_dir = f'{home}/dtd/images'

        return data_dir

    if (dataset == 'pcam-small') | (dataset == 'pcam-middle'):
        data_dir = f'{home}/PCam/png_images'

        return data_dir

    if dataset == 'sti10':
        data_dir = f'{home}/sti10'

        return data_dir

    if dataset == 'kimia':
        data_dir = f'{home}/kimia_path_960'

        return data_dir
