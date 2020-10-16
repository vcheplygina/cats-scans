def get_path(home, target_data):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param target_data: dataset used as target dataset
    :return: complete path to storage location of target dataset
    """

    if target_data == 'isic':
        img_dir = f'{home}/ISIC2018/ISIC2018_Task3_Training_Input'
        label_dir = f'{home}/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

        return img_dir, label_dir

    if target_data == 'chest':
        data_dir = f'{home}/chest_xray'

        return data_dir

    if target_data == 'slt10':
        # path to the binary train file with image data
        train_img_path = f'{home}/stl10_binary/train_X.bin'
        # path to the binary train file with labels
        train_label_path = f'{home}/stl10_binary/train_y.bin'
        # path to the binary train file with image data
        test_img_path = f'{home}/stl10_binary/test_X.bin'
        # path to the binary train file with labels
        test_label_path = f'{home}/stl10_binary/test_y.bin'

        return train_img_path, train_label_path, test_img_path, test_label_path

    if target_data == 'textures':
        data_dir = f'{home}/dtd/images'

        return data_dir

    if target_data == 'pcam':
        data_dir = f'{home}/PCam/png_images'

        return data_dir

    if target_data == 'imagenet_subset':
        data_dir = f'{home}/sti10'

        return data_dir
