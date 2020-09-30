import requests


def upload_model_to_osf(url, file, name):
    """
    :param url: URL with location where files need to be stored
    :param file: file that needs to be uploaded
    :param name: name of the file to use on OSF
    :return: PUT request that uploads the file to the location provided in the URL
    """
    values = {'name': name, 'kind': 'file'}
    headers = {'content-type': 'applications/json',
               'Authorization': 'Bearer C3JlSWkiMfuE3J03My7vYALl55C5lJo4khzrR9LVyN2K0PHhmCekVU6IIV4EmpJSRS5rYA'}
    r = requests.put(url=url, params=values, data=file, headers=headers)

    return r


def upload_csv_to_osf(url, file_name, name):
    """
    :param url: URL with location where files need to be stored
    :param file_name: name of file that needs to be uploaded
    :param name: name of the file to use on OSF
    :return: PUT request that uploads the file to the location provided in the URL
    """
    values = {'name': name}
    headers = {'content-type': 'multipart/form-data',
               'Authorization': 'Bearer C3JlSWkiMfuE3J03My7vYALl55C5lJo4khzrR9LVyN2K0PHhmCekVU6IIV4EmpJSRS5rYA'}
    r = requests.put(url=url, params=values, data=open(f'{file_name}', 'r', newline='\n'), headers=headers)

    return r


def upload_zip_to_osf(url, file_name, name):
    """
    :param url: URL with location where files need to be stored
    :param file_name: name of file that needs to be uploaded
    :param name: name of the file to use on OSF
    :return: PUT request that uploads the zip file to the location provided in the URL
    """
    values = {'name': name}
    headers = {'content-type': 'multipart/form-data',
               'Authorization': 'Bearer C3JlSWkiMfuE3J03My7vYALl55C5lJo4khzrR9LVyN2K0PHhmCekVU6IIV4EmpJSRS5rYA'}

    with open(file_name, 'rb') as f:
        r = requests.put(url=url, params=values, headers=headers, data=f)

    return r
