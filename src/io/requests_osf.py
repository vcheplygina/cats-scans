import requests


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
