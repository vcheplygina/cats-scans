import requests


def upload_to_osf(url, file, name):
    """
    :param url: URL with location where files need to be stored
    :param file: file to store
    :param name: name of the file
    :return: PUT request that uploads the file to the location provided in the URL
    """
    values = {'name': name, 'kind': 'file'}
    headers = {'content-type': 'applications/json',
               'Authorization': 'Bearer C3JlSWkiMfuE3J03My7vYALl55C5lJo4khzrR9LVyN2K0PHhmCekVU6IIV4EmpJSRS5rYA'}
    r = requests.put(url=url, params=values, data=file, headers=headers)

    return r
