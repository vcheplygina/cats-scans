import requests
from .access_keys import osf_key


def upload_zip_to_osf(url, file_name, name):
    """
    :param url: URL with location where zip needs to be uploaded
    :param file_name: name of file that needs to be uploaded
    :param name: desired filename to be used on OSF
    :param key: access/authorization key for uploads to OSF
    :return: PUT request that uploads the zip file to the location provided in the URL
    """
    values = {'name': name}
    headers = {'content-type': 'multipart/form-data',
               'Authorization': osf_key}

    with open(file_name, 'rb') as f:
        r = requests.put(url=url, params=values, headers=headers, data=f)

    return r
