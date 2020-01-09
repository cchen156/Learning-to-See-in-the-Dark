import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)



print('Dowloading Sony subset... (25GB)')
download_file_from_google_drive('1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx', 'dataset/Sony.zip')

print('Dowloading Fuji subset... (52GB)')
download_file_from_google_drive('1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH', 'dataset/Fuji.zip')

os.system('unzip dataset/Sony.zip -d dataset')
os.system('unzip dataset/Fuji.zip -d dataset')
