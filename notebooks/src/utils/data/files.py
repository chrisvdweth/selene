import os, re, yaml
import requests
import zipfile, tarfile
import bz2
from tqdm import tqdm


# Load config file
with open("../config.yaml") as f:
    config = yaml.safe_load(f)
    

def create_folder(folder_name, exist_ok=True):
    try:
        os.makedirs(folder_name, exist_ok=exist_ok)
        return folder_name
    except:
        return None


def download_file(file_path, source_folder=None, target_folder=None, overwrite=False):
    # If the source folder is not specified use default (this could be fetch by some request instead of hard-coded)
    if source_folder is None:
        source_folder = config["urls"]["downloads"]["data"]
    # If the target folder is not specified, derive it from the source path
    if target_folder is None:
        target_folder = "/".join(file_path.split("/")[0:-1]) + "/"
    url = source_folder + file_path
    # Preserve file name for download target
    file_name = url.split('/')[-1]
    # Create path if not exists
    create_folder(target_folder)
    # Check if file exists; only overwrite if specified
    if os.path.isfile(file_path) == True and overwrite is not True:
        print(f"File '{file_path}' already exists (use 'overwrite=True' to overwrite it).")
        return file_path, target_folder
    # Streaming, so we can iterate over the response
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    # Initialize progress bar
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    # Fetch file and write to path
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    # Check for errors
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return None, None
    return file_path, target_folder

        

def decompress_file(file_name, target_path='.', overwrite=False):
    if file_name.lower().endswith("zip"):
        zip_file = zipfile.ZipFile(file_name, 'r')
        zip_file.extractall(target_path)
        zip_file.close()
        return None
    elif file_name.lower().endswith("tar.gz"):
        tar = tarfile.open(file_name, "r:gz")
        tar.extractall(path=target_path)
        tar.close()
        return None
    elif file_name.lower().endswith("tar"):
        tar = tarfile.open(file_name, "r:")
        tar.extractall(path=target_path)
        tar.close()
        return None
    elif file_name.lower().endswith("bz2"):
        output_file_name = target_path + file_name.split('/')[-1]
        output_file_name = re.sub('.bz2', '', output_file_name, flags=re.I)
        # Check if file exists; only overwrite if specified
        if os.path.isfile(output_file_name) == True and overwrite is not True:
            print('File "{}" already exists.'.format(output_file_name))
            return output_file_name
        with open(output_file_name, 'wb') as output_file, bz2.BZ2File(file_name, 'rb') as file:
            for data in iter(lambda : file.read(100 * 1024), b''):
                output_file.write(data)
        return output_file_name
        
        
        