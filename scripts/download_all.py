import gdown


SMPL_MODEL_PATH = 'data/smpl_models/smpl'
DEMO_DATA_PATH = 'data/trials/demo'

try:
    # smpl model
    gdown.download_folder(url='https://drive.google.com/drive/u/0/folders/1gFdZC4quxsAzqGWR-aY7NFEWC_FeFHVy', quiet=False, output=SMPL_MODEL_PATH)
    # demo data
    gdown.download_folder(url='https://drive.google.com/drive/u/0/folders/1K9HQAC_YDNTNrYny4mlvi6SxBRWSl-JY', quiet=False, output=DEMO_DATA_PATH)
except IOError:
    print("There has been an error trying to install the SMPL Models")

