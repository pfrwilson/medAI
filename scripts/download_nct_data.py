"""
Downloads the NCT data from image.cs.queensu.ca and preprocesses it to RF and B-mode images.
"""

import argparse
import paramiko
import os
import pandas as pd
from threading import Thread
from queue import Queue
from medAI.utils.data.nct_preprocessing import iq_to_rf, stitch_focal_zones, to_bmode
from skimage.transform import resize
import numpy as np


IMAGE_SERVER_MATFILES_DIR = "/med-i_data/Data/Exact_Ultrasound/data/full_data"
IMAGE_SERVER_CSV_PATH = "/med-i_data/exact_prostate_segemnts/metadata.csv"

BMODE_SIZE = 833, 1372



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", type=str, required=True, help='Username to the image.cs.queensu.ca server')
    parser.add_argument("--password", type=str, required=True, help='Password to the image.cs.queensu.ca server')
    parser.add_argument("--target_dir", type=str, required=True, help='Target directory to download the data')

    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    q = Queue()
    data_downloader = Thread(target=download_data, args=(args.username, args.password, args.target_dir, q))
    data_downloader.start()

    preprocess_data(args.target_dir, q)


def download_data(username, password, target_dir, queue):
    print("Connecting to image.cs.queensu.ca")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('image.cs.queensu.ca', username=username, password=password)
    sftp = ssh.open_sftp()

    print("Downloading metadata.csv")
    if not 'metadata.csv' in os.listdir(target_dir):
        sftp.get(IMAGE_SERVER_CSV_PATH, os.path.join(target_dir, 'metadata.csv'))
        metadata = pd.read_csv(os.path.join(target_dir, 'metadata.csv'), index_col=0)
        metadata = metadata.drop(columns=['patient_id'])
        metadata.rename(columns={'core_specifier': 'core_id', 'patient_specifier': 'patient_id'}, inplace=True)
        metadata.to_csv(os.path.join(target_dir, 'metadata.csv'))
    metadata = pd.read_csv(os.path.join(target_dir, 'metadata.csv'), index_col=0)

    data_dir = os.path.join(target_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)
    
    for path in metadata['path_on_server'].to_list(): 
        remote_path = path 
        local_path = os.path.join(data_dir, os.path.basename(path))
        if os.path.exists(
            os.path.join(target_dir, 'rf', os.path.basename(path).replace('.mat', '.npy'))
        ): 
            continue    
        sftp.get(remote_path, local_path)
        queue.put(local_path)

    queue.put(None)


def preprocess_data(target_dir, queue: Queue): 

    rf_dir = os.path.join(target_dir, 'rf')
    os.makedirs(rf_dir, exist_ok=True)
    bmode_dir = os.path.join(target_dir, 'bmode')
    os.makedirs(bmode_dir, exist_ok=True)

    while True: 
        local_path = queue.get()
        if local_path is None: 
            break

        print(f"Processing {local_path}")

        # IQ to rf frames
        iq = loadmat(local_path)
        rf = iq_to_rf(iq['Q'], iq['I'])

        frames = []
        bmode_frames = []
        for i in range(rf.shape[-1]): 
            frame = rf[..., i]

            if frame.shape[1] > 512:
                frame = stitch_focal_zones(frame)

            from scipy.signal import decimate
            frame = decimate(frame, 4, axis=0)
            frames.append(frame)

            bmode_frame = to_bmode(frame)
            bmode_frame = resize(bmode_frame, BMODE_SIZE, anti_aliasing=True)
            bmode_frame = bmode_frame - bmode_frame.min()
            bmode_frame = bmode_frame / bmode_frame.max()
            bmode_frame = bmode_frame * 255
            bmode_frame = bmode_frame.astype(np.uint8)

            bmode_frames.append(bmode_frame)

        frames = np.stack(frames, axis=-1)  
        frames = frames.astype(np.float16)
        bmode_frames = np.stack(bmode_frames, axis=-1)

        np.save(os.path.join(rf_dir, os.path.basename(local_path)).replace('.mat', '.npy'), frames)
        np.save(os.path.join(bmode_dir, os.path.basename(local_path)).replace('.mat', '.npy'), bmode_frames)
        os.remove(local_path)

        print(f"Processed {len(os.listdir(rf_dir))} files")

        queue.task_done()


def loadmat(path): 
    import scipy.io as sio
    import mat73
    try: 
        return sio.loadmat(path)
    except NotImplementedError: 
        return mat73.loadmat(path)



if __name__ == '__main__':
    main()