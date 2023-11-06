from ftplib import FTP
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm

def get_list_to_download(df, lst, column):
    df2 = df[df[column].isin(lst)]
    df3 = df2.groupby(["label_id", "tray_id"]).first().reset_index()
    df4 = df3["filename"].str.split("/", expand=True)
    to_download = list(df4[0] + "/" + df4[1] + ".zip")
    return to_download


def download_file(save_path, folder, fname):
    ftp = FTP('dataserv.ub.tum.de')
    ftp.login('m1717366', 'm1717366')
    save_file_path = save_path / folder / fname
    save_file_path.parent.mkdir(exist_ok=True, parents=True)
    if not save_file_path.is_file():
        with open(save_file_path, 'wb') as f:
            ftp.retrbinary('RETR ' + folder + "/" + fname, f.write)
    return


def download_gt_file(save_path):
    """
    Download gt.csv file if it does not exist
    :param save_path: path to save this file to
    :return:
    """
    ftp = FTP('dataserv.ub.tum.de')
    ftp.login('m1717366', 'm1717366')
    save_file_path = save_path / "gt.csv"
    if not list(save_path.glob("gt.csv")):
        print("Downloading gt.csv...")
        with open(save_file_path, 'wb') as f:
            ftp.retrbinary('RETR ' + "gt.csv", f.write)
    return


def download_all_files_with_segmentation_masks(save_path, image_type, species=None):
    print("Downloading all files with segmentation masks...")
    download_gt_file(save_path)
    df = pd.read_csv(save_path / "gt.csv")
    tray_ids = [132801, 103814, 136813, 104806, 109811, 108807, 107907, 131803, 114905, 116814, 124832, 118934, 120902,
                139837]
    filenames = get_list_to_download(df, tray_ids, "tray_id")
    folders = [image_type, "masks/panoptic_segmentation", "masks/semantic_segmentation"]
    for folder in folders:
        trange = tqdm(filenames, total=len(filenames))
        for fname in trange:
            download_file(save_path, folder, fname)
            trange.set_description_str(folder+"/"+fname)
    return


def download_species(save_path, image_type, species):
    download_gt_file(save_path)
    print(f"Downloading {species}...")
    df = pd.read_csv(save_path / "gt.csv")
    filenames = get_list_to_download(df, species, "label_id")
    trange = tqdm(filenames, total=len(filenames))
    for fname in trange:
            download_file(save_path, image_type, fname)
            trange.set_description_str(image_type+"/"+fname)
    return

def download_trays(save_path, image_type, trays):
    download_gt_file(save_path)
    print(f"Downloading {trays}...")
    df = pd.read_csv(save_path / "gt.csv")
    trays = [int(tray) for tray in trays]  # cast to int
    filenames = get_list_to_download(df, trays, "tray_id")
    trange = tqdm(filenames, total=len(filenames))
    for fname in trange:
            download_file(save_path, image_type, fname)
            trange.set_description_str(image_type+"/"+fname)
    return


FUNCTION_MAP = {'species': download_species,
                'masks': download_all_files_with_segmentation_masks,
                'trays': download_trays}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('download', choices=FUNCTION_MAP.keys())
    parser.add_argument('-save_path', type=str, default='/local2')
    parser.add_argument('-files', type=str, default='ARTVU,CHEAL', help="Comma separated list of EPPO codes or trays IDS to download")
    parser.add_argument('-img_type', type=str, default='jpegs', help="Type of the images: 'jpegs' or 'pngs'")
    args = parser.parse_args()
    files = args.files.split(",")
    files = [spec.strip() for spec in files]
    func = FUNCTION_MAP[args.download]
    func(Path(args.save_path), args.img_type, files)
