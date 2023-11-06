from pathlib import Path
import pandas as pd
from skimage import io as skio
import zipfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_train_val_test(df, seed:int=42, test_size:int=0.2, validation_size:int=0.25):
    df["split"] = None
    df_tracks = df.groupby(by="track_id").first().reset_index()
    trainval, test = train_test_split(df_tracks, stratify=df_tracks.label_id,random_state=seed,test_size=test_size)
    train, validation = train_test_split(trainval, stratify=trainval.label_id,random_state=seed,test_size=validation_size)
    for idx, val in train.iterrows():
        df.loc[df.track_id == val.track_id, "split"] = "train"
    for idx, val in validation.iterrows():
        df.loc[df.track_id == val.track_id, "split"] = "validation"
    for idx, val in test.iterrows():
        df.loc[df.track_id == val.track_id, "split"] = "test"
    return

def get_tracks_per_species(df):
    df3 = df.groupby(by=["track_id", 'label_id']).count().reset_index()
    df4 = df3.groupby(by=['label_id']).count().reset_index()
    return df4[["label_id", "track_id"]]

def remove_unknown_weeds(df):
    df = df.loc[df.label_id!="Weed"]
    return df

def remove_specific_classes(df, classes: list):
    for cls in classes:
        df = df.loc[df.label_id!=cls]
    return df


def merge_varieties(df):
    df.loc[df.label_id=="SORFR", "label_id"] = "SORVU"
    df.loc[df.label_id=="SORHA", "label_id"] = "SORVU"
    df.loc[df.label_id=="SORKM", "label_id"] = "SORVU"
    df.loc[df.label_id=="SORKS", "label_id"] = "SORVU"
    df.loc[df.label_id=="SORRS", "label_id"] = "SORVU"
    df.loc[df.label_id=="SORSA", "label_id"] = "SORVU"
    df.loc[df.label_id=="ZEAKJ", "label_id"] = "ZEAMX"
    df.loc[df.label_id=="ZEALP", "label_id"] = "ZEAMX"
    return    


csv_path = Path("/local2/MFWD/gt.csv")
zip_path = Path("/local2/MFWD/jpegs")
save_path = Path("/local2/MFWD/patches")
save_path.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv_path)

merge_varieties(df) 
df=remove_unknown_weeds(df)
df=remove_specific_classes(df, ["VICVI", "POLAV"])
split_train_val_test(df)
tqdm_l = tqdm(sorted(zip_path.glob("*/*.zip")), total=len(sorted(zip_path.glob("*/*.zip"))))
for zip_file in tqdm_l:
    tqdm_l.set_description_str(f"{zip_file}")
    with zipfile.ZipFile(zip_file, mode="r") as archive:
        for file in sorted(archive.namelist())[1:]:
            print(f"working on file: {file}")
            img = skio.imread(archive.open(file))
            df2 = df.loc[df.filename.str.contains(Path(file).stem)]
            if len(df2) >0:  # if there are any annotations in the image
                for idx, val in df2.iterrows():
                    if val.label_id != "Weed":
                        patch = img[val.ymin:val.ymax,val.xmin:val.xmax,:]
                        save_folder_path= save_path /val.split / val.label_id
                        save_folder_path.mkdir(parents=True, exist_ok=True)
                        skio.imsave(f"{save_folder_path}/{val.tray_id}_{val.bbox_id}.jpeg",patch, check_contrast=False)
                        
print("finished!")
