from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


class Irma:
    """This is the IRMA dataset.
    Modified from: https://www.kaggle.com/raddar/irma-xray-dataset

    Deserno, T., & Ott, B. (2009). 15,363 IRMA images of 193 categories for ImageCLEFmed 2009.
    https://doi.org/10.18154/RWTH-2016-06143
    """

    def __init__(self, root, *args, **kwargs):
        self.data_dir = Path(root)

    def load(self):
        train_labels_path = self.data_dir / "ImageCLEFmed2009_train_codes.02.csv"
        train_images_path = self.data_dir / "ImageCLEFmed2009_train.02/ImageCLEFmed2009_train.02"
        test_labels_path = self.data_dir / "ImageCLEFmed2009_test_codes.03.csv"
        test_images_path = self.data_dir / "ImageCLEFmed2009_test.03/ImageCLEFmed2009_test.03"

        train_df = pd.read_csv(train_labels_path, delimiter=";")
        train_df.loc[:, "Path"] = train_df["image_id"].apply(lambda x: self._get_image_path(x, train_images_path))
        train_df.loc[:, "irma_code"] = train_df["irma_code"].apply(lambda x: x.replace("-", ""))
        train_df.loc[:, "Technical Code"] = train_df["irma_code"].apply(self._get_technical_code)
        train_df.loc[:, "Imaging Modality"] = train_df["Technical Code"].apply(self._get_imaging_modality)
        train_df.loc[:, "Directional Code"] = train_df["irma_code"].apply(self._get_directional_code)
        train_df.loc[:, "Imaging Orientation"] = train_df["Directional Code"].apply(self._get_imaging_orientation)
        train_df.loc[:, "Anatomical Code"] = train_df["irma_code"].apply(self._get_anatomical_code)
        train_df.loc[:, "Body Region"] = train_df["Anatomical Code"].apply(self._get_body_region)
        train_df.loc[:, "Central or Extremity"] = train_df["Anatomical Code"].apply(self._get_central_or_extremity)
        train_df.loc[:, "Binary Label"] = np.where(train_df['Central or Extremity'] == 'extremity', 0, 1)

        test_df = pd.read_csv(test_labels_path, delimiter=";")
        test_df.loc[:, "Path"] = test_df["image_id"].apply(lambda x: self._get_image_path(x, test_images_path))
        test_df.loc[:, "irma_code"] = test_df["irma_code"].apply(lambda x: x.replace("-", ""))
        test_df.loc[:, "Technical Code"] = test_df["irma_code"].apply(self._get_technical_code)
        test_df.loc[:, "Imaging Modality"] = test_df["Technical Code"].apply(self._get_imaging_modality)
        test_df.loc[:, "Directional Code"] = test_df["irma_code"].apply(self._get_directional_code)
        test_df.loc[:, "Imaging Orientation"] = test_df["Directional Code"].apply(self._get_imaging_orientation)
        test_df.loc[:, "Anatomical Code"] = test_df["irma_code"].apply(self._get_anatomical_code)
        test_df.loc[:, "Body Region"] = test_df["Anatomical Code"].apply(self._get_body_region)
        test_df.loc[:, "Central or Extremity"] = test_df["Anatomical Code"].apply(self._get_central_or_extremity)
        test_df.loc[:, "Binary Label"] = np.where(test_df['Central or Extremity'] == 'extremity', 0, 1)

        # I didn't like the original ~7:1 split, and am merging the two sets together to make new splits
        merged_df = pd.concat([train_df, test_df])
        self.df = merged_df

    def load_image(self, path: str) -> Image:
        """Cache and load an image."""
        return Image.open(path).convert("RGB")

    def _get_image_path(self, image_id: str, images_path: str) -> str:
        return f"{images_path}/{image_id}.png"

    def _get_technical_code(self, irma_code: str) -> str:
        return irma_code[:4]

    def _get_imaging_modality(self, technical_code: str):
        first, second, third, fourth = technical_code
        first_categories = {"0": "unspecified",
                            "1": "x-ray",
                            "2": "sonography",
                            "3": "magnetic resonance measurements",
                            "4": "nuclear medicine",
                            "5": "optical imaging",
                            "6": "biophysical procedure",
                            "7": "others",
                            "8": "secondary digitalization"}
        if first in first_categories:
            return first_categories[first]
        return technical_code

    def _get_directional_code(self, irma_code: str) -> str:
        return irma_code[4:7]

    def _get_imaging_orientation(self, directional_code: str) -> str:
        first, second, third = directional_code
        result = directional_code
        if first == 0:
            return "unspecified"
        elif first == 1:
            if second == 1:
                return "posteroanterior"
            elif second == 2:
                return "anteroposterior"
        elif first == 2:
            if second == 1:
                return "lateral, right-left"
            elif second == 2:
                return "lateral, left-right"
        return result

    def _get_anatomical_code(self, irma_code: str) -> str:
        return irma_code[7:10]

    def _get_central_or_extremity(self, anatomical_code: str) -> str:
        first, second, third = anatomical_code
        if first in ["4", "9"]:
            return "extremity"
        return "central"

    def _get_body_region(self, anatomical_code: str) -> str:
        first, second, third = anatomical_code
        first_categories = {
            "1": "whole body",
            "2": "cranium",
            "3": "spine",
            "4": "upper extremity/arm",
            "5": "chest",
            "6": "breast",
            "7": "abdomen",
            "8": "pelvis",
            "9": "lower extremity"
        }
        if first == "4":
            upper_categories = {
                "0": "unspecified upper extremity",
                "1": "hand",
                "2": "radio carpal joint",
                "3": "forearm",
                "4": "elbow",
                "5": "upper arm",
                "6": "shoulder"
            }
            return upper_categories[second]
        elif first == "9":
            upper_categories = {
                "0": "unspecified lower extremity",
                "1": "foot",
                "2": "ankle joint",
                "3": "lower leg",
                "4": "knee",
                "5": "upper leg",
                "6": "hip"
            }
            return upper_categories[second]
        return first_categories[first]



