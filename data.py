import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

class ImageData:
    def __init__(self, language, dpi, style, class_name, path):
        self.language = language
        self.dpi = dpi
        self.style = style
        self.class_name = class_name
        self.path = path

    def to_string(self) -> str:
        return f"('{self.language}', '{self.dpi}', '{self.style}', '{self.class_name}', '{self.path}')"

class DataPartitioner:
    def __init__(self):
        self.data = defaultdict(list)

    def add_data(self, key, data_points: List[ImageData]):
        self.data[key].extend(data_points)

    def save(self, output_file: Path):
        with output_file.open("w") as f:
            for key, points in self.data.items():
                for point in points:
                    f.write(f"{point.to_string()}\n")

def split_data(data: List[ImageData], tr_ratio: float, te_ratio: float, val_ratio: float) -> Tuple[List[ImageData], List[ImageData], List[ImageData]]:
    random.shuffle(data)
    tr_end = int(tr_ratio * len(data))
    te_end = tr_end + int(te_ratio * len(data))
    return data[:tr_end], data[tr_end:te_end], data[te_end:]

def prepare_datasets(args):
    if args.tr_ratio + args.te_ratio + args.val_ratio != 1:
        raise ValueError("Ratios must sum to 1.")

    train_data = DataPartitioner()
    test_data = DataPartitioner()
    val_data = DataPartitioner()
    lang_dpi_mapping = dict(zip(args.lang, args.dpi))
    all_images = []  

    for lang in args.lang:
        dpi = lang_dpi_mapping[lang]  
        for char_dir in (args.input_path / lang).iterdir():
            if char_dir.is_dir():
                class_name = char_dir.name
                for style in args.style:
                    image_paths = list((char_dir / dpi / style).glob("*.bmp"))
                    images = [ImageData(lang, dpi, style, class_name, img_path) for img_path in image_paths]
                    all_images.extend(images)  
    random.shuffle(all_images)
    train, test, val = split_data(all_images, args.tr_ratio, args.te_ratio, args.val_ratio)
    for lang in args.lang:
        dpi = lang_dpi_mapping[lang]
        for style in args.style:
            class_name = f"Class for {lang}" 
            key = (lang, dpi, style, class_name)
            train_data.add_data(key, train)
            test_data.add_data(key, test)
            val_data.add_data(key, val)

    if args.tr_ratio > 0:
        train_data.save(args.output_path / "training_set.txt")
    if args.te_ratio > 0:
        test_data.save(args.output_path / "testing_set.txt")
    if args.val_ratio > 0:
        val_data.save(args.output_path / "validation_set.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command line arguments for the script.")
    parser.add_argument("--lang", nargs='+', required=True, help="Specify the languages. Thai or English or both.")
    parser.add_argument("--dpi", nargs='+', required=True, help="Specify the dpi values (200,300,400). Two values also can be used or None.")
    parser.add_argument("--style", nargs='+', required=True, help="Specify the text styles. Normal, bold, bold_italic, italic")
    parser.add_argument("--tr_ratio", type=float, default=0.8, help="Proportion for the training dataset.")
    parser.add_argument("--te_ratio", type=float, default=0.1, help="Proportion for the test dataset.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion for the validation dataset.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the directory containing image data.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path for output dataset files.")

    arguments = parser.parse_args()
    prepare_datasets(arguments)
