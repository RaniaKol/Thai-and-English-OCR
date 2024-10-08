from collections import defaultdict
from pathlib import Path
import random
import argparse
from typing import List, Tuple

class Datasplit:
    def __init__(self):
        self.data = defaultdict(list)

    def add_data(self, key, data_points: List[Tuple[str, str, str, str, Path]]):
        self.data[key].extend(data_points)

    def save(self, output_file: Path):
        with output_file.open("w") as f:
            for points in self.data.values():
                for point in points:
                    f.write(f"('{point[0]}', '{point[1]}', '{point[2]}', '{point[3]}', '{point[4]}')\n")

def split_data(data: List[Tuple[str, str, str, str, Path]], tr_ratio: float, te_ratio: float, val_ratio: float) -> Tuple[List, List, List]:
    tr_end = int(tr_ratio * len(data))
    te_end = tr_end + int(te_ratio * len(data))
    return data[:tr_end], data[tr_end:te_end], data[te_end:]

def prepare_datasets(args):
    train_data, test_data, val_data = Datasplit(), Datasplit(), Datasplit()
    all_images = []

    for lang, dpi in zip(args.lang, args.dpi):
        for char_dir in (args.input_path / lang).iterdir():
            if char_dir.is_dir():
                for style in args.style:
                    images = [(lang, dpi, style, char_dir.name, img_path) 
                              for img_path in (char_dir / dpi / style).glob("*.bmp")]
                    all_images.extend(images)

    random.shuffle(all_images)
    train, test, val = split_data(all_images, args.tr_ratio, args.te_ratio, args.val_ratio)

    # Using a common key to avoid redundant loop
    for data_split, data_points in zip((train_data, test_data, val_data), (train, test, val)):
        data_split.add_data("default", data_points)

    if args.tr_ratio > 0:
        train_data.save(args.output_path / "train_set.txt")
    if args.te_ratio > 0:
        test_data.save(args.output_path / "test_set.txt")
    if args.val_ratio > 0:
        val_data.save(args.output_path / "val_set.txt")

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
