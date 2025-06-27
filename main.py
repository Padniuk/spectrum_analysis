import argparse
from utils import process_data


def main(folder_path, num_workers=2):
    process_data(folder_path, num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parameters")
    parser.add_argument("-t", "--threads", type=int, help="Number of workers")
    parser.add_argument("-f", "--folder", type=str, help="Folder path")
    args = parser.parse_args()
    main(args.folder, args.threads)
