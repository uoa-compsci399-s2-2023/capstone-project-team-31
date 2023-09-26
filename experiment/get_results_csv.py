import argparse
from results_csv import ResultsCSV


def parse_args():
    parser = argparse.ArgumentParser(description='Run the experiment')

    parser.add_argument("-d", "--images_dir",
                        required = True,
                        type = str,
                        help = "Source directory of the images, either directory or .db file"
    )
    parser.add_argument("-n", "--num_images",
                        default = 1,
                        type = int,
                        help = "How many images are being processed"
    )
    
    
    args = parser.parse_args()

    return args

args = parse_args()

result_csv = ResultsCSV(args.images_dir, args.num_images)

result_csv.run()
