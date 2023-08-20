import argparse
from adversarial_pattern_generator import AdversarialPatternGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Run the experiment')

    parser.add_argument("-a", "--accessory_type",
                        required = True,
                        type = str,
                        help = "Which accessory is being used in the used data"
    )
    parser.add_argument("-c", "--classification",
                        required = True,
                        choices = ["age", "emotion", "ethnicity", "gender"],
                        type = str,
                        help = "Which classification of the model is being dodged"
    )
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
    parser.add_argument("-s", "--step_size",
                        default = 20,
                        type = int,
                        help = "The step size used in the optimisation algorithm"
    )
    parser.add_argument("-l", "--lambda_tv",
                        default = 3,
                        type = float,
                        help = "Size of changes in total variation"
    )
    parser.add_argument("-p", "--printability_coeff",
                        default = 5,
                        type = float,
                        help = "" # TODO: Where is this used and for what
    )
    parser.add_argument("-m", "--momentum_coeff",
                        default = 0.4,
                        type = float,
                        help = "" # TODO: Where is this used and for what
    )
    parser.add_argument("-g", "--gauss_filtering",
                        default = 0,
                        type = float,
                        help = "" # TODO: Where is this used and for what
    )
    parser.add_argument("-i", "--max_iterations",
                        default = 15,
                        type = int,
                        help = "Maximum number of iterations during perturbtion" 
    )
    parser.add_argument("-f", "--channels_to_fix",
                        default = [],
                        type = list,
                        help = "Which RGB channels to fix"
    )
    parser.add_argument("-P", "--stop_probability",
                        default = 0.01,
                        type = float,
                        help = "The probability at which to stop iterating (if hit before max_iter)"
    )
    parser.add_argument("-H", "--horizontal_move",
                        default = 4,
                        type = int,
                        help = "Horizontal move distance"
    )
    parser.add_argument("-V", "--vertical_move",
                        default = 4,
                        type = int,
                        help = "Vertical movement distance")
    parser.add_argument("-R", "--rotational_move",
                        default = 4,
                        type = int,
                        help = "Rotational movement distance")
    parser.add_argument("-v", "--v",
                        default = True,
                        type = bool,
                        help = "Verbosity of experiment"
    )

    args = parser.parse_args()

    return args

args = parse_args()
adv_pattern_generator = AdversarialPatternGenerator(args.accessory_type, args.classification, args.images_dir) # and can specify any other paramenters from args

adv_pattern_generator.run()