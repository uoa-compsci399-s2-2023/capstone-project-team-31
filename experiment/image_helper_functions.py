def pre_process_imgs():
    # if needed

    pass


def prepare_images(colour_matrix, images_dir, num_images, accessory_type):
    '''
    Returns an object which specifies:
     * which images will be used for generating an adversarial pattern (could be all in the images directory)
     * the silouhette mask for the chosen accessory, in the given colour
    '''
    # inspiration: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/prepare_experiment.m

    if accessory_type == "glasses":
        # load glasses_silhouette, find what pixels are white (i.e. colour value not rgb (0,0,0)) and make a colour mask of the chosen colour

        pass

    # take a random permutation of image filenames from images_dir of size num_images
    # for each iamge filename, load the image and store it in some type of structure (e.g. list/json whatever)
    
    pass

def move_accessory(accessory_image, accessory_area, movement):
    '''
    Returns a list of [round_accessory_area, round_accessory_im, movement_info]
    '''
    # inspo: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/rand_movement.m

    # generate random values for horizontal, vertical and rotational shifts within the ranges given in 'movement' dict

    # shift the pixel values in accessory_area acording to those generated values

    # keep a record of what movements were made in movement_info
    
    pass

def reverse_accessory_move(movement_info):
    '''
    Reinstates original accessory position
    '''
    pass

def total_variation(image, beta):
    # take from source: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/total_variation.m

    pass

def non_printability_score(image, segmentation, printable_values):
    '''
    Evaluates the ability of a printer to match the colours in the pertubation
    '''

    # copy directly from: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/non_printability_score.m

    pass

def get_printable_vals():
    '''
    Sources what values are printable by comparing (?) to ./assets/printed-palette.png
    '''
    # inspo1: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/get_printable_vals.m
    # inspo2: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/make_printable_vals_struct.m