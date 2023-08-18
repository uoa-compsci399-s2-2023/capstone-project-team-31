from image_helper_functions import *
import numpy as np

## define an experiment object with relevant parameters

'''
Potential starting colours as in glasses paper (r,g,b)
Grey: (127, 127, 127)
Orangish: (220, 130, 0)
Brownish: (160, 105, 55)
Goldish: (200, 175, 30)
Yellowish: (220, 210, 50)
'''


class AdversarialPatternGenerator:

    def __init__(self, accessory_type, images_dir, num_images=1, step_size=20, lambda_tv=3, probability_coeff=5, momentum_coeff=0.4, gauss_filtering=0, max_iter=300, channels_to_fix=[], stop_prob=0.01, horizontal_move=4, vertical_move=4, rotational_move=4, verbose=True):
        self.accessory_type = accessory_type
        self.images_dir = images_dir
        self.num_images = num_images
        self.step_size = step_size
        self.lambda_tv = lambda_tv
        self.probability_coeff = probability_coeff
        self.momentum_coeff = momentum_coeff
        self.gauss_filtering = gauss_filtering
        self.max_iter = max_iter
        self.channels_to_fix = channels_to_fix
        self.stop_prob = stop_prob
        self.movement = dict()
        self.movement['horizontal'] = horizontal_move
        self.movement['vertical'] = vertical_move
        self.movement['rotational'] = rotational_move
        self.verbose = verbose

    def pre_process(self):
        ## pre-process images to standardise face positions
        ## may not need if:
        #### a) the photos used are already standardised, or
        #### b) we can use standardisation from deepface

        ## may need helper function to use GPU or not, such as in line 32 https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/physical_dodging.m

        pass

    def get_best_starting_colour(self):
        '''
        returns configuration where deepface is least confident in its prediction of the true class for each image
        '''

        ## for each colour in potential starting colours:
        #### intialise a 224x224 matrix/image where each cell/pixel is the starting colour
        #### call prepare_images(colour matrix, self.images_dir, self.num_images, self.accessory type) from image_processing.py to get a random selection of images and a mask of the accesssory type in the specified colour
        pass

    def dodge(self, experiment):
        '''
        pertubates the colours within the accessory mask using gradient descent to minimise deepface's confidence in predicting true labels
        '''
        # inspo: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/dodge.m

        ## divide self.step_size and self.lambda.tv by num_images

        pertubations = np.zeros(experiment['num_images']) ## placeholder, where information for each pertubation is stored

        i = 0
        scores = np.array()

        while i < self.max_iter and np.mean(scores) > self.stop_prob:

            #data storing:
            images = []
            movements = []
            areas_to_perturb = []

            for j in range(experiment['num_images']):

                # for every image, move the accessory mask slightly by calling 
                [round_accessory_area, round_accessory_im, movement_info] = move_accessory(experiment['accessory_image'], experiment['accessory_area'], self.movement)
                pertubations[j]['movement_info'] = movement_info

                # define area to perturb using round_accessory_area and don't touch any rgb channel which have been fixed (fixed_rgb_channels)

                # add accessory to image -- their approach was to replace the pixels marked out in round_accessory_area in the image with the coloured pertubation

                # store image -- update data storing arrays images, movements, areas_to_perturb

            # [scores, gradients] = find_gradient(images, true_classes) get the gradient and confidence in true class by running deepface model on images with current pertubation

            for x in range(experiment['num_images']):

                #get the xith image's data from data storage arrays (inl gradients)

                # normalise gradient

                # update the pertubation using total_variation from image_helper_functions

                # compute pertubation and reverse the movement using reverse_accesory_movement(movement_info)

                # apply gaussian filtering per specification given to self.gauss_filtering

                # store the pertubation
                pass

            
            # get printability score using non_printability_score in image_helper_functions.py

            # apply pertubations

            # display
            if self.verbose:
                # print out the iteration number, deepface's current confidence in image true classes, non-printability score
                # display the image with current pertubation
                pass

            i += 1 
            pass

        pass

    # return final pertubation result, with deepface's average confidence in predicting true classes

    def run(self):

        self.pre_process() ## if needed

        starting_point = self.get_gest_starting_colour()

        result = self.dodge(starting_point)

