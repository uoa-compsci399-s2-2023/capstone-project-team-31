from image_helper_functions import *
import numpy as np
from deepface_functions import *
from deepface_models import *
from PIL import Image

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

    def __init__(self, accessory_type, classification, images_dir, num_images=1, step_size=20, lambda_tv=3, printability_coeff=5, momentum_coeff=0.4, gauss_filtering=0, max_iter=15, channels_to_fix=[], stop_prob=0.01, horizontal_move=4, vertical_move=4, rotational_move=4, verbose=True):
        self.accessory_type = accessory_type
        self.classification = classification # what type of classification is being dodged - 'gender', 'age', 'ethnicity', 'emotion' (to do: emotion requires further preprocessing)
        if classification == 'ethnicity':
            self.class_num = 1
        elif classification == 'gender':
            self.class_num = 2
        elif classification == 'age':
            self.class_num = 3
        elif classification == 'emotion':
            self.class_num = 4
        else:
            print("ERROR: {} is an invalid classification. Check spelling is one of: 'ethnicity', 'gender', 'age', 'emotion".format(classification))
        self.images_dir = images_dir
        self.num_images = num_images
        self.step_size = step_size
        self.lambda_tv = lambda_tv
        self.printability_coeff = printability_coeff
        self.momentum_coeff = momentum_coeff
        self.gauss_filtering = gauss_filtering
        self.max_iter = max_iter
        self.channels_to_fix = channels_to_fix
        self.stop_prob = stop_prob
        self.movement = dict()
        self.movement['horizontal'] = horizontal_move
        self.movement['vertical'] = vertical_move
        self.movement['rotational'] = rotational_move
        self.colours = ['red', 'green', 'blue', 'yellow']
        self.verbose = verbose
        
        self.model = attributeModel(self.classification)

    # def pre_process(self, images):
        
    #     processed_imgs = []
        
    #     for i in range(len(images)):
            
    #         temp = convert_b64_to_np(images[i][0]) 

    #         contents = getImageContents(temp)
            
    #         detected_aligned = contents[0][0]
    #         detected_aligned = np.multiply(detected_aligned, 255).astype(np.uint8)

    #         processed_imgs.append(detected_aligned)

    #     ## may need helper function to use GPU or not, such as in line 32 https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/physical_dodging.m

    #     return processed_imgs

    def get_best_starting_colour(self):
        '''
        returns starting configuration where deepface is least confident in its prediction of the true class for each image
        '''

        processed_imgs = prepare_processed_images(self.images_dir, self.num_images)

        #processed_imgs = self.pre_process(images)
        
        best_start = []
        min_avg_true_class_conf = 1
        
        for colour in self.colours:
            
            accessory_img, accessory_mask = prepare_accessory(colour, "./assets/{}_silhouette.png".format(self.accessory_type), self.accessory_type)
            
            confidences = np.empty(len(processed_imgs))
            
            for i in range(len(processed_imgs)):
            
                temp_attack = apply_accessory(processed_imgs[i][0], accessory_img, accessory_mask)

                temp_attack = temp_attack.astype(np.uint8)
                
                
                cv2.imshow('image window', temp_attack)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                confidences[i] = get_confidence_in_true_class(temp_attack, self.classification, processed_imgs[i][self.class_num], self.model)
                
            avg_true_class_conf = np.mean(confidences)
            
            if avg_true_class_conf < min_avg_true_class_conf:
                min_avg_true_class_conf = avg_true_class_conf
                best_start = [accessory_img, accessory_mask, temp_attack]
                
                print('new best start found with colour {} and confidence {}'.format(colour, min_avg_true_class_conf))
                
            
        return best_start

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

        starting_point = self.get_best_starting_colour()
        

        # result = self.dodge(starting_point)

