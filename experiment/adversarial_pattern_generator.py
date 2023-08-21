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
        
        processed_imgs = prepare_processed_images(self.images_dir, self.num_images)
        self.processed_imgs = processed_imgs
        
        if len(processed_imgs) > self.num_images:
            ## a face hasn't been detected
            self.num_images = len(processed_imgs)
        
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

        best_start = []
        min_avg_true_class_conf = 1
        
        for colour in self.colours:
            
            accessory_img, accessory_mask = prepare_accessory(colour, "./assets/{}_silhouette.png".format(self.accessory_type), self.accessory_type)
            
            confidences = np.empty(self.num_images)
            
            for i in range(self.num_images):
            
                temp_attack = apply_accessory(self.processed_imgs[i][0], accessory_img, accessory_mask)

                temp_attack = temp_attack.astype(np.uint8)
                
                
                cv2.imshow('image window', temp_attack)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                self.processed_imgs[i][self.class_num] = cleanup_labels(self.processed_imgs[i][self.class_num])
                
                confidences[i] = get_confidence_in_true_class(temp_attack, self.classification, self.processed_imgs[i][self.class_num], self.model)
                
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

        step_size = self.step_size/self.num_images
        lambda_tv = self.lambda_tv/self.num_images

        printable_vals = get_printable_vals()
        pertubations = [[dict(), 0] for i in range(self.num_images + 1)] ## where information for each image pertubation is stored: [movement_info, r]

        print(pertubations)
        i = 0
        score_mean = 1

        while i < self.max_iter and score_mean > self.stop_prob:

            #data storing:
            attacks = [] * self.num_images
            movements = [] * self.num_images
            areas_to_perturb = [] * self.num_images
            gradients = [] * self.num_images
            scores = [] * self.num_images

            for j in range(self.num_images):

                # for every image, move the accessory mask slightly 
                [round_accessory_im, round_accessory_area, movement_info] = move_accessory(experiment[0], experiment[1], self.movement)
                pertubations[j][0] = movement_info
                
                area_to_perturb = round_accessory_area
                
                ##TODO: don't touch any rgb channel which have been fixed (fixed_rgb_channels) (if we want this?)
                
                attack = apply_accessory(self.processed_imgs[j][0], round_accessory_im, area_to_perturb)
                
                attacks[j] = attack
                movements[j] = movement_info
                areas_to_perturb[j] = area_to_perturb
                
                gradients[j] = self.model.find_gradient(attack, self.processed_imgs[j][self.class_num])
                scores[j] = get_confidence_in_true_class(attack, self.classification, self.processed_imgs[i][self.class_num])


            # [scores, gradients] = find_gradient(images, true_classes) get the gradient and confidence in true class by running deepface model on images with current pertubation

            for x in range(experiment['num_images']):
                # TODO: need to define processed_image and gradients variables :/
                
                # get the xith image's data from data storage arrays (inl gradients)
                im = processed_image[x,:,:,:]
                gradient = gradients[x,:,:,:]
                area_to_pert = areas_to_perturb[x,:,:,:]
                movement_info = movements[x]

                # normalise gradient
                gradient[area_to_pert != 1] = 0
                gradient = gradient/np.max(np.abs(gradient)) 

                # update the pertubation using total_variation from image_helper_functions
                _, dr_tv = total_variation(im)
                dr_tv[area_to_pert != 1] = 0
                dr_tv = dr_tv/np.max(np.abs(dr_tv))

                # compute pertubation and reverse the movement using reverse_accesory_movement(movement_info)
                r = self.step_size*gradient - dr_tv*self.lambda_tv
                r = np.reshape(r, im.shape) #TODO: Need to check if this is exactly the shape we wanted and implemented
                r = reverse_accessory_move(r, movement_info) #TODO: This function asks for three inputs, im not sure what else to put here :((
                r[experiment['accessory_area'] != 1] = 0

                # apply gaussian filtering per specification given to self.gauss_filtering
                
                # store the pertubation
                if i>1:
                    #TODO: What exactly is .r here, is perturbations its own class?
                    #Otherwise we can just store the r of each perturbation in another array :))
                    pertubations[x].r = self.momentum_coeff*pertubations[x].r + r 
                else:
                    pertubations[x].r = r
                pass


            # get printability score using non_printability_score in image_helper_functions.py
            nps, dr_nps = non_printability_score(experiment['accessory_image'], experiment['accessory_area'][:,:,0], printable_vals)
            if self.printability_coeff != 0:
                dr_nps = -dr_nps
                dr_nps[(dr_nps + experiment['accessory_image']) > 255] = 0
                dr_nps[(dr_nps + experiment['accessory_image']) < 0] = 0
                area_to_pert = experiment['accessory_area']
                dr_nps[:,:,self.channels_to_fix] = 0
                gradient[area_to_pert != 1] = 0
                experiment['accessory_image'] = experiment['accessory_image'] + self.printability_coeff*dr_nps

            # apply pertubations
            # TODO: Again check how to actually store and retrieve r value of each perturbation <3
            for r_i in range(pertubations.shape[0]):
                r = pertubations[r_i].r

                # perturb model
                r[(experiment['accessory_image'] + r) > 255] = 0
                r[(experiment['accessory_image'] + r) < 0] = 0
                experiment['accessory_image'] = experiment['accessory_image'] + r

            # TODO: quantization step looks sketchy, theyre just subtracting it by 0??? mod(x,1) is always 0 or am i tripping?

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
        

        result = self.dodge(starting_point)
        
        
def cleanup_labels(true_class:str):
## cleaning up different classification terms

    if true_class.lower() == 'female':
        true_class = 'Woman'
    elif true_class.lower() == 'male':
        true_class = 'Man'
    
    return true_class

