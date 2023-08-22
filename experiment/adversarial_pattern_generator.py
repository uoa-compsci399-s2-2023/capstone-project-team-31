from image_helper_functions import *
import numpy as np
from deepface_functions import *
from deepface_models import *
from PIL import Image
from pertubation import Pertubation
import tensorflow as tf

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
        
        if len(processed_imgs) < self.num_images:
            ## a face hasn't been detected
            print("unidentified face, num_images now {}".format(len(processed_imgs)))
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
                
                
                # cv2.imshow('image window', temp_attack)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                confidences[i] = get_confidence_in_true_class(cleanup_dims(temp_attack), self.classification, cleanup_labels(self.processed_imgs[i][self.class_num]), self.model)
                
            avg_true_class_conf = np.mean(confidences)
            
            if avg_true_class_conf < min_avg_true_class_conf:
                min_avg_true_class_conf = avg_true_class_conf
                best_start ={'accessory_image': accessory_img, 'accessory_mask': accessory_mask}
                
                print('new best start found with colour {} and confidence {}'.format(colour, min_avg_true_class_conf))
                
            
        return best_start

    def dodge(self, experiment):
        '''
        pertubates the colours within the accessory mask using gradient descent to minimise deepface's confidence in predicting true labels
        '''
        # inspo: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/dodge.m

        step_size = self.step_size/self.num_images
        lambda_tv = self.lambda_tv/self.num_images

        print("entered dodge")
        
        printable_vals = get_printable_vals()
        pertubations = [Pertubation() for i in range(self.num_images + 1)] ## where information for each image pertubation is stored: [movement_info, r]

        i = 0
        score_mean = 1

        while i < self.max_iter and score_mean > self.stop_prob:
            
            ("while loop iteration {}".format(i))
            #data storing:
            attacks = [None] * self.num_images
            movements = [None] * self.num_images
            areas_to_perturb = [None] * self.num_images
            gradients = [None] * self.num_images
            scores = [None] * self.num_images

            for j in range(self.num_images):
                
                print("first j loop: {}".format(j))

                # for every image, move the accessory mask slightly 
                [round_accessory_im, round_accessory_area, movement_info] = move_accessory(experiment['accessory_image'], experiment['accessory_mask'], self.movement)
                pertubations[j].movement_info = movement_info
                
                area_to_perturb = round_accessory_area
                
                ##TODO: don't touch any rgb channel which have been fixed (fixed_rgb_channels) (if we want this?)
                
                attack = apply_accessory(self.processed_imgs[j][0], round_accessory_im, area_to_perturb)
                
                attacks[j] = attack
                movements[j] = movement_info
                areas_to_perturb[j] = area_to_perturb
                
                print("image data: {}".format(self.processed_imgs[j][1:]))
                
                label = cleanup_labels(self.processed_imgs[j][self.class_num])
                
                print("label: {}".format(label))
                
                print("True class: {}".format(label))
                
                #expand attack dim to work with deepface
                attack = cleanup_dims(attack)
                
                attack = int_to_float(attack)
                
                tens = tf.convert_to_tensor(attack)
                
                gradients[j] = self.model.find_resized_gradient(tens, self.model.generateLabelFromText(label))[0]
                scores[j] = get_confidence_in_true_class(attack, self.classification, label, self.model)


            # [scores, gradients] = find_gradient(images, true_classes) get the gradient and confidence in true class by running deepface model on images with current pertubation
            print("gradients: {}\nscores: {}".format(gradients, scores))
            
            for x in range(self.num_images):
                
                print("second x loop: {}".format(x))
                # TODO: need to define processed_image and gradients variables :/
                
                # get the xith image's data from data storage arrays (inl gradients)
                im = attacks[x]
                gradient = gradients[x]
                area_to_pert = areas_to_perturb[x]
                movement_info = movements[x]
                
                print("gradient shape: {}\ngradient: {}".format(np.shape(gradient), gradient))
                print("pertubation area shape: {}\n pertubation area: {}".format(np.shape(area_to_pert), area_to_pert))

                # normalise gradient
                mask = np.all(area_to_pert == [255, 255, 255], axis=2)
                gradient = gradient.numpy()
                gradient[mask, :] = [0, 0, 0] 
                gradient = gradient/np.max(np.abs(gradient)) 
                

                # update the pertubation using total_variation from image_helper_functions
                _, dr_tv = total_variation(im)
                print("total variation: {}\ndr_tv: {}".format(_, dr_tv))
                dr_tv[mask,:] = [0, 0, 0]
                dr_tv = dr_tv/np.max(np.abs(dr_tv))

                # compute pertubation and reverse the movement using reverse_accesory_movement(movement_info)
                r = step_size*gradient - dr_tv*lambda_tv
                r = np.reshape(r, im.shape) #TODO: Need to check if this is exactly the shape we wanted and implemented
                
                print("r shape: {}\nr:{}".format(np.shape(r), r))
                r = reverse_accessory_move(r, experiment['accessory_mask'], movement_info) 
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
            score_mean = np.mean(scores)
            # display
            if self.verbose:
                
                print("iteration: {}".format(i))
                print("average confidence in true class: {}".format(score_mean))
                print("non-printability score: {}".format(nps))
                print("attack:")
                
                print("attack shape: {}\n attack format: {}".format(np.shape(attack[0]), attack[0]))
                
                cv2.imshow('image window', attacks[0])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # print out the iteration number, deepface's current confidence in image true classes, non-printability score
                # display the image with current pertubation

            i += 1 

        return attacks

    # return final pertubation result, with deepface's average confidence in predicting true classes

    def run(self):

        starting_point = self.get_best_starting_colour()
        

        result = self.dodge(starting_point)
        
        
def cleanup_labels(true_class:str):
## cleaning up different classification terms

    if true_class.lower() == 'female':
        result = 'Woman'
    elif true_class.lower() == 'male':
        result = 'Man'
    
    return result


def cleanup_dims(image):
    
    ## cleaning up dimension issues:
    if len(np.shape(image)) == 3:
        image = np.expand_dims(image, axis=0)
        
    return image

def int_to_float(image):
    image = image.astype(np.float32)
    image = np.divide(image, 255)
    
    return image

def float_to_int(image):
    
    image = np.multiply(image, 255)
    image = image.astype(np.uint8)