from image_helper_functions import *
import numpy as np
from deepface_functions import *
from deepface_models import *
from PIL import Image
from pertubation import Pertubation
from experiment_model import Experiment
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import matplotlib.pyplot as plt

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

    def __init__(self, mode, accessory_type, classification, images_dir, num_images=1, decay_rate=0, step_size=10, lambda_tv=3, printability_coeff=5, momentum_coeff=0.4, gauss_filtering=0, max_iter=15, channels_to_fix=[], stop_prob=0.01, horizontal_move=4, vertical_move=4, rotational_move=4, target=None, verbose=True):
        self.mode = mode
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
        
        self.decay_rate = decay_rate
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
        self.movement['rotation'] = rotational_move
        self.colours = ['grey', 'organish', 'brownish', 'goldish', 'yellowish']
                
        self.target = None
        if target is not None:
            self.target = cleanup_labels(target)
            
        self.verbose = verbose
        
        self.model = attributeModel(self.classification)


    def get_best_starting_colour(self):
        '''
        returns starting configuration where deepface is least confident in its prediction of the true class for each image
        '''

        best_start = []

        min_avg_true_class_conf = 1
        
        for colour in self.colours:
            
            accessory_img, accessory_mask = prepare_accessory(colour, "./assets/{}.png".format(self.accessory_type.lower()), self.accessory_type)
            
            confidences = np.empty(self.num_images)
            
            for i in range(self.num_images):
                
                img_copy = np.copy(self.processed_imgs[i][0]) #otherwise self.processed_imgs edited
            
                temp_attack = apply_accessory(img_copy, accessory_img, accessory_mask)

                temp_attack = temp_attack.astype(np.uint8)
                
                if self.mode == "impersonation":
                    label = self.target
                elif self.mode == "dodge":
                    label = cleanup_labels(self.processed_imgs[i][self.class_num])
                
                _, confidences[i] = get_confidence_in_selected_class(cleanup_dims(temp_attack), self.classification, label, self.model, True)
                
            avg_true_class_conf = np.mean(confidences)
            
            if avg_true_class_conf < min_avg_true_class_conf:
                min_avg_true_class_conf = avg_true_class_conf
                best_start = Experiment(accessory_img, accessory_mask)
                
                print('new best start found with colour {} and confidence {}'.format(colour, min_avg_true_class_conf))
                
            
        return best_start

    def run_experiment(self, experiment: Experiment):
        '''
        pertubates the colours within the accessory mask using gradient descent to minimise deepface's confidence in predicting true labels
        '''
        # inspo: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/dodge.m
        step_size = self.step_size/self.num_images
        lambda_tv = self.lambda_tv/self.num_images
        print_coeff = self.printability_coeff/self.num_images

        print("Num GPUs Available:", tf.config.list_physical_devices('GPU'))
        
        scores_over_time = []
        printable_vals = get_printable_vals()
        pertubations = [Pertubation() for i in range(self.num_images)] ## where information for each image pertubation is stored: [movement_info, r]
        lowest_pert = [np.NaN, np.inf, np.NaN] # Stores the lowest recorded attack, score, and accessory
        final_attacks = [np.NaN]*self.num_images # Stores final attack without any accessory movement

        i = 0
        score_mean = 1

        while i < self.max_iter and score_mean > self.stop_prob:
            
            #data storing:
            attacks = [None] * self.num_images
            movements = [None] * self.num_images
            areas_to_perturb = [None] * self.num_images
            gradients = [None] * self.num_images
            scores = [None] * self.num_images
            labels = [None] * self.num_images

            # Learning rate decay
            step_size = step_size/(1+self.decay_rate*self.max_iter)
            lambda_tv = lambda_tv/(1+self.decay_rate*self.max_iter)
            print_coeff = print_coeff/(1+self.decay_rate*self.max_iter)

            # Calculate gradient of each accessory movement
            for j in range(self.num_images):

                # for every image, move the accessory mask slightly 
                [round_accessory_im, round_accessory_area, movement_info] = move_accessory(experiment.get_image(), experiment.get_mask(), self.movement)
                pertubations[j].movement_info = movement_info
                
                ##TODO: don't touch any rgb channel which have been fixed (fixed_rgb_channels) (if we want this?)             
                
                img_copy = np.copy(self.processed_imgs[j][0])
                attack = apply_accessory(img_copy, round_accessory_im, round_accessory_area)

                attacks[j] = attack
                movements[j] = movement_info
                areas_to_perturb[j] = round_accessory_area
                
                if labels[j] is None:
                    
                    if self.mode == "impersonation":
                        labels[j] = self.target
                    elif self.mode == "dodge":
                        labels[j] = cleanup_labels(self.processed_imgs[j][self.class_num])
                
                #expand attack dim to work with deepface
                attack = cleanup_dims(attack)
                tens = int_to_float(attack)
                tens = tf.convert_to_tensor(tens)
                gradients[j] = self.model.find_resized_gradient(tens, self.model.generateLabelFromText(labels[j]))[0]
            
            # Calculate tv and dr/tv
            for x in range(self.num_images):
                # get the xith image's data from data storage arrays (inl gradients)
                im = attacks[x]
                gradient = gradients[x]
                area_to_pert = areas_to_perturb[x]
                movement_info = movements[x]

                # normalise gradient
                mask = np.all(area_to_pert != [0,0,0], axis=2)
                gradient = gradient.numpy()
                
                if self.mode == "impersonation":
                    gradient = np.multiply(gradient, -1) ## is there a better way to do this?
                
                gradient[mask, :] = 0
                gradient = gradient/np.max(np.abs(gradient)) 
                                
                # update the pertubation using total_variation from image_helper_functions
                tv, dr_tv = total_variation(im)
                dr_tv = np.nan_to_num(dr_tv)
                dr_tv[mask,:] = 0
                dr_tv = dr_tv/np.max(np.abs(dr_tv))

                dr_tv = np.nan_to_num(dr_tv)

                # compute pertubation and reverse the movement using reverse_accesory_movement(movement_info)
                r = step_size*gradient - dr_tv*lambda_tv
                r = np.nan_to_num(r)
                r = np.reshape(r, im.shape)
                
                r, r_mask = reverse_accessory_move(r, experiment.get_mask(), movement_info)
                r[experiment.get_mask() != 0] = 0

                # apply gaussian filtering per specification given to self.gauss_filtering
                if self.gauss_filtering != 0:
                    r = gaussian_filter(r, sigma = self.gauss_filtering)
                
                # store the pertubation
                if i>1:
                    pertubations[x].r = self.momentum_coeff*pertubations[x].r + r 
                else:
                    pertubations[x].r = r
            
            # get printability score using non_printability_score in image_helper_functions.py
            nps, dr_nps = non_printability_score(experiment.get_image(), experiment.get_mask()[:,:,0], printable_vals)
            if self.printability_coeff != 0:
                dr_nps[(dr_nps + experiment.get_image()) > 255] = 0
                dr_nps[(dr_nps + experiment.get_image()) < 0] = 0
                area_to_pert = experiment.get_mask()
                #dr_nps[:,:,self.channels_to_fix] = 0
                gradient[area_to_pert != 1] = 0
                experiment.set_image(experiment.get_image() - print_coeff*dr_nps)

            # apply pertubations
            for r_i in range(len(pertubations)):
                r = pertubations[r_i].r
                r = (np.rint(r)).astype(int)
                r[(experiment.get_image() + r) > 255] = 0
                r[(experiment.get_image() + r) < 0] = 0       
                result = np.add(r, experiment.get_image())                
                
                experiment.set_image(result)

            for k in range(self.num_images):
                final_attacks[k] = apply_accessory(np.copy(self.processed_imgs[k][0]), experiment.get_image(), experiment.get_mask())
                if self.mode == "impersonation":
                    label = self.target
                elif self.mode == "dodge":
                    label = cleanup_labels(self.processed_imgs[k][self.class_num])

                output, scores[k] = get_confidence_in_selected_class(cleanup_dims(final_attacks[k]), self.classification, label, self.model)
            
            score_mean = np.mean(scores)
            scores_over_time.append(score_mean)
            
            if score_mean <= lowest_pert[1]:
                lowest_pert[0] = np.copy(final_attacks[0])
                lowest_pert[1] = score_mean

                mask = np.where(experiment.get_mask() != 0)
                acc_img = experiment.get_image().astype(np.uint8)
                acc_img[mask] = 255

                lowest_pert[2] = np.copy(acc_img)
                lowest_output = output
                
            # display
            if self.verbose:
                
                flavour_text = "true"
                
                if self.mode == "impersonation":
                    flavour_text = "target"
                                        
                print("scores: {}".format(scores))
                print("iteration: {}".format(i))
                print("average confidence in {} class: {}".format(flavour_text, score_mean))
                print("non-printability score: {}".format(nps))
                print("total_variation score: {}".format(tv))
                
                # print out the iteration number, deepface's current confidence in image true classes, non-printability score
                # display the image with current pertubation

            i += 1 
        
        imgs = np.concatenate((lowest_pert[2], lowest_pert[0]), axis=0)
        cv2.imshow('Final Attack', imgs)
        plt.plot(scores_over_time)
        plt.figtext(0.5, 0.01, 'Classified: {}, Confidence: {}'.format(lowest_output['classified'], lowest_output['confidence']), ha="center")
        plt.show()

        #cv2.imwrite('Test_pert.png', lowest_pert[2])
        return final_attacks, experiment

    # return final pertubation result, with deepface's average confidence in predicting true classes

    def run(self):

        starting_point = self.get_best_starting_colour()

        result, experiment_result = self.run_experiment(starting_point)
        
        cv2.imwrite("./results/accessory_image.png", experiment_result.get_image())        
        
        for attack in result:
            cv2.imshow('image window', attack)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
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
