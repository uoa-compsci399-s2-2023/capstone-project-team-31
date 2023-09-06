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

    def __init__(self, accessory_type, classification, images_dir, num_images=1, decay_rate=1, step_size=-10, lambda_tv=3, printability_coeff=5, momentum_coeff=0.4, gauss_filtering=0, max_iter=15, channels_to_fix=[], stop_prob=0.01, horizontal_move=4, vertical_move=4, rotational_move=4, verbose=True):
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
        self.colours = ['grey', 'organish', 'brownish', 'goldish']
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
                
                confidences[i] = get_confidence_in_true_class(cleanup_dims(temp_attack), self.classification, cleanup_labels(self.processed_imgs[i][self.class_num]), self.model, True)
                
            avg_true_class_conf = np.mean(confidences)
            
            if avg_true_class_conf < min_avg_true_class_conf:
                min_avg_true_class_conf = avg_true_class_conf
                best_start = Experiment(accessory_img, accessory_mask)
                
                print('new best start found with colour {} and confidence {}'.format(colour, min_avg_true_class_conf))
                
            
        return best_start

    def dodge(self, experiment: Experiment):
        '''
        pertubates the colours within the accessory mask using gradient descent to minimise deepface's confidence in predicting true labels
        '''
        # inspo: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/dodge.m
        step_size = self.step_size/self.num_images
        lambda_tv = self.lambda_tv/self.num_images
        print_coeff = self.printability_coeff/self.num_images

        print("Num GPUs Available:", tf.config.list_physical_devices('GPU'))
        print("entered dodge")
        
        scores_over_time = []
        printable_vals = get_printable_vals()
        pertubations = [Pertubation() for i in range(self.num_images)] ## where information for each image pertubation is stored: [movement_info, r]
        lowest_pert = [['NAN', np.inf, 'NAN'] for i in range(self.num_images)]

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

            for j in range(self.num_images):

                # for every image, move the accessory mask slightly 
                
                [round_accessory_im, round_accessory_area, movement_info] = move_accessory(experiment.get_image(), experiment.get_mask(), self.movement)
                pertubations[j].movement_info = movement_info
                
                ##TODO: don't touch any rgb channel which have been fixed (fixed_rgb_channels) (if we want this?)
                
                img_copy = np.copy(self.processed_imgs[j][0]) #otherwise self.processed_imgs edited
                
                attack = apply_accessory(img_copy, round_accessory_im, area_to_perturb)
                
                
                img_copy = np.copy(self.processed_imgs[j][0])
                attack = apply_accessory(img_copy, round_accessory_im, round_accessory_area)

                #imgs = np.concatenate((attack, temppp), axis=0)
                #cv2.imshow('VERTICAL', imgs)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                attacks[j] = attack
                movements[j] = movement_info
                areas_to_perturb[j] = round_accessory_area
                print("image data: {}".format(self.processed_imgs[j][1:]))
                
                if labels[j] is None:
                    labels[j] = cleanup_labels(self.processed_imgs[j][self.class_num])

                print("label: {}".format(labels[j]))
                print("True class: {}".format(labels[j]))
                
                #expand attack dim to work with deepface
                attack = cleanup_dims(attack)
                tens = int_to_float(attack)
                tens = tf.convert_to_tensor(tens)
                gradients[j] = self.model.find_resized_gradient(tens, self.model.generateLabelFromText(labels[j]))[0]

            # [scores, gradients] = find_gradient(images, true_classes) get the gradient and confidence in true class by running deepface model on images with current pertubation
            
            for x in range(self.num_images):
                
                print("second x loop: {}".format(x))
                
                # get the xith image's data from data storage arrays (inl gradients)
                im = attacks[x]
                gradient = gradients[x]
                area_to_pert = areas_to_perturb[x]
                movement_info = movements[x]

                print("PRE ROUND: max gradient: {}\nPRE ROUND: min gradient: {}".format(np.max(gradient), np.min(gradient)))
                print("gradient shape: {}\n".format(np.shape(gradient)))
                print("pertubation area shape: {}\n".format(np.shape(area_to_pert)))

                # normalise gradient
                mask = np.all(area_to_pert != [0,0,0], axis=2)
                gradient = gradient.numpy()
                gradient[mask, :] = 0
                gradient = gradient/np.max(np.abs(gradient)) 
                                

                # update the pertubation using total_variation from image_helper_functions
                _, dr_tv = total_variation(im)
                dr_tv = np.nan_to_num(dr_tv)
                print("total variation: {}\n".format(_))
                dr_tv[mask,:] = 0
                dr_tv = dr_tv/np.max(np.abs(dr_tv))

                dr_tv = np.nan_to_num(dr_tv)

                # compute pertubation and reverse the movement using reverse_accesory_movement(movement_info)

                r = step_size*gradient - dr_tv*lambda_tv
                r = np.nan_to_num(r)
                r = np.reshape(r, im.shape) #TODO: Need to check if this is exactly the shape we wanted and implemented
                
                r, r_mask = reverse_accessory_move(r, experiment['accessory_mask'], movement_info) ## TODO: what is this accessory area - this one or round_accessory area
                r[experiment['accessory_mask'] == 225] = 0

                print("r shape: {}".format(np.shape(r)))
                print("max r: {}\nmin r: {}".format(np.max(r), np.min(r)))

                # apply gaussian filtering per specification given to self.gauss_filtering
                if self.gauss_filtering != 0:
                    r = gaussian_filter(r, sigma = self.gauss_filtering)
                    print("post gauss max r: {}\npost gauss min r: {}".format(np.max(r), np.min(r)))
                
                # store the pertubation
                if i>1:
                    pertubations[x].r = self.momentum_coeff*pertubations[x].r + r 
                else:
                    pertubations[x].r = r
                pass

            # get printability score using non_printability_score in image_helper_functions.py
            nps, dr_nps = non_printability_score(experiment.get_image(), experiment.get_mask(), printable_vals)
            if self.printability_coeff != 0:
                #dr_nps[(dr_nps + experiment['accessory_image']) > 255] = 0
                #dr_nps[(dr_nps + experiment['accessory_image']) < 0] = 0
                area_to_pert = experiment['accessory_mask']
                #dr_nps[:,:,self.channels_to_fix] = 0
                gradient[area_to_pert != 1] = 0
                print("PRE ROUND: max dr_nps: {}\nPRE ROUND: min dr_nps: {}".format(np.max(dr_nps), np.min(dr_nps)))
                experiment.set_image(experiment.get_image() - print_coeff*dr_nps)

            # apply pertubations
            for r_i in range(len(pertubations)):
                r = pertubations[r_i].r
                                
                
                r = (np.rint(r)).astype(int)
                print("PRE ROUND: max r: {}\nPRE ROUND: min r: {}".format(np.max(r), np.min(r)))
                
                # perturb model
                #r[(experiment['accessory_image'] + r) > 255] = 0
                #r[(experiment['accessory_image'] + r) < 0] = 0
                experiment['accessory_image'] = experiment['accessory_image'] + r
                r[(experiment.get_image() + r) > 255] = 0
                r[(experiment.get_image() + r) < 0] = 0

                            
                result = np.add(r, experiment.get_image())                
                
                experiment.set_image(result)


            # TODO: quantization step looks sketchy, theyre just subtracting it by 0??? mod(x,1) is always 0 or am i tripping?
            # TODO: might need to change processed_imgs here once we add more images

            final_attack = apply_accessory(np.copy(self.processed_imgs[0][0]), experiment['accessory_image'], experiment['accessory_mask'])
            output, scores[j] = get_confidence_in_true_class(cleanup_dims(final_attack), self.classification, labels[j], self.model)

            score_mean = np.mean(scores)
            scores_over_time.append(score_mean)
            
            if score_mean <= lowest_pert[j][1]:
                lowest_pert[j][0] = np.copy(final_attack)
                lowest_pert[j][1] = score_mean

                mask = np.where(experiment['accessory_mask'] != 0)
                acc_img = experiment['accessory_image'].astype(np.uint8)
                acc_img[mask] = 255

                lowest_pert[j][2] = np.copy(acc_img)
                lowest_output = output
                
            # display
            if self.verbose:
                                        
                print("scores: {}".format(scores))
                print("iteration: {}".format(i))
                print("average confidence in true class: {}".format(score_mean))
                print("non-printability score: {}".format(nps))
                
                print("attack shape: {}\n".format(np.shape(attack[0])))
                
                # print out the iteration number, deepface's current confidence in image true classes, non-printability score
                # display the image with current pertubation

            i += 1 
        
        imgs = np.concatenate((lowest_pert[j][2], lowest_pert[j][0]), axis=0)
        cv2.imshow('VERTICAL', imgs)
        plt.plot(scores_over_time)
        plt.figtext(0.5, 0.01, 'Classified: {}, Confidence: {}'.format(lowest_output['classified'], lowest_output['confidence']), ha="center")
        plt.show()

        cv2.imwrite('Test_pert.png', lowest_pert[j][2])
        return attacks, experiment

    # return final pertubation result, with deepface's average confidence in predicting true classes

    def run(self):

        starting_point = self.get_best_starting_colour()

        result, experiment_result = self.dodge(starting_point)
        
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