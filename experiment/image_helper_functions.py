import numpy as np
import cv2, random, sqlite3, os, base64
from PIL import Image

def pre_process_imgs():
    # if needed

    pass


def prepare_images(images_dir: str, num_images: int) -> list:
    '''
    Which images will be used for generating an adversarial pattern (could be all in the images directory)
    the silouhette mask for the chosen accessory, in the given colour
    
    Args:
    * images_dir: the directory containing the images to be used for generating an adversarial pattern, only supports .db
    * num_images: the number of images to be used for generating an adversarial pattern
    * accessory_type: the type of accessory to be added to the image
    
    Returns:
    * images: a list of the images to be used for generating an adversarial pattern, each image in the list follows: (img_base64, ethnicity, gender, age, emotion)
    '''
    # inspiration: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/prepare_experiment.m
    # take a random permutation of image filenames from images_dir of size num_images
    # for each iamge filename, load the image and store it in some type of structure (e.g. list/json whatever)
    
    if images_dir.endswith(".db"): # if the images are stored in a database
        conn = sqlite3.connect(images_dir)
        cursor = conn.cursor()
        images = cursor.execute("SELECT * FROM images ORDER BY RANDOM() LIMIT ?", (num_images,)).fetchall()
        conn.close()
        return images
    else: # if the images are stored in a directory
        return random.sample(os.listdir(images_dir), num_images)

def prepare_accessory(colour: str, accessory_dir: str, accessory_type: str) -> tuple:
    """
    Returns the silouhette mask for the chosen accessory, in the given colour
    
    Args:
        colour (str): colour of the accessory, must be one of: red, green, blue, yellow
        accessory_dir (str): directory of the accessory
        accessory_type (str): type of accessory, must be one of: glasses

    Returns:
        tuple: (accessory_image, silhouette_mask)
    """
    
    if accessory_type == "glasses":
        # load glasses_silhouette, find what pixels are white (i.e. colour value not rgb (0,0,0)) and make a colour mask of the chosen colour
        glasses = cv2.imread(accessory_dir)
        glasses = np.bitwise_not(glasses)
        # glasses = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(glasses, 0, 1, cv2.THRESH_BINARY)[1]
        
        if colour == "red":
            colour = [255, 0, 0]
        elif colour == "green":
            colour = [0, 255, 0]
        elif colour == "blue":
            colour = [0, 0, 255]
        elif colour == "yellow":
            colour = [255, 255, 0]
            
        coloured_matrix = np.array([[colour for i in range(glasses.shape[1])] for j in range(glasses.shape[0])])
        coloured_glasses = np.multiply(coloured_matrix, mask).astype(np.uint8)
        coloured_glasses = cv2.cvtColor(coloured_glasses, cv2.COLOR_RGB2BGR)
        return coloured_glasses, np.bitwise_not(glasses)
    

def move_accessory(accessory_image: np.ndarray, accessory_mask: np.ndarray, movement: dict) -> tuple:
    '''
    Moves the accessory in the image by a random amount within the ranges specified in the movement dict
    Returns the new accessory area, the new accessory image and a dict of the movements made
    
    Args:
    * accessory_image: the image of the accessory to be moved
    * movement: a dict specifying the ranges of movement in the horizontal, vertical and rotational directions
    
    Returns:
    * accessory_image: the new image of the accessory
    * accessory_mask: the new area of the image where the accessory is
    * movement_info: a dict specifying the movements made in the horizontal, vertical and rotational directions
    '''
    # inspo: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/rand_movement.m

    # generate random values for horizontal, vertical and rotational shifts within the ranges given in 'movement' dict
    shift_x = random.randint(-1*movement['horizontal'], movement['horizontal'])
    shift_y = random.randint(-1*movement['horizontal'], movement['vertical'])
    rotation = random.randint(-1*movement['rotation'], movement['rotation'])
    # shift the pixel values in accessory_mask acording to those generated values

    # keep a record of what movements were made in movement_info
    shift_x = random.randint(0, 3)
    shift_y = random.randint(0, 3)
    
    # Transform the image
    accessory_image = np.roll(accessory_image, (shift_x, shift_y), axis=(0, 1))
    accessory_image = Image.fromarray(accessory_image)
    accessory_image = accessory_image.rotate(rotation)
    accessory_image = np.array(accessory_image)
    
    # Transform the mask
    accessory_mask = np.roll(accessory_mask, (shift_x, shift_y), axis=(0, 1))
    accessory_mask = Image.fromarray(accessory_mask)
    accessory_mask = accessory_mask.rotate(rotation, fillcolor=(255, 255, 255), )
    accessory_mask = np.array(accessory_mask)
    
    movement_info = {"horizontal": shift_x, "vertical": shift_y, "rotation": rotation}
    return accessory_image, accessory_mask, movement_info

def reverse_accessory_move(accessory_image: np.ndarray, accessory_mask: np.ndarray, movement_info: dict) -> tuple:
    '''
    Reinstates original accessory position
    
    Args:
    * accessory_image: the image of the accessory to be moved, in np.ndarray format
    * movement_info: a dict specifying the ranges of movement in the horizontal, vertical and rotational directions
    
    Returns:
    * accessory_image: the new image of the accessory in np.ndarray format
    * accessory_mask: the new mask of the accessory in np.ndarray format
    '''
    
    # Transform the image
    accessory_image = Image.fromarray(accessory_image)
    accessory_image = accessory_image.rotate(movement_info['rotation'] * -1)
    accessory_image = np.array(accessory_image)
    accessory_image = np.roll(accessory_image, (movement_info['horizontal'] * -1, movement_info['vertical'] * -1), axis=(0, 1))
    
    # Transform the mask
    accessory_mask = Image.fromarray(accessory_mask)
    accessory_mask = accessory_mask.rotate(movement_info['rotation'] * -1, fillcolor=(255, 255, 255))
    accessory_mask = np.array(accessory_mask)
    accessory_mask = np.roll(accessory_mask, (movement_info['horizontal'] * -1, movement_info['vertical'] * -1), axis=(0, 1))
    return accessory_image, accessory_mask


def apply_accessory(image: np.ndarray, aug_accessory_image: np.ndarray, org_accessory_image) -> np.ndarray:
    temp = np.bitwise_and(image, org_accessory_image)
    return np.bitwise_or(temp, aug_accessory_image)

def total_variation(image: np.array, beta = 1) -> tuple:
    '''
        Calculates total variation of perturbation: image(1,224,224,3)
    '''

    # TV calculation
    d1, d2 = np.roll(image, -1, axis = 1), np.roll(image, -1, axis = 2)
    d1[:,-1,:,:], d2[:,:,-1,:] = image[:,-1,:,:], image[:,:,-1,:]

    d1 = d1 - image 
    d2 = d2 - image
    v = np.power(np.sqrt(d1*d1 + d2*d2), beta)
    tv = np.sum(v)

    # dr_tv calculation
    dr_beta = 2*(beta/2 - 1)/beta
    d1_ = np.power(np.maximum(v, 1e-5), dr_beta) * d1
    d2_ = np.power(np.maximum(v, 1e-5), dr_beta) * d2
    d11, d22 = np.roll(d1_, 1, axis = 1), np.roll(d2_, 1, axis = 2)
    d11[:,0,:,:], d22[:,:,0,:] = d1_[:,0,:,:], d2_[:,:,0,:]

    d11 = d11 - d1_ 
    d22 = d22 - d2_
    d11[:,0,:,:]  = -d1_[:,0,:,:] 
    d22[:,:,0,:]  = -d2_[:,:,0,:] 
    dr_tv = beta*(d11+d22)

    return tv, dr_tv

def softmax_loss(pred: np.array, label: np.array) -> tuple:
    '''
        Softmax loss to use for gradient descent
    '''

    tot = 0
    for i in range(len(label[0])):
        tot += np.exp(pred[0][i])
    loss = -np.log(np.exp(np.inner(pred[0], label[0],1))/tot)
    return loss

def non_printability_score(image: np.array, segmentation: np.array, printable_values: np.array) -> tuple:
    '''
    Evaluates the ability of a printer to match the colours in the pertubation
    
    Args:
        image: single rgb matrix with shape (h,w,3)
        segmentation: area of image that needs to be perturbed
        printable_values: printable values retrieved from printed palette
    '''
    # copy directly from: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/non_printability_score.m

    def norm(x,y):
        return np.sum((np.subtract(x,y))*(np.subtract(x,y)))
    
    printable_vals = np.reshape(printable_values, (printable_values.shape[0],1,3))
    max_norm = norm(np.array([0,0,0]), np.array([80,80,80]))

    # Compute non-printability scores per pixel
    scores = np.ones((image.shape[0], image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            print(segmentation[i,j,0])
            if segmentation[i,j,0] == 1:
                for k in range(printable_vals.shape[0]):
                    scores[i,j] = scores[i,j]*norm(image[i,j,:], printable_vals[k,0,:])/max_norm
            else:
                scores[i,j] = 0
    
    score = np.sum(scores)
    print(scores)

    # Compute gradient
    gradient = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if segmentation[i,j,0] == 1 and scores[i,j] != 0:
                for k in range(printable_vals.shape[0]):
                    f_k = norm(image[i,j,:], printable_vals[k,0,:])

                    # Gradients for R,G,B respectively
                    gradient[i,j,0] = gradient[i,j,0] + 2*(image[i,j,0] - printable_vals[k,0,0])*(scores[i,j]/f_k)
                    gradient[i,j,1] = gradient[i,j,1] + 2*(image[i,j,1] - printable_vals[k,0,1])*(scores[i,j]/f_k)
                    gradient[i,j,2] = gradient[i,j,2] + 2*(image[i,j,2] - printable_vals[k,0,2])*(scores[i,j]/f_k)

    gradient = gradient/np.max(np.abs(gradient))

    return score, gradient

def get_printable_vals(num_colors = 32) -> np.array:
    '''
    Creates an Nx3 matrix of all RGB values that exist in printed image
    '''
    # inspo1: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/get_printable_vals.m
    # inspo2: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/make_printable_vals_struct.m
    
    print_img = Image.open('experiment/assets/printed-palette.png')
    img_arr = np.asarray(print_img)

    # Cuts 3% of edges from each side (subject to change)
    cut_h = round(0.015*img_arr.shape[1])
    cut_v = round(0.015*img_arr.shape[0])
    img_arr = img_arr[cut_v:-cut_v, cut_h:-cut_h,:]

    # Uniform quantization, Minimum Variance Optimization not available in python (subject to change)
    printable_vals = np.round(img_arr*(num_colors/255))*(255//num_colors)
    printable_vals = printable_vals.reshape(-1, img_arr.shape[2])
    printable_vals.sort(axis=0)

    return np.unique(printable_vals, axis = 0)

def convert_b64_to_np(img_b64: str):
    decoded_img = base64.b64decode(img_b64)
    img = np.frombuffer(decoded_img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

"""
Below is a demo of the above functions
"""
# red_glasses, glasses = prepare_accessory("red", "experiment/assets/glasses_silhouette.png", "glasses")

# red_glasses, glasses, movement_info = move_accessory(red_glasses, glasses, {"horizontal": 10, "vertical": 10, "rotation": 10})

# images = prepare_images("Faces.db", 1)
# # convert to np array
# img = convert_b64_to_np(images[0][0]) 
# # Resize to 224x224 (should be done in preprocessing, together with standardization of position)
# img = cv2.resize(img, (224, 224))

# # Apply the accessory to the image
# overlay = apply_accessory(img, red_glasses, glasses)

# # Reverse the movement
# red_glasses, glasses = reverse_accessory_move(red_glasses, glasses, movement_info)

# cv2.imshow("img", img)
# cv2.imshow("overlay", overlay)
# cv2.waitKey(0)
