import numpy as np
import cv2, random, sqlite3, os, base64
from PIL import Image
import sys
import os, json
from imgaug import augmenters as iaa
from imgaug import parameters as iap

def prepare_images(images_dir: str, json_dir: str, num_images: int, mode="dodge", classification=None, target=None, seperate=True) -> list:
    '''
    Which images will be used for generating an adversarial pattern (could be all in the images directory)
    the silouhette mask for the chosen accessory, in the given colour
    
    Args:
    * images_dir: the directory containing the images to be used for generating an adversarial pattern, supports folders and db
    * json_dir: the directory containing information of each image in folder. Must be in json format
    * num_images: the number of images to be used for generating an adversarial pattern
    * accessory_type: the type of accessory to be added to the image
    
    Returns:
    * images: a list of the images to be used for generating an adversarial pattern, each image in the list follows: (img_base64, ethnicity, gender, age, emotion)
    '''
    # inspiration: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/prepare_experiment.m
    # take a random permutation of image filenames from images_dir of size num_images
    # for each iamge filename, load the image and store it in some type of structure (e.g. list/json whatever)

    #random.seed(0)

    abs_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, images_dir)) #bug fixing
    
    if images_dir.endswith(".db"): # if the images are stored in a database
        conn = sqlite3.connect(abs_path)
        cursor = conn.cursor()
        if mode == "impersonation" and seperate:
            print(classification, target)
            command = "SELECT * FROM images WHERE {} != ? ORDER BY RANDOM() LIMIT {}".format(classification, num_images)
            images = cursor.execute(command, (target,)).fetchall()
        else:
            images = cursor.execute("SELECT * FROM images ORDER BY RANDOM() LIMIT ?", (num_images,)).fetchall()
        conn.close()
        return images
    else: # if the images are stored in a directory
        images = os.listdir(abs_path)
        output = []

        if json_dir == '':
            raise Exception('json directory missing')
        
        with open(json_dir, 'r') as f:
            data = json.load(f)
            temp_images = random.sample(list(data.keys()), num_images)

            for img in temp_images:
                temp =  cv2.imread(os.path.join(abs_path, img))
                output.append([temp, data[img]['ethnicity'], data[img]['gender'], data[img]['age'], data[img]['emotion']])
        return output
    
from deepface.commons import functions

def getImageObjects(img_path,
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
):
    img_objs = functions.extract_faces(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    
    return img_objs
    
def getImageContents(img_path,
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
):
    img_objs = getImageObjects(img_path, 
                               enforce_detection = enforce_detection,
                               align = align, 
                               detector_backend = detector_backend)
    contents = []
    for (content, region, _) in img_objs:
        contents.append(content)
        
    return contents


def image_to_face(image: tuple):
    '''
    Takes an image, and returns a normalized 224x224 image centered on face using Deepface image detection
    
    Args:
    * images : image to detect from, in the form outputted from prepare_images
    
    Returns
    * Tuple of (img , ethnicity, gender, age, emotion)
    * Normalized image in the shape (1,224,224,3) <- this is what deepface outputs - might be worth changing to (224,224,3)
    
    '''
    #b64 = "data:image/jpg;base64/," + image[0]
    b64 = image[0]
    if type(b64) == str:
        img = convert_b64_to_np(b64)
    else:
        img = b64

    try:
        img = getImageContents(img)[0]
    except ValueError:
        #This means that deepface did not detect a face in this image
        return None
    
    img = np.multiply(img, 255).astype(np.uint8)
    
    outputImage = (img[0], image[1], image[2], image[3],image[4])
    
    return outputImage

def prepare_processed_images(images_dir: str, json_dir: str, num_images: int,  mode="dodge", classification=None, target=None, seperate=True):
    '''
    Detect faces and normalize a given amount of images
    
    Args:
    * images_dir: the directory containing the images to be used for generating an adversarial pattern, only supports .db
    * json_dir: the directory containing information of each image in folder. Must be in json format
    * num_images: the number of images to get
    
    Returns:
    * list of processed images in the form  (img , ethnicity, gender, age, emotion)
    '''
    image_list = prepare_images(images_dir, json_dir, num_images, mode=mode, classification=classification, target=target, seperate=seperate)
    output_list = []
    for image in image_list:
        prepared_image = image_to_face(image)     
        if(prepared_image != None):
            output_list.append(prepared_image)
        
    return output_list


def prepare_accessory(colour: str, accessory_dir: str, accessory_type: str) -> tuple:
    """
    Returns the silouhette mask for the chosen accessory, in the given colour
    
    Args:
    * colour (str): colour of the accessory, must be one of: red, green, blue, yellow
    * accessory_dir (str): directory of the accessory
    * accessory_type (str): type of accessory. Name must correspond to filename

    Returns:
    * tuple: (accessory_image, silhouette_mask)
    """
    accessories = os.listdir('./experiment/assets')
    fname = accessory_type.lower()

    if fname == "glasses" or fname == "facemask" or fname == "bandana" or fname == "earrings" or \
        (fname + '.png') in accessories or (fname + '.jpg') in accessories or (fname + '.jpeg') in accessories:
        # load glasses_silhouette, find what pixels are white (i.e. colour value not rgb (0,0,0)) and make a colour mask of the chosen colour
        accessory = cv2.imread(accessory_dir)
    else:
        print("Please check your accessory spelling. Image files supported (.jpg, .jpeg, .png)")
        
    if accessory is None:
        print("Error loading accessory from {}".format(accessory_dir))
    accessory = np.bitwise_not(accessory)
    mask = cv2.threshold(accessory, 0, 1, cv2.THRESH_BINARY)[1]
    
    # make a colour mask of the chosen colour
    colour_info = json.load(open("./experiment/assets/starting_colours.json", 'r'))
    colour = colour_info[colour]
        
    coloured_matrix = np.array([[colour for i in range(accessory.shape[1])] for j in range(accessory.shape[0])])
    coloured_accessory = np.multiply(coloured_matrix, mask).astype(np.uint8)
    coloured_accessory = cv2.cvtColor(coloured_accessory, cv2.COLOR_RGB2BGR)
    return coloured_accessory, np.bitwise_not(accessory)
    

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

    # generate random values for horizontal, vertical and rotational shifts within the ranges given in 'movement' dict
    shift_x = random.randint(-1*movement['horizontal'], movement['horizontal'])
    shift_y = random.randint(-1*movement['horizontal'], movement['vertical'])

    rotation = random.randint(-1*movement['rotation'], movement['rotation'])
    
    # Save the movement info
    movement_info = {"horizontal": shift_x, "vertical": shift_y, "rotation": rotation}

    # Transform the image
    accessory_image = np.roll(accessory_image, (shift_x, shift_y), axis=(0, 1))
    rot_aug = iaa.Affine(rotate=iap.Deterministic(rotation))
    accessory_image = rot_aug.augment_image(accessory_image)
    
    # Transform the mask
    accessory_mask = np.roll(accessory_mask, (shift_x, shift_y), axis=(0, 1))
    rot_aug = iaa.Affine(rotate=iap.Deterministic(rotation), cval=255)
    accessory_mask = rot_aug.augment_image(accessory_mask)
    
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
    accessory_image = np.copy(accessory_image)
    rot_aug = iaa.Affine(rotate=iap.Deterministic(movement_info["rotation"] * -1))
    accessory_image = rot_aug.augment_image(accessory_image)
    accessory_image = np.roll(accessory_image, (movement_info['horizontal'] * -1, movement_info['vertical'] * -1), axis=(0, 1))
    
    # Transform the mask
    accessory_mask = np.copy(accessory_mask)
    rot_aug = iaa.Affine(rotate=iap.Deterministic(movement_info["rotation"] * -1), cval=255)
    accessory_mask = rot_aug.augment_image(accessory_mask)
    accessory_mask = np.roll(accessory_mask, (movement_info['horizontal'] * -1, movement_info['vertical'] * -1), axis=(0, 1))
    return accessory_image, accessory_mask

def merge_accessories(accessory_dir: str, num_images: int) -> np.ndarray:
    '''
    Merges accessories into one by averaging each pixel

    Args:
    * accessory_dir: directory of accessories
    * num_images: sample size

    Returns:
    * final_mask: averaged mask
    '''
    abs_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, accessory_dir))

    images = os.listdir(accessory_dir)
    rand_images = random.sample(images, num_images)
    mat_sum = np.zeros((224,224,3))

    for img in rand_images:
        temp =  cv2.imread(os.path.join(abs_path, img))
        mat_sum = mat_sum + temp
    
    mat_sum = mat_sum/num_images
    final_mask = mat_sum.astype(np.uint8)

    return final_mask

def average_nonzero(arr_int, mask):
    '''
    Returns average non-zero value from an array 
    '''
    n = 0
    a = 0
    arr = arr_int.astype("float")
    for i in range(len(arr_int)):
        row = arr_int[i]
        for j in range(len(row)):
            pixel = row[j]
            if((mask[i][j] == [255,255,255]).all()):
                continue
            
            if(n == 0):
                a = np.zeros_like(pixel)
            #if(n < 100):
                #print(mask[i][j])
            a = a + pixel
            n += 1
    #print(a)
    a = a / n
    return a

def merge_accessories_delta(accessory_dir: str, colour = "organish", accessory_type = "facemask", directory = "./experiment/assets/facemask.png" ) -> np.ndarray:
    '''
    Merges accessories into one by removing colour, and combining differences together

    Args:
    * accessory_dir: directory of accessories
    * colour: colour of all the accesories

    Returns:
    * final_mask: combined mask
    '''
    abs_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, accessory_dir))
    images = os.listdir(accessory_dir)
    #rand_images = random.sample(images, num_images)
    accessory_image, accessory_mask = prepare_accessory(colour, directory, accessory_type)
    accessory_image = accessory_image.astype("float")
    #accessory_image = cv2.cvtColor(accessory_image, cv2.COLOR_BGR2RGB)
    mat_sum = np.zeros((224,224,3))
    for img in images:
        
        temp =  cv2.imread(accessory_dir + "/" + img)
        temp = cv2.resize(temp, dsize = (224,224))
        temp = temp.astype("float")
        #temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        mask = np.all(accessory_mask != [255, 255, 255], axis=-1)
        
        #print(np.shape(temp))
        avg = average_nonzero(temp,accessory_mask)
        #print(avg)
        temp[mask] -= avg
        delta = temp 
        mat_sum = mat_sum + delta
    
    mat_sum = mat_sum + accessory_image
    mat_sum = mat_sum.clip(0,255)
    final_mask = mat_sum.astype(np.uint8)
    rgb_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB)
    PILImage = Image.fromarray(rgb_mask)
    PILImage.save("finalmask.png")
    return final_mask

def apply_accessory(image: np.ndarray, aug_accessory_image: np.ndarray, org_accessory_image: np.ndarray) -> np.ndarray:
    '''
    Applies accessory to image

    Args:
    * image: single rgb matrix with shape (h,w,3)
    * aug_accessory_image: augmented accessory rgb matrix with shape (h,w,3)
    * org_accessory_image: mask of accessory

    Returns:
    * image: image with new accessory
    '''

    mask = np.where(org_accessory_image == 0)
    image[mask] = aug_accessory_image[mask]
    return image

def change_con_bright(image: np.ndarray, range: int) -> tuple:
    '''
    Changes contrast and brightness of image

    Args:
    * image: single rgb matrix with shape (h,w,3)
    * range: range at which brightness and contrast is randomised
    
    Returns:
    * changed_image: image with modified brightness and contrast
    * ab_params: contrast and brightness parameters
    '''
    if range < 0:
        raise Exception("Brightness and contrast variation should be positive or 0")

    if range != 0:
        beta = np.random.randint(-range, range)
        alpha =  1 + beta/100

        img_copy = np.copy(image)
        changed_image = cv2.convertScaleAbs(img_copy, alpha=alpha, beta=beta)
        ab_params = (alpha, beta)

        return changed_image, ab_params
    
    else:
        return image, (0,0)

def total_variation(image: np.array, beta = 1) -> tuple:
    '''
    Calculates total variation of perturbation
    
    Args:
    * image: Single rgb matrix with shape (h,w,3) 
    * beta: Magnitude to increase exponential value
    
    Returns:
    * tv: Total variation of image
    * dr_tv: Total variation gradient of each pixel (h,w,3)
    '''

    # TV calculation
    d1, d2 = np.roll(image, -1, axis = 0), np.roll(image, -1, axis = 1)
    d1[-1,:,:], d2[:,-1,:] = image[-1,:,:], image[:,-1,:]

    d1 = d1 - image 
    d2 = d2 - image
    v = np.power(np.sqrt(d1*d1 + d2*d2), beta)
    tv = np.sum(v)

    # dr_tv calculation
    dr_beta = 2*(beta/2 - 1)/beta
    d1_ = np.multiply(np.power(np.maximum(v, 1e-5), dr_beta), d1)
    d2_ = np.multiply(np.power(np.maximum(v, 1e-5), dr_beta), d2)
    d11, d22 = np.roll(d1_, 1, axis = 0), np.roll(d2_, 1, axis = 1)
    d11[0,:,:], d22[:,0,:] = d1_[0,:,:], d2_[:,0,:]

    d11 = d11 - d1_ 
    d22 = d22 - d2_
    d11[0,:,:]  = -d1_[0,:,:] 
    d22[:,0,:]  = -d2_[:,0,:] 
    dr_tv = beta*(d11+d22)

    return tv, dr_tv

def softmax_loss(pred: np.array, label: np.array) -> float:
    '''
    Softmax loss to use for gradient descent

    Args:
    * pred: Prediction array. Currently has shape (1,2)
    * label: Label array. Currently has shape (1,2)
        
    Returns
    * loss: Softmax loss
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
    * image: Single rgb matrix with shape (h,w,3)
    * segmentation: Area of image that needs to be perturbed
    * printable_values: Printable values retrieved from printed palette (N, 3)
 
    Returns:
    * score: Non printability score
    * gradient: Non printability score gradient with shape (h,w,3)
    '''

    def norm(x,y):
        return np.sum((np.subtract(x,y))*(np.subtract(x,y)))
    
    printable_vals = np.reshape(printable_values, (printable_values.shape[0],1,3))
    max_norm = norm(np.array([0,0,0]), np.array([80,80,80]))
    
    # Compute non-printability scores per pixel
    scores = np.ones((image.shape[0], image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if segmentation[i,j] == 0:
                for k in range(printable_vals.shape[0]):
                    scores[i,j] = scores[i,j]*norm(image[i,j], printable_vals[k,0,:])/max_norm
            else:
                scores[i,j] = 0
    
    score = np.sum(scores)

    # Compute gradient
    gradient = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if segmentation[i,j] == 0 and scores[i,j] != 0:
                for k in range(printable_vals.shape[0]):
                    f_k = norm(image[i,j], printable_vals[k,0])

                    # Gradients for R,G,B respectively
                    gradient[i,j,0] = gradient[i,j,0] + 2*(image[i,j,0] - printable_vals[k,0,0])*(scores[i,j]/f_k)
                    gradient[i,j,1] = gradient[i,j,1] + 2*(image[i,j,1] - printable_vals[k,0,1])*(scores[i,j]/f_k)
                    gradient[i,j,2] = gradient[i,j,2] + 2*(image[i,j,2] - printable_vals[k,0,2])*(scores[i,j]/f_k)

    gradient = gradient/np.max(np.abs(gradient))

    return score, gradient

def get_printable_vals(num_colors = 100) -> np.array:
    '''
    Creates an Nx3 matrix of all RGB values that exist in printed image

    Args:
    * num_colors: Number (roughly num_colors*3) of unique colors to reduce image to
    
    Returns:
    * printable_vals: Matrix of unique colors (N,3)
    '''

    printable_vals = []
    with open('./experiment/assets/printable_vals.txt') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            line = list(map(int, line))
            printable_vals.append(line)

    return np.array(printable_vals)

def convert_b64_to_np(img_b64: str):
    '''
    For converting b64 images to rgb matrix (h,w,3)

    Args:
    * img_b64: image stored in b64 format

    Returns:
    * img: rgb np array (h,w,3)
    '''
    decoded_img = base64.b64decode(img_b64)
    img = np.frombuffer(decoded_img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


