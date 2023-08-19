from adversarial_pattern_generator import AdversarialPatternGenerator

accessory_type = 'glasses' ## get from args
img_dir = './adversarial_training.db' # soemthing like this, change when we have actual db

adv_pattern_generator = AdversarialPatternGenerator(accessory_type, img_dir) # and can specify any other paramenters from args