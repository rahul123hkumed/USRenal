






import os
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cpu")




# DenseNet121 for binary valid/invalid classification
validity_model = models.densenet121(pretrained=True)
for param in validity_model.parameters():
    param.requires_grad = True

validity_model.classifier = nn.Sequential(
    nn.Linear(1024, 2),  # 2 classes for valid and not valid
)
validity_model = validity_model.to(device)






# DenseNet121 for single-label classification
single_label_model = models.densenet121(pretrained=True)
for param in single_label_model.parameters():
    param.requires_grad = True
    
single_label_model.classifier = nn.Sequential(
    nn.Linear(1024, 3),  # Change 4 to the number of single-label classes
)
single_label_model = single_label_model.to(device)




# ResNet50 for multi-label classification
multi_label_model = models.resnet50(pretrained=True)
num_ftrs = multi_label_model.fc.in_features
multi_label_model.fc = nn.Sequential(
    nn.Dropout(0.00001),  # Add dropout layer with the appropriate dropout rate
    nn.Linear(num_ftrs, 3),  # Change 3 to the number of multi-label classes
    nn.Sigmoid()
)
multi_label_model = multi_label_model.to(device)




# Load pre-trained weights
validity_weights = 'trained_Invalid_vs_valid_DN121.pth'
single_label_weights = 'trained_ValidsDN121.pth'
multi_label_weights = 'trained_multi_label_model.pth'




validity_model.load_state_dict(torch.load(validity_weights, map_location=device))
single_label_model.load_state_dict(torch.load(single_label_weights, map_location=device))
multi_label_model.load_state_dict(torch.load(multi_label_weights, map_location=device))

# Set models to evaluation mode

validity_model.eval()
single_label_model.eval()
multi_label_model.eval()





validity_class_names = ['Not a Valid Image', 'Valid']
class_names = [ 'Optimal', 'SubOptimal', 'Wrong']
multi_label_class_names = ['Artefact', 'Incorrect Gain', 'Incorrect Position']
multi_label_num_classes = 3


def generate_feedback(predicted_classes):
    feedback = []
    for prediction in predicted_classes:
        feedback_for_prediction = []
        labels = prediction.split("__")  # Change this line to split by "__" instead of "_"
        for label in labels:
            if label == "Artefact":
                feedback_for_prediction.append('''You can avoid acoustic shadowing from the ribs by asking the subject to take a 
                                               deep inspiration, which will lower the diaphragm and hence the position of the
                                                kidney away from the ribcage, or by positioning the probe in between the ribs to 
                                               avoid the artefact. Fasting or gently applying pressure on the probe to displace 
                                               the gas away from the area may partially overcome ring-down artefacts from bowel gas.
                                                <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')
                
            elif label == "Incorrect_Gain":
                feedback_for_prediction.append('''The image is either too "bright" or too "dark". <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a> or
                                                <a href="https://123sonography.com/blog/ultrasound-101-part-5-gain-and-time-gain-compensation#:~:text=What%20is%20gain%3F,much%20each%20echo%20is%20amplified." target="_blank">Visit Sonography123</a>''')
                
            elif label == "Incorrect_Position":
                feedback_for_prediction.append('''The kidney is not centrally placed or incompletely imaged. 
                                               Having the subject in decubitus position helps to get good access to 
                                               image the kidney. Additionally, blurry images may stem from incorrect hand probe positioning. 
                                               Ensuring proper alignment is essential for capturing precise and optimal images./ (<a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>) ''')
                
            elif label == "Optimal":
                feedback_for_prediction.append('''Well done/good work for obtaining optimal image quality of the kidney.''')
                
            elif label == "Wrong":
                feedback_for_prediction.append('''The image acquired is not of the kidney. <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')
            
            elif label == "Not_a_Valid_Image":
                feedback_for_prediction.append('''The image is a not valid image. <a href="https://moodle.hku.hk/mod/page/view.php?id=2748898" target="_blank">Please Visit Moodle</a>''')

            else:
                feedback_for_prediction.append('''I am sorry, something went wrong''')
        feedback.append(feedback_for_prediction)
    return feedback
