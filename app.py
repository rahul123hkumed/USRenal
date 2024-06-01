




from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
from torchvision import transforms, models
from util import (validity_class_names, validity_model, class_names, multi_label_num_classes, single_label_model, 
                  single_label_weights, multi_label_model, multi_label_weights, 
                  device, multi_label_class_names, generate_feedback)
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from email import encoders


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
app.jinja_env.globals.update(zip=zip)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']






def send_email_with_attachments(to, subject, text, files):
    from_address = "hkumed01@gmail.com"
    password = "yunhxkwbgxatiaat"

    msg = MIMEMultipart()
    msg["From"] = from_address
    msg["To"] = COMMASPACE.join([to])
    msg["Subject"] = subject
    msg.attach(MIMEText(text))

    for file in files:
        part = MIMEBase("application", "octet-stream")
        with open(file, "rb") as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file)}")
        msg.attach(part)

    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    # server.starttls()  # Remove or comment out this line
    server.login(from_address, password)
    server.sendmail(from_address, to, msg.as_string())
    server.quit()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash('No selected file')
            return redirect(request.url)

        name = request.form.get('name')
        scholar_id = request.form.get('scholar_id')
        email = request.form.get('email')

        filenames = []
        predicted_classes = []
        saved_file_paths = []


        # Data preprocessing
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Load the image
                img = Image.open(file_path).convert('RGB')

                # Preprocess the image
                input_tensor = data_transform(img)
                input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
                input_batch = input_batch.to(device)

                # Make predictions with the validity classifier
                with torch.no_grad():
                    validity_preds = validity_model(input_batch)
                    validity_class = torch.argmax(validity_preds).item()

                if validity_class_names[validity_class] == 'Valid':
                    # Make predictions with the single-label classifier
                    with torch.no_grad():
                        preds1 = single_label_model(input_batch)
                        class1 = torch.argmax(preds1).item()

                    # If the image is SubOptimal, use the multi-label classifier
                    if class_names[class1] == 'SubOptimal':
                        with torch.no_grad():
                            preds2 = multi_label_model(input_batch)
                            class2 = (preds2 > 0.5).squeeze().cpu().numpy().astype(int)

                        # Create a prefix for the new filename
                        prefix = '__'.join([multi_label_class_names[i].replace(" ", "_") for i, c in enumerate(class2) if c == 1]) + '__'
                        predicted_label = prefix[:-2]

                    else:
                        predicted_label = class_names[class1]

                    predicted_classes.append(predicted_label)

                    predicted_label = predicted_label.replace(" ", "_")
                    filename = secure_filename(f"{name}_{scholar_id}_{email}_{predicted_label}_{file.filename}")
                    os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    filenames.append(filename)
                    saved_file_paths.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                else:
                    # If the image is not valid, skip the other classifiers and set the predicted_label to 'Not_Valid'
                    predicted_label = 'Not a Valid Image'
                    predicted_classes.append(predicted_label)
                    filename = secure_filename(f"{name}_{scholar_id}_{email}_{predicted_label}_{file.filename}")
                    os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    filenames.append(filename)
                    saved_file_paths.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))


                # Send email with the uploaded images
        to_email = "hkumed01@gmail.com"
        subject = "Uploaded Images"
        text = f"Here are the uploaded images from {name} (Scholar ID: {scholar_id})."
        send_email_with_attachments(to_email, subject, text, saved_file_paths)

        feedback_messages = generate_feedback(predicted_classes)  # Use your custom generate_feedback function here
        file_urls = [url_for('uploaded_file', filename=f) for f in filenames]

        return render_template('index.html', predicted_classes=predicted_classes, feedback_messages=feedback_messages, file_urls=file_urls, feedback_data=zip(files, feedback_messages))

    return render_template('index.html')






@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)






@app.route('/demo')
def demo_images():
    folder_names =  ['Optimal','Artefact','Wrong','Incorrect_gain', 'Incorrect_position']
    folder_images = {}
    
    for folder in folder_names:
        image_names = os.listdir(os.path.join('static', folder))
        folder_images[folder] = image_names

    return render_template('demo_images.html', folder_images=folder_images)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))