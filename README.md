# Spam Email Detector
This program classifies email as spam or ham using LSTM and machine learning algorithms. 
The program takes emails as input and provides the probability that the email is spam or ham. The program provides keywords associated with spam emails.

## Datasets Used
The subset of Enron Email Dataset was used to train the model. The subset contains 5854 emails of which 1496 are classified as spam and the rest correspond to Ham emails.  

<b>Spam emails from Enron Dataset</b> </br>
  File name: labeled_emails.csv </br>
  Original link: https://www.kaggle.com/code/juanagsolano/spam-email-classifier-from-enron-dataset/input </br>

## How to run this code

1. Create a virtual environment:
```
python3 -m venv myenv
```
2. Download and unzip project files:
```
unzip Spam-Email-Detector-main.zip 
```

3. Move files to your virtual environment folder
4. Activate your virtual environment:

On MacOS:
```
source myvenv/bin/activate
```
On Windows:
```
.\venv\Scripts\activate.bat
```
5. Navigate to your Spam-Email-Detector-main folder:
6. Install dependencies:
```
pip3 install -r requirements.txt
```
7. Run the script:
```
python3 app.py
```

8. Copy & Paste the provided URL link on your browser:
```
http://127.0.0.1:5000
```

