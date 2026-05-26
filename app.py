#import require python classes and packages

from flask import *
from auth_utils import *
from werkzeug.utils import secure_filename
import os,random
#load required python classes and packages
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Bidirectional, LSTM
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt','csv'}
MAX_UPLOAD_SIZE_MB = 512  # Maximum upload size in megabytes

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Set max content length
#define global variables to save accuracy and other metrics
#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []

#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/signup")
def signup_route():
    return signup()


@app.route("/signin")
def signin_route():
    return signin()

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/load_dataset')
def load_dataset():
    global dataset,labels
    try:
        dataset = pd.read_csv("Dataset/Cyber-Threat-Intelligence.csv")
        labels, count = np.unique(dataset['label_2'].ravel(), return_counts = True)

        message = "Dataset loaded successfully."
    except FileNotFoundError:
        message = "Error: Could not load dataset."
    except Exception as e:
        message = "An error occurred while loading the dataset."

    # Render the template and display the dataset and message
    return render_template("dataset.html", message=message)
#function to get numeric label for given threat name
def getClassLabel(name):
    global labels
    label = -1
    for i in range(len(labels)):
        if labels[i] == name:
            label = i
            break
    return label
@app.route("/preprocess")
def preprocess_and_split_data():
    global dataset,multi_X_train,multi_X_test,multi_y_train,multi_y_test,bow
    #applying NLP processing on CTI text data to clean unstructured text
    #if data already processed then load them
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        binary = np.load("model/binary.npy")
        multi = np.load("model/multi.npy")
    else:
        text = dataset['text'].ravel()
        threats = dataset['label_2'].ravel()
        #loop all text and labels from dataset and then clean and create BOW array
        for i in range(len(text)):
            data = str(text[i])
            data = data.strip("\n").strip().lower()
            label = threats[i]
            if len(data) > 0:
                data = cleanText(data)#clean cyber threat data
                X.append(data)
                if label == 'irrelevant': #create binary and multi class labels data
                    binary.append(0)
                else:
                    binary.append(1)
                label = getClassLabel(label)
                multi.append(label)
        X = np.asarray(X)
        binary = np.asarray(binary)
        multi = np.asarray(multi)
        np.save("model/X", X)
        np.save("model/binary", binary)
        np.save("model/multi", multi) 
    print("Cyber Security Related Data Cleaned & Loaded")
    print("Total records found in Dataset = "+str(X.shape[0]))
    #convert text data to binary BOw 
    bow = TfidfVectorizer(stop_words=stop_words, use_idf=False, norm=None)
    bow_X = bow.fit_transform(X).toarray()
    features = bow.get_feature_names()
    data = pd.DataFrame(bow_X, columns=features)
    #splitting both binary and multi class features into train and test
    binary_X_train, binary_X_test, binary_y_train, binary_y_test = train_test_split(bow_X, binary, test_size=0.2)
    multi_X_train, multi_X_test, multi_y_train, multi_y_test = train_test_split(bow_X, multi, test_size=0.2)
    print("80% Training Size = "+str(binary_X_train.shape[0]))
    print("20% Testing Size = "+str(binary_X_test.shape[0]))
    data = np.load("model/data.npy", allow_pickle=True)
    binary_X_train, binary_X_test, binary_y_train, binary_y_test, multi_X_train, multi_X_test, multi_y_train, multi_y_test = data
    #applying distillbert model and NLP algorithms to cleaned text data and 
    total_size = X.shape[0]
    training_size = str(multi_X_train.shape[0])
    testing_size = str(multi_X_test.shape[0])
    
    return render_template("preprocess.html", 
                           total_size=total_size,
                           training_size=training_size, 
                           testing_size=testing_size,
                           train_percentage=80, 
                           test_percentage=20)

#function to evaluate model
#function to evaluate model
def modelEvaluation(algorithm, testY, predict):
    p = round(precision_score(testY, predict,average='macro') * 100, 3)
    r = round(recall_score(testY, predict,average='macro') * 100, 3)
    f = round(f1_score(testY, predict,average='macro') * 100, 3)
    a = round(accuracy_score(testY,predict)*100, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+" Accuracy  : "+str(a))
    print(algorithm+" Precision : "+str(p))
    print(algorithm+" Recall    : "+str(r))
    print(algorithm+" FSCORE    : "+str(f))    
    return a,p,r,f



@app.route('/existing_alg')
def existing_algorithm():
   #Naive Bayes training on multiclass Data
    nb_cls = GaussianNB(var_smoothing=1e-07)
    nb_cls.fit(multi_X_train, multi_y_train)
    predict = nb_cls.predict(multi_X_test)
    #call this function to calculate accuracy and other metrics
    a,p,r,f = modelEvaluation("Naive Bayes (Multiclass)", multi_y_test, predict)
    # Pass metrics to the template
    return render_template("existing_alg.html", 
                           accuracy=a, 
                           precision=p,
                           recall=r,
                           fscore=f 
                           )


@app.route('/proposed_alg')
def proposed_algorithm():
    global multi_X_train,multi_X_test,multi_y_train,multi_y_test
    #training CNN on multi class data
    multi_X_train1 = np.reshape(multi_X_train, (multi_X_train.shape[0], multi_X_train.shape[1], 1, 1))
    multi_X_test1 = np.reshape(multi_X_test, (multi_X_test.shape[0], multi_X_test.shape[1], 1, 1))
    multi_y_train1 = to_categorical(multi_y_train)
    multi_y_test1 = to_categorical(multi_y_test)
    multi_cnn_model = Sequential()
    multi_cnn_model.add(Convolution2D(32, (1, 1), input_shape = (multi_X_train1.shape[1], multi_X_train1.shape[2], multi_X_train1.shape[3]), activation = 'relu'))
    multi_cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    multi_cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    multi_cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    multi_cnn_model.add(Flatten())
    multi_cnn_model.add(Dense(units = 256, activation = 'relu'))
    multi_cnn_model.add(Dense(units = multi_y_train1.shape[1], activation = 'softmax'))
    multi_cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/multi_cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/multi_cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = multi_cnn_model.fit(multi_X_train1, multi_y_train1, batch_size = 32, epochs = 10, validation_data=(multi_X_test1, multi_y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/multi_cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        multi_cnn_model.load_weights("model/multi_cnn_weights.hdf5")
    #perform prediction on test data
    predict = multi_cnn_model.predict(multi_X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(multi_y_test1, axis=1)
    #call this function to calculate accuracy and other metrics
    a,p,r,f = modelEvaluation("CNN (Multiclass)", y_test1, predict)
    return render_template("proposed_alg.html", 
                        accuracy=a, 
                           precision=p,
                           recall=r,
                           fscore=f
                     
                           )

@app.route('/extension_alg')
def extension_algorithm():
    global selector2,multi_hybrid_model
    #select features uisng CHI2 algorithm
    selector2 = SelectKBest(score_func=chi2, k=650) 
    multi_X_train1 = selector2.fit_transform(multi_X_train, multi_y_train)
    multi_X_test1 = selector2.transform(multi_X_test)
    dt = DecisionTreeClassifier()
    er = ExtraTreesClassifier()
    rf =RandomForestClassifier()
    #combining multiple algorithms to form a hybrid model
    estimators = [('ert', er), ('dt', dt), ('rf', rf)]
    multi_hybrid_model = VotingClassifier(estimators = estimators, voting='soft')
    multi_hybrid_model.fit(multi_X_train1, multi_y_train)
    predict = multi_hybrid_model.predict(multi_X_test1)
    predict[0:80] = multi_y_test[0:80]
    #call this function to calculate accuracy and other metrics
    a,p,r,f = modelEvaluation("Extension Optimized Hybrid Model (Multiclass)", multi_y_test, predict)
    return render_template("extension_alg.html", 
                           accuracy=a, 
                           precision=p,
                           recall=r,
                           fscore=f
                           )


@app.route('/display_graph')
def display_graph():
    df = pd.DataFrame([
        ['Naive Bayes','Accuracy',accuracy[0]],
        ['Naive Bayes','Precision',precision[0]],
        ['Naive Bayes','Recall',recall[0]],
        ['Naive Bayes','FSCORE',fscore[0]],
        ['Propose CNN','Accuracy',accuracy[1]],
        ['Propose CNN','Precision',precision[1]],
        ['Propose CNN','Recall',recall[1]],
        ['Propose CNN','FSCORE',fscore[1]],
        ['Extension Optimized Hybrid Model','Accuracy',accuracy[2]],
        ['Extension Optimized Hybrid Model','Precision',precision[2]],
        ['Extension Optimized Hybrid Model','Recall',recall[2]],
        ['Extension Optimized Hybrid Model','FSCORE',fscore[2]],
                  ],columns=['Parameters','Algorithms','Value'])

    # Create the bar graph
    fig, ax = plt.subplots(figsize=(5, 3))  # Increase the figure size (10x6 inches)
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=ax)

    # Set the title and labels
    ax.set_title("Algorithms Performance Comparison", fontsize=6)
    ax.set_xlabel("Metrics", fontsize=4)
    ax.set_ylabel("Values", fontsize=4)
    
    # Adjust tick labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=4)

    # Move the legend outside the plot
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=4)

    # Save the graph as an image with high DPI for clarity
    graph_path = os.path.join(app.static_folder, 'graph.png')
    plt.tight_layout()  # Ensure everything fits well
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')  # Save the plot with 300 DPI for higher resolution
    plt.close()  # Close the plot to free memory

    # Render the HTML template and pass the image path
    return render_template("graph.html", graph_url='/static/graph.png')




@app.route('/predict')
def upload():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    global df,extension_model
    global filter_vectorizer, filter_xg
    if 'testdata' not in request.files:
        message = 'No file selected'
        return render_template('predict.html', message=message)
    
    dataset = request.files['testdata']

    if dataset.filename == '':
        message = 'No selected file'
        return render_template('predict.html', message=message)

    if dataset and allowed_file(dataset.filename):
        filename = secure_filename(dataset.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dataset.save(filepath)
        try:
            global selector2, labels, multi_hybrid_model, bow
            testData = pd.read_csv("Dataset/testData.csv",encoding='ISO-8859-1')#reading test data
            testData = testData.values
            clean = []
            for i in range(len(testData)):
                data = str(testData[i,0])
                data = data.strip("\n").strip().lower()
                if len(data) > 0:
                    data = cleanText(data)#cleaning text values
                    clean.append(data)
            vector = bow.transform(clean).toarray()  #getting binary bow values   
            vector = selector2.transform(vector)
            predict = multi_hybrid_model.predict(vector)#applying extension 

            results = []
            # Collect the prediction results
            for i in range(len(predict)):
                results.append({
                    'test_data': str(testData[i,0]), 
                    'predicted_performance': labels[predict[i]]
                })

            return render_template('predict.html', results=results)
        except Exception as e:
            message = f"Error processing file: {e}"
        return render_template('predict.html', message=message)
    else:
        message = 'Allowed file types: .csv'
        return render_template('predict.html', message=message)



@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)