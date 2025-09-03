import streamlit as st
from langchain_google_genai.llms import GoogleGenerativeAI
import time
import graphviz
import os


GOOGLE_API_KEYs = st.secrets.GOOGLE_API_KEY
model_name = "gemini-1.5-flash-latest"  #"gemini-pro"

if "output_explain_key" not in st.session_state:
    st.session_state.output_explain_key = ""

def stream_data_explain():
    for word in st.session_state.output_explain_key.split(" "):
        yield word + " "
        time.sleep(0.03)

@st.cache_resource(show_spinner=False)
def llm_model(prompt):
    llm = GoogleGenerativeAI(
    model= model_name,       
    api_key=GOOGLE_API_KEYs
    )
    # st.session_state.output_explain_key = llm.invoke(prompt)
    generated_answer = llm.invoke(prompt)
    return generated_answer

@st.cache_data(show_spinner=False)
def explain_code_prompt(code_snippets):
    code_explanation = f"""
Code Snippet:
import cv2
import numpy as np 
import sqlite3
import os
import playsound
import tkinter
from tkinter import messagebox
from tkinter import *

uname =""
conn = sqlite3.connect('database.db')

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

c = conn.cursor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  if faces is ():
    return None

  for (x, y, w, h) in faces:
    cropped_face = img[y:y + h, x:x + w]

  return cropped_face


cap = cv2.VideoCapture(0)

root=Tk()
root.configure(background="white")
# -------------------------------------------
def getInput1():
  global uname
  inputValue = textBox.get("1.0","end-1c")
  if inputValue != "":
    uname = inputValue
  else:
    print('Please enter the name. it is mandatory')
    root.withdraw()
    messagebox.showwarning("Warning", "Please enter the name. It is mandatory field...")
    exit()
  print(inputValue)
  root.destroy()

L1 = Label(root, text = 'Enter Your Name : ',font=("times new roman",12))
L1.pack()
textBox = Text(root, height=1, width = 20,font=("times new roman",12),bg="Pink",fg='Red')
textBox.pack()
textBox.focus()
buttonSave = Button(root, height = 1, width = 10,font=("times new roman",12), text = "Save", command = lambda:getInput1())
buttonSave.pack()
# buttonQuit = Button(root, height = 1, width = 10,font=("times new roman",12), text = "Exit", command = lambda:quit())
# buttonQuit.pack()
mainloop()
# -------------------------------------------
# uname = input("Enter your name: ")
c.execute('INSERT INTO users1 (name) VALUES (?)', (uname,))
uid = c.lastrowid
count = 0
while True:
  ret, frame = cap.read()
  # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  # for (x,y,w,h) in faces:
  #   sampleNum = sampleNum+1
  #   cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
  #   cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
  #   cv2.waitKey(100)
  # cv2.imshow('img',img)
  # cv2.waitKey(1);
  # if sampleNum >= 20:
  # break
  if face_extractor(frame) is not None:
    count += 1
    face = cv2.resize(face_extractor(frame), (150, 150))
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    file_name_path = 'DataSet/User.'+str(uid)+"."+str(count)+'.jpg'
    # file_name_path1 = 'DataSet/frame'+str(count)+'.jpg'

    cv2.imwrite(file_name_path, face)
    # cv2.imwrite(file_name_path1,frame)

    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Face Cropper', face)
  else:
    print("Face not Found")
    pass
  if cv2.waitKey(1) == 13 or count == 20:
    break

print("\nSamples captured successfully...")
playsound.playsound('sound.mp3')

cap.release()
conn.commit()
conn.close()
cv2.destroyAllWindows()



Code Explaination:
This code is a face data collection tool that captures and stores images of a user's face. It is part of a facial recognition or registration system where images are saved in a dataset for later use, such as training facial recognition models.

Step-by-Step Explanation

    1. Import Libraries

            import cv2
            import numpy as np 
            import sqlite3
            import os
            import playsound
            import tkinter
            from tkinter import messagebox
            from tkinter import *

        ⦿ cv2: Handles computer vision tasks like face detection and image capturing.

        ⦿ sqlite3: Interacts with an SQLite database.

        ⦿ os: Checks for file/folder existence and manages the file system.

        ⦿ playsound: Plays a sound after capturing images.

        ⦿ tkinter: Provides GUI elements for user input.

    2. Set Up the Database and Dataset Folder

            conn = sqlite3.connect('database.db')
            if not os.path.exists('./dataset'):
                os.makedirs('./dataset')
            Creates a connection to database.db.

        ⦿ Ensures that a dataset folder exists to save face images.

    3. Initialize Face Detection

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ⦿ Loads the Haar Cascade classifier for face detection.

    4. Define Face Extraction Function

            def face_extractor(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if faces is ():
                    return None

                for (x, y, w, h) in faces:
                    cropped_face = img[y:y + h, x:x + w]
                return cropped_face

        ⦿ Converts the frame to grayscale for better face detection.

        ⦿ Detects faces and returns the cropped face region.

        ⦿ Returns None if no face is detected.

    5. Set Up GUI for User Input

            root=Tk()
            root.configure(background="white")

            def getInput1():
                global uname
                inputValue = textBox.get("1.0","end-1c")
                if inputValue != "":
                    uname = inputValue
                else:
                    root.withdraw()
                    messagebox.showwarning("Warning", "Please enter the name. It is mandatory field...")
                    exit()
                root.destroy()

            L1 = Label(root, text = 'Enter Your Name : ',font=("times new roman",12))
            L1.pack()
            textBox = Text(root, height=1, width = 20,font=("times new roman",12),bg="Pink",fg='Red')
            textBox.pack()
            textBox.focus()
            buttonSave = Button(root, height = 1, width = 10,font=("times new roman",12), text = "Save", command = lambda:getInput1())
            buttonSave.pack()
            mainloop()

        ⦿ Creates a simple Tkinter GUI where the user enters their name.
        
        ⦿ If no name is entered, a warning message is shown, and the program exits.

    6. Save User to Database

            c.execute('INSERT INTO users1 (name) VALUES (?)', (uname,))
            uid = c.lastrowid

        ⦿ Inserts the username into the database.

        ⦿ Fetches the uid (unique ID) of the newly added user for naming the images.

    7. Capture Face Images

            cap = cv2.VideoCapture(0)
            count = 0

            while True:
                ret, frame = cap.read()
                if face_extractor(frame) is not None:
                    count += 1
                    face = cv2.resize(face_extractor(frame), (150, 150))

                    file_name_path = 'DataSet/User.'+str(uid)+"."+str(count)+'.jpg'
                    cv2.imwrite(file_name_path, face)

                    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
                    cv2.imshow('Face Cropper', face)
                else:
                    print("Face not Found")
                    pass

                if cv2.waitKey(1) == 13 or count == 20:  # Exit if 'Enter' is pressed or 20 images are captured
                    break
            
        ⦿ Captures video from the webcam.

        ⦿ Extracts the face using face_extractor.

        ⦿ Resizes and saves the face image as User.<uid>.<count>.jpg in the DataSet folder.

        ⦿ Stops capturing when 20 images are captured or the Enter key (13) is pressed.

    8. Completion and Cleanup

            print("\nSamples captured successfully...")
            playsound.playsound('sound.mp3')

            cap.release()
            conn.commit()
            conn.close()
            cv2.destroyAllWindows()

        ⦿ Plays a sound to indicate successful capture.

        ⦿ Releases the camera, commits database changes, and closes connections.

How It Works
    ⦿ Run the Script: The program opens a GUI for the user to input their name.
    ⦿ Enter Name: The user inputs their name and clicks "Save".

    ⦿ Capture Images:
        The webcam starts, and the program detects and captures 20 images of the user’s face.
        Each image is saved with a unique filename in the DataSet folder.
    
    ⦿ Save Data: The user’s name and associated images are stored in the SQLite database.

    ⦿ Completion: A sound plays, and the program exits.

Key Features
    1.Face Detection: Uses Haar Cascade for efficient face detection.
    2.User Interface: Tkinter GUI for username input.
    3.Database Integration: Stores user details in SQLite.
    4.Dataset Creation: Saves cropped face images for future use.





Code Snippet:
def mergeSort(array):
    if len(array) > 1:

        #  r is the point where the array is divided into two subarrays
        r = len(array)//2
        L = array[:r]
        M = array[r:]

        # Sort the two halves
        mergeSort(L)
        mergeSort(M)

        i = j = k = 0

        # Until we reach either end of either L or M, pick larger among
        # elements L and M and place them in the correct position at A[p..r]
        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1

        # When we run out of elements in either L or M,
        # pick up the remaining elements and put in A[p..r]
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1


# Print the array
def printList(array):
    for i in range(len(array)):
        print(array[i], end=" ")
    print()


# Driver program
if __name__ == '__main__':
    array = [6, 5, 12, 10, 9, 1]

    mergeSort(array)

    print("Sorted array is: ")
    printList(array)

Code Explanation:

Let’s simplify the explanation of the MergeSort code step by step:

What is MergeSort?
Merge Sort is a divide-and-conquer sorting algorithm. It divides an array into smaller subarrays, sorts those subarrays, and then merges them to produce a sorted result. It is widely used for its efficiency and stability.

Step-by-Step Explanation of the Code

    Step 1: Divide the Array

            r = len(array)//2
            L = array[:r]
            M = array[r:]

        ⦿ The array is split into two halves:

        ⦿ L: Left half of the array (from start to the middle).

        ⦿ M: Right half of the array (from middle to the end).

    Step 2: Recursively Sort Both Halves

            mergeSort(L)
            mergeSort(M)

        ⦿ Keep splitting each half into smaller parts until the subarrays have just one element.

        ⦿ A single-element array is already sorted.

    Step 3: Merge Two Sorted Halves
        ⦿ The goal is to combine two sorted halves (L and M) into a single sorted array (array).

        ⦿ Merging Two Halves
        ⦿ The merging process involves comparing the smallest elements of the two halves and adding the smaller one to the final array.

    Step 3.1: Initialize Pointers

            i = j = k = 0

        ⦿ i: Tracks the position in the left half (L).

        ⦿ j: Tracks the position in the right half (M).

        ⦿ k: Tracks the position in the original array (array).

    Step 3.2: Compare and Pick Smallest Elements

            while i < len(L) and j < len(M):
                if L[i] < M[j]:
                    array[k] = L[i]
                    i += 1
                else:
                    array[k] = M[j]
                    j += 1
                k += 1

         Compare L[i] (current element of the left half) with M[j] (current element of the right half):

            ⦿ If L[i] is smaller, add it to the array and move the i pointer forward.

            ⦿ Otherwise, add M[j] to the array and move the j pointer forward.

            ⦿ Move the k pointer forward in the original array to fill the next position.

    Step 3.3: Add Remaining Elements
        If one half is fully used, copy the remaining elements from the other half:

            while i < len(L):
                array[k] = L[i]
                i += 1
                k += 1

            while j < len(M):
                array[k] = M[j]
                j += 1
                k += 1

        ⦿ Copy all remaining elements from L if any are left.

        ⦿ Copy all remaining elements from M if any are left.

Example Walkthrough
    Input Array:
    array = [6, 5, 12, 10, 9, 1].

    Step 1: Divide the Array
        Split into [6, 5, 12] and [10, 9, 1].
        Further split until each subarray has 1 element:
        [6], [5], [12], [10], [9], [1].
    Step 2: Merge and Sort
        Start merging:
        [6] and [5] → [5, 6].
        [5, 6] and [12] → [5, 6, 12].
        [10] and [9] → [9, 10].
        [9, 10] and [1] → [1, 9, 10].
    Step 3: Final Merge
        Merge [5, 6, 12] and [1, 9, 10] → [1, 5, 6, 9, 10, 12].
Output
    The sorted array is:
        [1, 5, 6, 9, 10, 12].

Summary
    1.Divide the array into two halves until each subarray has 1 element.
    2.Sort and merge the halves by comparing their elements.
    3.Repeat until the entire array is sorted.





Code Snippet:
import cv2
import numpy as np
from os import listdir
# from os.path import isfile, join
from playsound import playsound

data_path = 'DataSet/'
onlyFiles = [f for f in listdir(data_path)]
print(onlyFiles)

Training_Data= []
Labels  = []

for i, files in enumerate(onlyFiles):
    image_path = data_path + onlyFiles[i]
    print(image_path)
    images = cv2.imread(image_path)
    cv2.imshow('Train to Dataset', images)
    cv2.waitKey(100)

    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    Training_Data.append(np.asarray(images, dtype=np.uint8))
    # print(Training_Data)
    Labels.append(i)
    # print(Labels)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")
# playsound('TrainingCompleted.M4A')


Code Explanation:

This code snippet is designed to train a face recognition model using the Local Binary Patterns Histograms (LBPH) algorithm. It loads images from a dataset and extracts facial features to create a model that can recognize faces.

Step-by-Step Breakdown:

1. Import Libraries:

        import cv2
        import numpy as np
        from os import listdir
        # from os.path import isfile, join
        from playsound import playsound

    ⦿ These lines import necessary libraries for image processing, data manipulation, and playing a sound.

2. Load Dataset Images:

        data_path = 'DataSet/'
        onlyFiles = [f for f in listdir(data_path)]

    ⦿ data_path specifies the directory containing the dataset images.
    ⦿ onlyFiles lists all the image files in the directory.

3. Initialize Training Data and Labels:

        Training_Data= []
        Labels  = []

    ⦿ These lists will store the training images and their corresponding labels (indicating which person the image belongs to).

4. Read and Process Images:

        for i, files in enumerate(onlyFiles):
            image_path = data_path + onlyFiles[i]
            print(image_path)
            images = cv2.imread(image_path)
            cv2.imshow('Train to Dataset', images)
            cv2.waitKey(100)

            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            Training_Data.append(np.asarray(images, dtype=np.uint8))
            # print(Training_Data)
            Labels.append(i)
            # print(Labels)

        ⦿ This loop iterates through each image in the dataset.
        ⦿ Each image is read in color (RGB) and displayed for 100 milliseconds.
        ⦿ Then, it is converted to grayscale, which is more suitable for face recognition.
        ⦿ The grayscale image is added to the Training_Data list, and its label (the index i) is added to the Labels list.

5. Convert Labels to NumPy Array:

        Labels = np.asarray(Labels, dtype=np.int32)

    ⦿ Converts the Labels list to a NumPy array of 32-bit integers.

6. Create and Train the Model:

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(np.asarray(Training_Data), np.asarray(Labels))

    ⦿ An LBPH face recognizer model is created using cv2.face.LBPHFaceRecognizer_create().
    ⦿ The train() method is used to train the model with the Training_Data and Labels.

7. Training Completion Notification:

        print("Model Training Complete!!!!!")
        # playsound('TrainingCompleted.M4A')

    ⦿ This line prints a message indicating that the model training is complete.
    ⦿ Optionally, it can play a sound to notify the user.

Key Features:

    ⦿ Dataset Loading: Loads images from a specified directory.
    ⦿ Image Preprocessing: Converts images to grayscale and resizes them for consistent processing.
    ⦿ Data Augmentation: Creates multiple images from a single face by flipping and adding noise (optional).
    ⦿ Model Creation: Initializes and trains an LBPH face recognizer model.
    ⦿ Training Completion Notification: Informs the user when training is finished.

How It Works:

    ⦿ The program loads images from the dataset directory and converts them to grayscale.
    ⦿ It creates a training dataset by converting each image to a NumPy array and adding its corresponding label.
    ⦿ An LBPH face recognizer model is created and trained using the training dataset.
    ⦿ Once training is complete, the model can be used to recognize faces in new images.

"""
    prompt_explanation= f"""
    you act like ai code pyhton explainer and give the explain of the code in 500 words of its working
    Explain the following Python code briefly in 500 words and provide a step-by-step explanation of what the code does. For each line or block of code, describe its functionality in simple terms. Focus on clarity and conciseness.
    ####
    {code_explanation}
    ####
    Code Snippet is shared below, delimited with triple backticks:
    and Code Snippet is shared below is not Code or is a general question then answer the query that like that "Sorry, I Cannot Answer this question because i am only a AI-code-explainer Bot. Please Provide Code Snippit so i can help you" , delimited with triple backticks:
    ```
    {code_snippets}
    ```
    """
    return prompt_explanation

@st.cache_data(show_spinner=False)
def improve_code_prompt(code_snippets):
    code_improve=f"""
Code Snippet:
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        crop1 = face_extractor(frame)
        face = cv2.resize(crop1,(120,120))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'DataSet/user'+str(count)+'.jpg'     #Dataset/user4.jpg
        # file_name_path1 = 'DataSet/frame'+str(count)+'.jpg'
        # print(file_name_path)
        # print(file_name_path1)

        cv2.imwrite(file_name_path,face)
        # cv2.imwrite(file_name_path1,frame)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==27 or count==25:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Completed!!!')



code Improve:
Here are some tips to improve the code in terms of functionality, performance, and maintainability:

1. face_extractor function

    Suggestions:

        ⦿ Edge case handling: 
            
            The if faces is () condition is incorrect. It should be if len(faces) == 0 or if not faces. The condition faces is () checks if faces is the same object as an empty tuple, which may not work as intended in this case.

        ⦿ Return early if no face detected: 
            
            It's more efficient to return immediately when no face is detected, without proceeding further in the function.

        ⦿ Code readability: 
            
            The for loop to extract the face is unnecessary because the function is only interested in the first face detected. It's better to return after cropping the first detected face.

    Improved face_extractor function:

        def face_extractor(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:  # No faces found
                return None

            # Only return the first face found
            x, y, w, h = faces[0]
            cropped_face = img[y:y+h, x:x+w]
    
            return cropped_face

2. Main Loop (Capturing Faces)

    Suggestions:

        ⦿ Repeated function call: 
            
            The face_extractor function is called twice for the same frame. This is inefficient as it processes the same image twice. Call it once and reuse the result.

        ⦿ Face count control: 
            
            Rather than hardcoding 25 as the maximum number of faces to capture, it's better to make it a configurable parameter or print a message when the collection is done.

        ⦿ Code readability: 
            
            Adding descriptive comments can improve the readability of the loop, especially explaining the importance of each action, such as saving files and displaying frames.

    Improved main loop:

        cap = cv2.VideoCapture(0)
        count = 0
        max_samples = 25  # Define the maximum number of samples to collect

        while True:
            ret, frame = cap.read()

            face = face_extractor(frame)  # Only call face_extractor once

            if face is not None:
                count += 1
                face_resized = cv2.resize(face, (120, 120))  # Resize the face to 120x120

                # Define the file path where the face image will be saved
                file_name_path = f'DataSet/user.jpg'

                # Save the cropped and resized face image
                cv2.imwrite(file_name_path, face_resized)

                # Add a text counter on the image to indicate the number of samples
                cv2.putText(face_resized, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face_resized)
            else:
                print("Face not found")

            # Exit when the user presses the 'Esc' key or when 25 samples are collected
            if cv2.waitKey(1) == 27 or count >= max_samples:
                break

        cap.release()
        cv2.destroyAllWindows()

        print('Collecting Samples Completed!!!')

3. Other Improvements

    Suggestions:

        ⦿ Path handling: 
            
            It's a good idea to make sure the directory (DataSet/) exists before trying to save images there. You can use os.makedirs() to ensure the directory exists.

        ⦿ Clear code flow: 
            
            If you are storing images of users, a good practice is to check whether the image directory exists and handle errors like permission issues.

        ⦿ Graceful exit: 
            
            Adding a message to inform the user when the script is about to exit would be helpful (e.g., "Collection completed" or "Press Esc to exit").

    Adding directory check:
        import os

        # Ensure the dataset directory exists
        if not os.path.exists('DataSet'):
            os.makedirs('DataSet')

Summary of Improvements

    ⦿ Efficiency:
        
          Calling face_extractor once per frame instead of twice, saving computation.

    ⦿ Condition checks: 
        
        Correcting the face detection condition if len(faces) == 0 to handle the case when no face is detected.

    ⦿ Path handling: 
        
        Ensuring the DataSet directory exists before saving images.

    ⦿ Clear and concise loop: 
        
        The loop logic is simplified and unnecessary variables are removed for clarity.

These improvements will help the code run more efficiently, reduce redundancy, and be easier to maintain.
    




Code Snippet:
def mergeSort(array):
    if len(array) > 1:

        #  r is the point where the array is divided into two subarrays
        r = len(array)//2
        L = array[:r]
        M = array[r:]

        # Sort the two halves
        mergeSort(L)
        mergeSort(M)

        i = j = k = 0

        # Until we reach either end of either L or M, pick larger among
        # elements L and M and place them in the correct position at A[p..r]
        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1

        # When we run out of elements in either L or M,
        # pick up the remaining elements and put in A[p..r]
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1


# Print the array
def printList(array):
    for i in range(len(array)):
        print(array[i], end=" ")
    print()


# Driver program
if __name__ == '__main__':
    array = [6, 5, 12, 10, 9, 1]

    mergeSort(array)

    print("Sorted array is: ")
    printList(array)




Code Improve:
Here are some tips to improve the provided Merge Sort implementation for better readability, efficiency, and maintainability:

1. mergeSort function

    Suggestions:

        ⦿ Function name consistency:
            
              The function name mergeSort could be more consistent with PEP8 naming conventions. While not strictly necessary, it’s often recommended to use a name like merge_sort (lowercase with underscores).

        ⦿ Avoid redundant array slicing: 
            
            Instead of using array[:r] and array[r:] for splitting the array, which creates copies of the subarrays, you can split the array with index manipulation to save memory, especially for large datasets.

        ⦿ In-place merge: 
            
            Since the merge operation creates new lists (L and M), it's not memory-efficient for large datasets. You can improve this by merging the elements directly into the original array.

        ⦿ Adding comments and docstrings: 
            
            A brief explanation of the function’s purpose will help improve the readability of the code.

    Improved mergeSort function:

        def merge_sort(array):
            if len(array) > 1:
                mid = len(array) // 2  # Find the middle point
                left_half = array[:mid]
                right_half = array[mid:]

                merge_sort(left_half)  # Recursively sort the left half
                merge_sort(right_half)  # Recursively sort the right half

                # Merge the sorted halves
                i = j = k = 0

                while i < len(left_half) and j < len(right_half):
                    if left_half[i] < right_half[j]:
                        array[k] = left_half[i]
                        i += 1
                    else:
                        array[k] = right_half[j]
                        j += 1
                    k += 1

                # Copy any remaining elements from the left half
                while i < len(left_half):
                    array[k] = left_half[i]
                    i += 1
                    k += 1

                # Copy any remaining elements from the right half
                while j < len(right_half):
                    array[k] = right_half[j]
                    j += 1
                    k += 1

2. printList function

    Suggestions:

        ⦿ Efficiency: 
            
            The printList function iterates over the array and prints each element individually. Using print(" ".join(map(str, array))) is more efficient for printing all elements in a single line.

        ⦿ Improved readability:
            
              The use of a more concise approach to printing makes the function clearer and more efficient.

    Improved printList function:
        def print_list(array):
            print(" ".join(map(str, array)))  # Convert each element to string and join with space

3. Main Program Block (if __name__ == '__main__':)

    Suggestions:
    
        ⦿ More descriptive variable names:
            
              The variable name array is fine, but a more specific name such as unsorted_array or numbers can help clarify its purpose.

        ⦿ Test with dynamic input: 
            
            It would be beneficial to allow the user to input an array for sorting, so the program becomes more flexible.

        ⦿ Remove hardcoded array: 
            
            Instead of directly initializing the array inside the code, prompt the user to input it or use a default array for simplicity in the demonstration.

    Improved main program block:

        if __name__ == '__main__':
            # Optionally, allow user input for array or use a default one
            try:
                numbers = list(map(int, input("Enter numbers to sort (space-separated): ").split()))
            except ValueError:
                print("Invalid input. Please enter integers only.")
                exit()

            print("Unsorted array:")
            print_list(numbers)  # Print the original array
            merge_sort(numbers)  # Sort the array using merge_sort
            print("\nSorted array:")
            print_list(numbers)  # Print the sorted array

4. General Recommendations

    ⦿ PEP8 Naming Conventions: 
        
        Ensure function names follow the PEP8 convention of using lowercase words separated by underscores (e.g., merge_sort instead of mergeSort).

    ⦿ Edge Case Handling:
        
          Consider adding checks for edge cases like empty arrays or arrays with only one element. In such cases, the array is already sorted, so the function can return early.

    ⦿ Documentation:
        
          Add docstrings to each function explaining what it does. This is especially helpful when working on larger codebases or collaborating with others.

Summary of Improvements

    ⦿ Function Names: 
    
        Follow PEP8 naming conventions (e.g., merge_sort, print_list).

    ⦿ Memory Efficiency:
        
        Avoid unnecessary array slicing to save memory, especially for large datasets.

    ⦿ Code Readability:

         Simplified the printList function and added more meaningful variable names.

    ⦿ User Input: 
        
        Allow for dynamic input to make the program more flexible and user-friendly.

These improvements will make the code more efficient, readable, and user-friendly while maintaining the core functionality of the merge sort algorithm.


Code snippet:
import cv2
import numpy as np 
import sqlite3
import os
import playsound
import tkinter
from tkinter import messagebox
from tkinter import *

uname =""
conn = sqlite3.connect('database.db')

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

c = conn.cursor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  if faces is ():
    return None

  for (x, y, w, h) in faces:
    cropped_face = img[y:y + h, x:x + w]

  return cropped_face


cap = cv2.VideoCapture(0)

root=Tk()
root.configure(background="white")
# -------------------------------------------
def getInput1():
  global uname
  inputValue = textBox.get("1.0","end-1c")
  if inputValue != "":
    uname = inputValue
  else:
    print('Please enter the name. it is mandatory')
    root.withdraw()
    messagebox.showwarning("Warning", "Please enter the name. It is mandatory field...")
    exit()
  print(inputValue)
  root.destroy()

L1 = Label(root, text = 'Enter Your Name : ',font=("times new roman",12))
L1.pack()
textBox = Text(root, height=1, width = 20,font=("times new roman",12),bg="Pink",fg='Red')
textBox.pack()
textBox.focus()
buttonSave = Button(root, height = 1, width = 10,font=("times new roman",12), text = "Save", command = lambda:getInput1())
buttonSave.pack()
# buttonQuit = Button(root, height = 1, width = 10,font=("times new roman",12), text = "Exit", command = lambda:quit())
# buttonQuit.pack()
mainloop()
# -------------------------------------------
# uname = input("Enter your name: ")
c.execute('INSERT INTO users1 (name) VALUES (?)', (uname,))
uid = c.lastrowid
count = 0
while True:
  ret, frame = cap.read()
  # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  # for (x,y,w,h) in faces:
  #   sampleNum = sampleNum+1
  #   cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
  #   cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
  #   cv2.waitKey(100)
  # cv2.imshow('img',img)
  # cv2.waitKey(1);
  # if sampleNum >= 20:
  # break
  if face_extractor(frame) is not None:
    count += 1
    face = cv2.resize(face_extractor(frame), (150, 150))
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    file_name_path = 'DataSet/User.'+str(uid)+"."+str(count)+'.jpg'
    # file_name_path1 = 'DataSet/frame'+str(count)+'.jpg'

    cv2.imwrite(file_name_path, face)
    # cv2.imwrite(file_name_path1,frame)

    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Face Cropper', face)
  else:
    print("Face not Found")
    pass
  if cv2.waitKey(1) == 13 or count == 20:
    break

print("\nSamples captured successfully...")
playsound.playsound('sound.mp3')

cap.release()
conn.commit()
conn.close()
cv2.destroyAllWindows()

Code Improve:
1. face_extractor function

    Suggestions:

        ⦿ Handle edge cases: 
            
            Instead of returning None when no faces are detected, raise an exception or return a more informative value like [] to indicate no faces were found.

        ⦿ Remove unnecessary variables: 
            
            The cropped_face variable is not needed, as we can directly return the cropped face in the for-loop.

        ⦿ Efficient gray scale conversion: 
            
            You can check if the image is already in grayscale to avoid redundant processing.

    Improved code:
        def face_extractor(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img  # Convert only if not already grayscale
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
            if len(faces) == 0:  # Improved check for no faces found
                return []  # Return an empty list instead of None
    
            # Assuming the first detected face is the correct one
            (x, y, w, h) = faces[0]  # Using the first face if multiple are detected
            return img[y:y + h, x:x + w]  # Return the cropped face directly

2. getInput1 function

    Suggestions:

        ⦿ Error handling: 
            
            It's better to handle input errors gracefully. Use proper validation and ensure that the user has entered a name.

        ⦿ Improve variable scope: 
            
            The global variable uname is being set inside the function. While this is fine for small scripts, you might want to pass it explicitly or return it.

    Improved code:

        def getInput1():
            inputValue = textBox.get("1.0", "end-1c").strip()  # Strip any extra spaces
            if not inputValue:
                messagebox.showwarning("Warning", "Please enter the name. It is a mandatory field...")
                return None  # Return None if input is invalid
            return inputValue  # Return the user input for further processing

3. Main code block for capturing faces

    Suggestions:

        ⦿ Avoid hardcoding paths: 
        
            It's good practice to avoid hardcoding file paths. Use variables to construct paths dynamically.

        ⦿ Refactor repetitive code: 
            
            A significant portion of the code is devoted to capturing faces, which can be made cleaner and more readable by extracting the logic into smaller functions.

        ⦿ Use meaningful variable names: 
             
             Rename variables like count and uid to be more descriptive.

        ⦿ Break out of the infinite loop early if desired conditions are met: Using while True is less readable. A better alternative is to check for the desired conditions upfront.

    Improved code:

        def capture_face_image(frame, uid, count):
            face = face_extractor(frame)
            if face:
                count += 1
                resized_face = cv2.resize(face, (150, 150))
                file_name_path = f'DataSet/User.jpg'  # Dynamically create the file path
                cv2.imwrite(file_name_path, resized_face)
                cv2.putText(resized_face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow('Face Cropper', resized_face)
            return count

        # Driver program
        def main():
            uname = getInput1()  # Retrieve user input
            if uname is None:
                exit()  # Exit if the name was not entered

            c.execute('INSERT INTO users1 (name) VALUES (?)', (uname,))
            uid = c.lastrowid
            count = 0

            cap = cv2.VideoCapture(0)
            while count < 20:
                ret, frame = cap.read()
                if not ret:
                    break
                count = capture_face_image(frame, uid, count)

            print("\nSamples captured successfully...")
            playsound.playsound('sound.mp3')
            cap.release()
            conn.commit()
            conn.close()
            cv2.destroyAllWindows()

        if __name__ == "__main__":
            main()

4. Miscellaneous Improvements

    ⦿ Database connection management:
     
         You should always close the database connection in a finally block to ensure that it is closed even if an error occurs.

    ⦿ Resource handling: 
    
        It's best to handle video capture release and database connection in a finally block to ensure they are closed correctly.

    ⦿ Optimizing GUI logic: 
    
        The GUI should not block the rest of the code, so consider using threading or separating the logic.

General Suggestions:

    ⦿ Logging: 

        Use logging instead of print statements for better traceability and to handle production-level debugging.

    ⦿ External libraries: 

        Consider adding requirements and handling missing libraries in a requirements.txt file.

    ⦿ Refactoring the UI part: 

        GUI and program logic could be split into separate files for better maintainability.

These changes make the code more modular, readable, and easier to maintain while improving error handling and performance in some places.

"""
    prompt_improve=f"""
    Please provide suggestions to improve the following Python code. For each function in the code, offer one or more improvements, optimizations, or best practices. After each function's improvements.
    {code_improve}
    ####
    Code Snippet is shared below, delimited with triple backticks:
    ####
    ```
    {code_snippets}
    ```
"""
    return prompt_improve

@st.cache_data(show_spinner=False)
def metadata_code_prompt(code_snippets):
    code_meta="""
Code Snippet:
nterms = int(input("How many terms? "))

# first two terms
n1, n2 = 0, 1
count = 0

# check if the number of terms is valid
if nterms <= 0:
   print("Please enter a positive integer")
# if there is only one term, return n1
elif nterms == 1:
   print("Fibonacci sequence upto",nterms,":")
   print(n1)
# generate fibonacci sequence
else:
   print("Fibonacci sequence:")
   while count < nterms:
       print(n1)
       nth = n1 + n2
       # update values
       n1 = n2
       n2 = nth
       count += 1



Code MetaData:

Metadata Summary

Time Complexity:

    1.Iterative: O(n)

        The iterative solution loops through the number of terms, performing constant time operations in each iteration.

        Recursive (Naive): O(2^n)

        The naive recursive solution recalculates Fibonacci numbers multiple times, leading to exponential time complexity.

    2.Recursive (Memoized): O(n)

        With memoization, each Fibonacci number is computed only once and stored, which results in a linear time complexity.

Space Complexity:

    1.Iterative: O(1)
        The iterative approach only uses a fixed number of variables (n1, n2, count), and the space complexity does not depend on the input size.

    2.Recursive (Naive): O(n)
        Due to recursion, the call stack grows in size as deep as n, which leads to a linear space complexity in the case of naive recursion.

    3.Recursive (Memoized): O(n)
        In memoized recursion, the space is used to store previously calculated Fibonacci numbers, resulting in linear space complexity.

Libraries Used:

    No external libraries are used in this code.

    input():

    Documentation: Python input() function

    Description: The input() function reads a line of text entered by the user. The string entered by the user is returned as the output.

    Usage: Used to take input from the user, specifically the number of terms (nterms) for the Fibonacci sequence.
    
Variables:

    nterms:

    Type: Integer
    Description: Stores the number of terms to generate in the Fibonacci sequence as provided by the user.

    n1:

        Type: Integer
        Description: Stores the first term in the Fibonacci sequence (initially 0).

    n2:

        Type: Integer
        Description: Stores the second term in the Fibonacci sequence (initially 1).

    count:

        Type: Integer
        Description: A counter to keep track of how many terms of the Fibonacci sequence have been printed.

    Objects:
        No complex objects are used in the code. The program only manipulates basic data types:
        Integers (nterms, n1, n2, count)




Code Snippet:
def mergeSort(array):
    if len(array) > 1:

        #  r is the point where the array is divided into two subarrays
        r = len(array)//2
        L = array[:r]
        M = array[r:]

        # Sort the two halves
        mergeSort(L)
        mergeSort(M)

        i = j = k = 0

        # Until we reach either end of either L or M, pick larger among
        # elements L and M and place them in the correct position at A[p..r]
        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1

        # When we run out of elements in either L or M,
        # pick up the remaining elements and put in A[p..r]
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1


# Print the array
def printList(array):
    for i in range(len(array)):
        print(array[i], end=" ")
    print()


# Driver program
if __name__ == '__main__':
    array = [6, 5, 12, 10, 9, 1]

    mergeSort(array)

    print("Sorted array is: ")
    printList(array)


Code MetaData:
Metadata Summary

Time Complexity:

    Merge Sort has a time complexity of O(n log n), where n is the number of elements in the array.

        Reason: The array is divided into halves recursively (log n splits), and each split requires a linear time operation (O(n)) to merge the subarrays.

Space Complexity:

    O(n) for merge sort.

        Reason: Merge Sort requires additional space for the left and right subarrays (temporary arrays L and M). The space used grows with the size of the input array.

Libraries Used:

    No external libraries are used in the code. It only uses built-in Python functions and features.

Variables:

    array:
        Type: List
        Description: The input array that needs to be sorted.
    r:
        Type: Integer
        Description: The index where the array is divided into two subarrays (left and right). It is calculated as len(array)//2.
    L:
        Type: List
        Description: The left subarray after dividing the original array.
    M:
        Type: List
        Description: The right subarray after dividing the original array.
    i, j, k:
        Type: Integer
        Description: Used as counters/indices for traversing through arrays L, M, and array, respectively, while merging the elements.

Functions:

    mergeSort(array):
        Purpose: Recursively sorts the input array by splitting it into halves, sorting the halves, and merging them back together.

    Parameters:
        array (List): The array to be sorted.

    Logic:
        The array is divided recursively until each subarray has a single element.
        The merge process then compares the elements of subarrays and places them in order.

    printList(array):
        Purpose: Prints the sorted array after the sorting process is completed.

    Parameters:
        array (List): The sorted array.

    Logic: 
        Iterates over the array and prints each element, separated by a space.

Example Input/Output:
Input:
    array = [6, 5, 12, 10, 9, 1]
Output:
    "Sorted array is: 1 5 6 9 10 12"





Code snippet:
import cv2
import numpy as np 
import sqlite3
import os
import playsound
import tkinter
from tkinter import messagebox
from tkinter import *

uname =""
conn = sqlite3.connect('database.db')

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

c = conn.cursor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  if faces is ():
    return None

  for (x, y, w, h) in faces:
    cropped_face = img[y:y + h, x:x + w]

  return cropped_face


cap = cv2.VideoCapture(0)

root=Tk()
root.configure(background="white")
# -------------------------------------------
def getInput1():
  global uname
  inputValue = textBox.get("1.0","end-1c")
  if inputValue != "":
    uname = inputValue
  else:
    print('Please enter the name. it is mandatory')
    root.withdraw()
    messagebox.showwarning("Warning", "Please enter the name. It is mandatory field...")
    exit()
  print(inputValue)
  root.destroy()

L1 = Label(root, text = 'Enter Your Name : ',font=("times new roman",12))
L1.pack()
textBox = Text(root, height=1, width = 20,font=("times new roman",12),bg="Pink",fg='Red')
textBox.pack()
textBox.focus()
buttonSave = Button(root, height = 1, width = 10,font=("times new roman",12), text = "Save", command = lambda:getInput1())
buttonSave.pack()
# buttonQuit = Button(root, height = 1, width = 10,font=("times new roman",12), text = "Exit", command = lambda:quit())
# buttonQuit.pack()
mainloop()
# -------------------------------------------
# uname = input("Enter your name: ")
c.execute('INSERT INTO users1 (name) VALUES (?)', (uname,))
uid = c.lastrowid
count = 0
while True:
  ret, frame = cap.read()
  # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  # for (x,y,w,h) in faces:
  #   sampleNum = sampleNum+1
  #   cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
  #   cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
  #   cv2.waitKey(100)
  # cv2.imshow('img',img)
  # cv2.waitKey(1);
  # if sampleNum >= 20:
  # break
  if face_extractor(frame) is not None:
    count += 1
    face = cv2.resize(face_extractor(frame), (150, 150))
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    file_name_path = 'DataSet/User.'+str(uid)+"."+str(count)+'.jpg'
    # file_name_path1 = 'DataSet/frame'+str(count)+'.jpg'

    cv2.imwrite(file_name_path, face)
    # cv2.imwrite(file_name_path1,frame)

    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Face Cropper', face)
  else:
    print("Face not Found")
    pass
  if cv2.waitKey(1) == 13 or count == 20:
    break

print("\nSamples captured successfully...")
playsound.playsound('sound.mp3')

cap.release()
conn.commit()
conn.close()
cv2.destroyAllWindows()





Code MetaData :
Time Complexity

    Face Detection:

        Using cv2.CascadeClassifier.detectMultiScale: O(n) per frame, where n is proportional to the number of pixels being analyzed.

    Overall Loop:

        Frame capture and processing: O(n * m), where n is the number of frames captured, and m is the face detection complexity for each frame.

Space Complexity

    O(1): Minimal auxiliary space is used during the loop, as images are processed one frame at a time.

    Storage Requirement:

        O(k): Storing k cropped face images to disk.

Libraries Used

    cv2 (OpenCV):
        Used for computer vision tasks, such as face detection and image manipulation.

        Documentation: OpenCV Python Docs

        Key Methods Used:

            cv2.CascadeClassifier: Loads the Haar cascade classifier for face detection.
            cv2.cvtColor: Converts images to grayscale.
            cv2.resize: Resizes images to a specified dimension.
            cv2.putText: Adds text annotations to images.
            cv2.VideoCapture: Captures live video from the camera.
            cv2.imwrite: Saves the processed image to disk.
            cv2.destroyAllWindows: Closes all OpenCV windows.

    sqlite3:

        Used to interact with a local SQLite database.

        Documentation: SQLite3 Python Docs

        Key Methods Used:
            sqlite3.connect: Connects to the SQLite database.
            cursor.execute: Executes SQL commands.
            cursor.lastrowid: Retrieves the ID of the last inserted record.
            conn.commit: Commits changes to the database.

    os:

        Used for filesystem interactions.

        Documentation: os Python Docs

        Key Methods Used:

            os.makedirs: Creates a directory if it doesn’t exist.
            os.path.exists: Checks for the existence of a directory or file.

    playsound:

        Plays audio files (used for alerting the user when capture completes).

        Documentation: playsound Python Docs

    tkinter:

        Used for GUI creation and user input.

        Documentation: tkinter Python Docs

        Key Methods Used:
            Tk: Creates the main application window.
            Label, Text, Button: GUI widgets for text input and interaction.
            messagebox.showwarning: Displays a warning dialog.

Variables

    uname:
        Type: String
        Purpose: Stores the name entered by the user.

    conn:
        Type: Database Connection Object
        Purpose: Connects to the SQLite database (database.db).

    face_cascade:
        Type: OpenCV CascadeClassifier Object
        Purpose: Detects faces using a Haar cascade file.

    cap:
        Type: VideoCapture Object
        Purpose: Captures live video feed from the webcam.

    count:
        Type: Integer
        Purpose: Tracks the number of face images captured.

    file_name_path:
        Type: String
        Purpose: Stores the path for each saved face image.

Functions

    face_extractor(img):
        Purpose: Detects faces in the provided frame and extracts the cropped face region.
        Parameters:
            img (Image Frame): The input frame captured from the webcam.
        Returns: Cropped face region or None if no face is detected.

    getInput1():
        Purpose: Collects the user’s name through a GUI input box.
        Side Effects: Exits the program if no name is entered

Output

    Captures and saves 20 face images of the user in the dataset folder.

    Displays a live feed with face regions annotated.

    Plays a sound to indicate successful completion.
    
"""

    prompt_meta=f"""
    Provide detailed metadata for the following Python code. Include:

    1.Time Complexity: Analyze the time complexity of the main operations or algorithms in the code.
    2.Space Complexity: Explain the memory usage of the program.
    3.Libraries Used: List the libraries imported in the code, their purpose, and a suggested official link to their official documentation.
    4.Variables and Objects: Provide a list of variables and objects used in the code along with their roles.
    5.Potential Improvements: Suggest ways to optimize or improve the code if applicable.
    ####
    {code_meta}
    ####
    Code Snippet is shared below, delimited with triple backticks:
    ```
    {code_snippets}
    ```
    """
    return prompt_meta

@st.cache_data(show_spinner=False)
def flowchart_code_prompt(code_snippets):
    python_code_example = """
    ----------------------------
    Example 1: Code Snippet
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = bubble_sort(arr)
    print(sorted_arr)

    dot code: !! don't add triple hash and  triple backticks in dote code 
    digraph G {
        node [shape=box];
        
        start [label="Start", shape=circle];
        input [label="Input Array", shape=box];
        outer_loop [label="For i in range(n)", shape=box];
        inner_loop [label="For j in range(n-i-1)", shape=box];
        comparison [label="arr[j] > arr[j+1]?", shape=diamond];
        swap [label="Swap arr[j] and arr[j+1]", shape=box];
        end_inner_loop [label="End Inner Loop", shape=box];
        end_outer_loop [label="End Outer Loop", shape=box];
        return [label="Return Sorted Array", shape=box];
        stop [label="Stop", shape=circle];
        
        start -> input;
        input -> outer_loop;
        outer_loop -> inner_loop;
        inner_loop -> comparison;
        comparison -> swap [label="True"];
        swap -> end_inner_loop;
        comparison -> end_inner_loop [label="False"];
        end_inner_loop -> inner_loop [label="Continue Inner Loop"];
        end_inner_loop -> end_outer_loop [label="End Inner Loop"];
        end_outer_loop -> outer_loop [label="Continue Outer Loop"];
        outer_loop -> return [label="End Outer Loop"];
        return -> stop;
    }


    -----------------------------

    Example 2: Code Snippet
    def find_prime_numbers(n):
        primes = []
        for num in range(2, n+1):
            is_prime = True
            for j in range(2, int(num ** 0.5) + 1):
                if num % j == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
        return primes

    n = 20
    prime_numbers = find_prime_numbers(n)
    print(prime_numbers)

    dot code: !! don't add triple hash and  triple backticks in dote code
    digraph G {
        node [shape=box];

        start [label="Start", shape=circle];
        input [label="Input n", shape=box];
        init_primes [label="Initialize primes list", shape=box];
        outer_loop [label="For num in range(2, n+1)", shape=box];
        init_is_prime [label="Set is_prime = True", shape=box];
        inner_loop [label="For i in range(2, sqrt(num)+1)", shape=box];
        check_divisible [label="num % j == 0?", shape=diamond];
        set_not_prime [label="Set is_prime = False", shape=box];
        end_inner_loop [label="End Inner Loop", shape=box];
        append_prime [label="Append num to primes", shape=box];
        end_outer_loop [label="End Outer Loop", shape=box];
        return [label="Return primes list", shape=box];
        stop [label="Stop", shape=circle];

        start -> input;
        input -> init_primes;
        init_primes -> outer_loop;
        outer_loop -> init_is_prime;
        init_is_prime -> inner_loop;
        inner_loop -> check_divisible;
        check_divisible -> set_not_prime [label="True"];
        set_not_prime -> end_inner_loop;
        check_divisible -> end_inner_loop [label="False"];
        end_inner_loop -> outer_loop [label="Continue Outer Loop"];
        end_inner_loop -> append_prime [label="is_prime is True"];
        append_prime -> end_outer_loop;
        end_outer_loop -> return;
        return -> stop;
    }

    ------------------------------
    """

    prompt = f"""
        Your Task is to act as generater of a dot code for graphiz to generate flowchart.
        I'll give you a Code Snippet.
        Your Job is to generate a dot code for graphiz to generate flowchart
        Few good examples of python code output between #### seperator:
        ####
        {python_code_example}
        ####
        Code Snippet is shared below, delimited with triple backticks:
        ```
        {code_snippets}
        ```
        """
    return prompt

# ----------------------Explainer--------------------
title_conatiner= st.container()
title_conatiner.title("🤖 AI Code Explainer")
title_conatiner.caption(f"\nYour Smart Assistant for Code Understanding.")

input_cont = st.container(border=True)
code_snippets = input_cont.text_area("Enter Your Code Here",height=120,help="Only applicable for Python",placeholder=f"a = 5\nb = 5\nc = a + b\nprint(c)")


button_cont = st.container(border=False)
output_cont = st.container(border=True)
options = ["Explain Code", "Meta Data", "Tips to improve","Flowchart","Show Code"]
selection = input_cont.segmented_control(" ", options, selection_mode="single")

if selection == "Explain Code":
    prompt_for_explain =explain_code_prompt(code_snippets=code_snippets)
    ans = llm_model(prompt=prompt_for_explain)
    output_cont.write(ans)   #(stream_data_explain)

if selection == "Flowchart":    
    prompt_for_flowchart = flowchart_code_prompt(code_snippets=code_snippets)
    ans = llm_model(prompt=prompt_for_flowchart)
    cwd = os.getcwd()
    dot_file_path:str = os.path.join(cwd,"mygraph.dot")
    with open(dot_file_path, "w") as f:
        f.write(ans) #(st.session_state.output_explain_key)

    with open(dot_file_path, 'r') as fr:
        lines = fr.readlines()
        with open(dot_file_path, 'w') as fw:
            for line in lines:
                if line.strip('\n') != "```" and line.strip('\n') != "###" :
                    fw.write(line)
    dot_graph = graphviz.Source.from_file(dot_file_path)
    st.write(dot_graph)
    os.remove(dot_file_path)


if selection == "Meta Data":
    prompt_for_metadata = metadata_code_prompt(code_snippets=code_snippets)
    ans = llm_model(prompt=prompt_for_metadata)
    output_cont.write(ans)#(stream_data_explain)

if selection == "Tips to improve":
    prompt_for_improve = improve_code_prompt(code_snippets=code_snippets)
    ans = llm_model(prompt=prompt_for_improve)
    output_cont.write(ans)#(stream_data_explain)

if selection == "Show Code":
    st.code(code_snippets)