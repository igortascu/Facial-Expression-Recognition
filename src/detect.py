from deepface import DeepFace
import os
from random import randint
from helpers import all_class_labels, load_dataset
from more_itertools import take
# Directory containing your images

dataset = load_dataset("assets/train2", flatten=True)

# Loop through each file in the directory

for label, filepath in dataset:
    try:
        # Analyze the image for emotion
        analysis = DeepFace.analyze(img_path=filepath, actions=['emotion'], enforce_detection=False)

    except Exception as e:
     # Analyze the image for emotion
        try:
            analysis = DeepFace.analyze(img_path=filepath, actions=['emotion'], enforce_detection=True)
            with open("detect-face-failed.txt", "a") as f:
                f.write("Could not find face in image: " + filepath + "\n")
        except Exception as e:
            with open("detect-remove.txt", "a") as f:
                f.write(f"removed {filepath}\n")
                # os.remove(filepath)
        continue
    
    finally:
        # Print out the results
        emotions = analysis[0]['emotion']
        print(f"Results for {filepath}:")
        print(emotions)
        choice = max(emotions, key=lambda key: emotions[key])
        if choice == "angry":
            choice = "anger"
        elif choice == "surprise":
            choice = "surprised"
            
        print(label + " != " + choice)
        if label.lower() != choice.lower():
            with open("detect-replace.txt", "a") as f:
                ext = os.path.splitext(filepath)[1]
                new_path = f"assets/train2/{choice.capitalize()}/{choice}-{randint(10000, 999999)}{ext}"
                f.write(f"{filepath} {new_path}\n")
                # os.rename(filepath, new_path)