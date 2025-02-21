import joblib


model = joblib.load(r"C:\Users\Dell\OneDrive - University of Hertfordshire\Unlocking_Emotions_NLP_Game\Unlocking_Emotions_NLP_Game\sentiment_model.pkl")
vectorizer = joblib.load(r"C:\Users\Dell\OneDrive - University of Hertfordshire\Unlocking_Emotions_NLP_Game\Unlocking_Emotions_NLP_Game\vectorizer.pkl")


def predict_emotion(text):
    input_text = vectorizer.transform([text])
    return model.predict(input_text)[0]


emotion_memory = {"positive": 0, "negative": 0, "neutral": 0}

# **Scene 1
print("\n You wake up in a dense, misty forest. Your head is pounding. A hooded figure steps out from the shadows.")
print("Hooded Figure: 'Are you alright?'")

user_input = input("\nYou: ")
emotion = predict_emotion(user_input)
emotion_memory[emotion] += 1  # Track player's emotion


if emotion == "positive":
    print("\nHooded Figure: '(Smiling) Good! You still have your strength. You must come with me!'")
    scene = "village"
elif emotion == "negative":
    print("\nHooded Figure: '(Concerned) You must have lost your memories… But there's no time to explain!'")
    scene = "castle"
else:
    print("\nHooded Figure: '(Stern) Then let me show you what’s happening.'")
    scene = "battlefront"

# **Scene 2
if scene == "village":
    print("\n You arrive at a small village, where people seem fearful. A ruined temple stands ahead.")
    print("Hooded Figure: 'This temple holds our past. What do you think of it?'")
elif scene == "castle":
    print("\n You are taken to an abandoned castle. A throne sits empty with a sword beside it.")
    print("Hooded Figure: 'This place was once full of life. What do you feel?'")
elif scene == "battlefront":
    print("\n You see warriors battling shadow creatures. The ground shakes beneath you.")
    print("Hooded Figure: 'This war will decide everything. What should we do?'")

user_input = input("\nYou: ")
emotion = predict_emotion(user_input)
emotion_memory[emotion] += 1  # Update memory

# **Scene 3
if emotion_memory["positive"] >= 2:
    print("\nHooded Figure: '(Encouraged) You have a strong heart. We can rebuild this land together!'")
    role = "hero"
elif emotion_memory["negative"] >= 2:
    print("\nHooded Figure: '(Worried) You doubt too much. The darkness will consume you if you let it.'")
    role = "dark path"
else:
    print("\nHooded Figure: '(Neutral) You stand in the middle. That is rare, but dangerous.'")
    role = "wanderer"

# **Scene 4
print("\n A great trial awaits. A choice must be made.")

if role == "hero":
    print("NPC: 'Will you stand and fight for the kingdom?'")
elif role == "dark path":
    print("NPC: 'Will you let the kingdom fall into ruin?'")
else:
    print("NPC: 'Do you even know what you want?'")

user_input = input("\nYou: ")
emotion = predict_emotion(user_input)
emotion_memory[emotion] += 1

# **Scene 5
if emotion_memory["positive"] >= 3:
    print("\n You become the kingdom’s hero and lead them to victory!")
    print(" Ending 1: The Hero’s Journey ")
elif emotion_memory["negative"] >= 3:
    print("\n You let the kingdom fall into darkness, ruling as its shadowy leader.")
    print(" Ending 2: The Tyrant’s Rule ")
else:
    print("\n You walk away, neither saving nor destroying the land.")
    print(" Ending 3: The Wanderer’s Path ")

print("\n Game Over – Thanks for playing!")
