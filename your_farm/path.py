import os

model_dir = "C:/Users/Alif Osman Otoo/Desktop/your_farm/converted_savedmodel/model.savedmodel"
model_file = os.path.join(model_dir, "saved_model.pb")

if os.path.exists(model_dir) and os.path.isdir(model_dir):
    print(f"Directory exists: {model_dir}")
else:
    print(f"Directory does not exist: {model_dir}")

if os.path.exists(model_file):
    print(f"File exists: {model_file}")
else:
    print(f"File does not exist: {model_file}")

if os.access(model_file, os.R_OK):
    print(f"File is readable: {model_file}")
else:
    print(f"File is not readable: {model_file}")
