import os, shutil, random

MAX_IMAGES_PER_CATEGORY = 1000
BASE_DIR = 'dataset'
SELECT_DIR = 'selected_images'
CATEGORIES = [
    'apple(Organik)', 'banana(Organik)', 'battery(B3)', 'cardboard(Anorganik)', 'compost(Organik)',
    'cucumber(Organik)', 'glass(B3)', 'metal(Anorganik)', 'orange(Organik)', 'paper(Anorganik)',
    'plastic(Anorganik)', 'potato(Organik)', 'tomato(Organik)'
]



# Create the output directory if it doesn't exist
os.makedirs(SELECT_DIR, exist_ok=True)

# Select images for each category
for category in CATEGORIES:
    category_path = os.path.join(BASE_DIR, category)
    if os.path.exists(category_path):
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        selected_images = random.sample(images, min(MAX_IMAGES_PER_CATEGORY, len(images)))
        
        # Create a subdirectory for each category in the output directory
        category_output_dir = os.path.join(SELECT_DIR, category)
        os.makedirs(category_output_dir, exist_ok=True)
        
        # Copy the selected images to the category subdirectory in the output directory
        for img_path in selected_images:
            shutil.copy(img_path, category_output_dir)

# Check the number of selected images per category
selected_counts = {category: len(os.listdir(os.path.join(SELECT_DIR, category))) for category in CATEGORIES}
