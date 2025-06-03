# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor

# # Load cleaned dataset
# df_cleaned = pd.read_csv("cleaned_recipes.csv")

# # Spoonacular API Key (Replace with your own)
# API_KEY = ""

# # Function to fetch Non-Veg recipes from Spoonacular
# def fetch_non_veg_recipes(calories, protein, carbs, fats):
#     url = "https://api.spoonacular.com/recipes/complexSearch"
#     params = {
#         "apiKey": API_KEY,
#         "maxCalories": int(calories),
#         "minProtein": int(protein * 0.9),
#         "maxProtein": int(protein * 1.1),
#         "minCarbs": int(carbs * 0.9),
#         "maxCarbs": int(carbs * 1.1),
#         "minFat": int(fats * 0.9),
#         "maxFat": int(fats * 1.1),
#         "number": 5  # Get 5 recipes
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         return response.json().get("results", [])
#     return []

# # Streamlit UI
# st.title("AI Meal Planner")

# # User Inputs
# st.sidebar.header("User Details")
# height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
# weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
# age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
# gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
# activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])
# goal = st.sidebar.selectbox("Goal", ["Maintain weight", "Lose weight", "Gain weight"])
# diet_preference = st.sidebar.selectbox("Diet Preference", ["Veg", "Non-Veg", "Vegan"])

# # BMR Calculation
# def calculate_bmr(weight, height, age, gender):
#     if gender == "Male":
#         return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
#     else:
#         return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

# def get_daily_calories(bmr, activity_level, goal):
#     activity_multipliers = {
#         "Sedentary": 1.2,
#         "Lightly active": 1.375,
#         "Moderately active": 1.55,
#         "Very active": 1.725,
#         "Super active": 1.9
#     }
    
#     daily_calories = bmr * activity_multipliers.get(activity_level, 1.2)
    
#     if goal == "Lose weight":
#         daily_calories -= 500
#     elif goal == "Gain weight":
#         daily_calories += 500
    
#     return daily_calories

# bmr = calculate_bmr(weight, height, age, gender)
# daily_calories = get_daily_calories(bmr, activity_level, goal)
# st.sidebar.write(f"### Recommended Daily Calories: {int(daily_calories)} kcal")

# # Macronutrient Preferences
# st.sidebar.header("Macronutrient Preferences")
# fats = st.sidebar.slider("Fat (g)", 0, 100, 50)
# carbs = st.sidebar.slider("Carbs (g)", 0, 300, 150)
# protein = st.sidebar.slider("Protein (g)", 0, 200, 75)

# # Filter recipes based on dietary preference
# if diet_preference == "Veg":
#     df_filtered = df_cleaned[~df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton", case=False, na=False)]
# elif diet_preference == "Non-Veg":
#     df_filtered = df_cleaned[df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton", case=False, na=False)]
# else:  # Vegan
#     df_filtered = df_cleaned[~df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton|dairy|cheese|milk|butter", case=False, na=False)]

# # Apply KNN Model for Veg/Vegan Diets (Only if Recipes Exist)
# if diet_preference != "Non-Veg" and not df_filtered.empty:
#     features = ["fat_g", "carbs_g", "protein_g", "estimated_calories"]
#     df_filtered = df_filtered.dropna(subset=features)
    
#     if not df_filtered.empty:
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(df_filtered[features])
        
#         knn_model = KNeighborsRegressor(n_neighbors=10, metric='euclidean')
#         knn_model.fit(X_scaled, df_filtered[features])

#         user_input = np.array([[fats, carbs, protein, daily_calories]])
#         user_scaled = scaler.transform(user_input)
#         distances, indices = knn_model.kneighbors(user_scaled)

#         # Display Veg/Vegan Recommendations
#         st.header("Recommended Recipes")
#         for idx in indices[0]:
#             recipe = df_filtered.iloc[idx]
#             st.subheader(recipe["recipe_name"])
#             st.write(f"**Calories:** {recipe['estimated_calories']} kcal")
#             st.write(f"**Fats:** {recipe['fat_g']} g | **Carbs:** {recipe['carbs_g']} g | **Protein:** {recipe['protein_g']} g")
#             st.image(recipe["img_src"], width=300)
#             st.write(f"**Ingredients:** {recipe['ingredients']}")
#             st.write(f"**Recipe Instructions:** {recipe.get('directions', 'Instructions not available')}")
#     else:
#         st.warning("No suitable recipes found in the dataset.")

# # Handle Non-Veg Recipe Fetching via API
# if diet_preference == "Non-Veg" and df_filtered.empty:
#     st.warning("No Non-Veg recipes found locally. Fetching online recommendations...")
#     api_recipes = fetch_non_veg_recipes(daily_calories, protein, carbs, fats)
    
#     if api_recipes:
#         for recipe in api_recipes:
#             st.subheader(recipe["title"])
#             st.image(recipe["image"], width=300)
#             st.write(f"[View Recipe](https://spoonacular.com/recipes/{recipe['title'].replace(' ', '-')}-{recipe['id']})")
#     else:
#         st.error("No recipes found even from external sources. Try adjusting your preferences.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Load cleaned dataset
df_cleaned = pd.read_csv("cleaned_recipes.csv")

# Function to calculate BMR using Mifflin-St Jeor Equation
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

# Adjust BMR based on activity level
def get_daily_calories(bmr, activity_level, goal):
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Super active": 1.9
    }
    
    daily_calories = bmr * activity_multipliers.get(activity_level, 1.2)
    
    if goal == "Lose weight":
        daily_calories -= 500
    elif goal == "Gain weight":
        daily_calories += 500
    
    return daily_calories

# Streamlit UI
st.title("AI Meal Planner")

# User Inputs
st.sidebar.header("User Details")
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=100)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=30)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=10)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])
goal = st.sidebar.selectbox("Goal", ["Maintain weight", "Lose weight", "Gain weight"])
diet_preference = st.sidebar.selectbox("Diet Preference", ["Veg", "Non-Veg", "Vegan"])

# Calculate recommended daily calories
bmr = calculate_bmr(weight, height, age, gender)
daily_calories = get_daily_calories(bmr, activity_level, goal)
st.sidebar.write(f"### Recommended Daily Calories: {int(daily_calories)} kcal")

# User Macro Preferences
st.sidebar.header("Macronutrient Preferences")
fats = st.sidebar.slider("Fat (g)", 0, 100, 50)
carbs = st.sidebar.slider("Carbs (g)", 0, 300, 150)
protein = st.sidebar.slider("Protein (g)", 0, 200, 75)

# Filter recipes based on dietary preference
if diet_preference == "Veg":
    df_filtered = df_cleaned[~df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False)]
elif diet_preference == "Non-Veg":
    df_filtered = df_cleaned[df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False)]
else:  # Vegan
    df_filtered = df_cleaned[~df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton|dairy|cheese|milk|butter|tuna|salmon", case=False, na=False)]

# Feature scaling for KNN regression
features = ["fat_g", "carbs_g", "protein_g", "estimated_calories"]
df_filtered = df_filtered.dropna(subset=features)

if df_filtered.empty:
    st.warning("No recipes found matching your criteria. Please adjust your selections.")
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[features])
    
    # Use KNeighborsRegressor instead of NearestNeighbors
    knn_model = KNeighborsRegressor(n_neighbors=10, metric='euclidean')
    knn_model.fit(X_scaled, df_filtered[features])  # Training the model

    # Predict the closest recipes
    user_input = np.array([[fats, carbs, protein, daily_calories]])
    user_scaled = scaler.transform(user_input)
    distances, indices = knn_model.kneighbors(user_scaled)

    # Get recommended recipes (ordered by distance)
    st.header("Recommended Recipes")
    for idx in indices[0]:  # Keep the closest ones in order
        recipe = df_filtered.iloc[idx]
        st.subheader(recipe["recipe_name"])
        st.write(f"**Calories:** {recipe['estimated_calories']} kcal")
        st.write(f"**Fats:** {recipe['fat_g']} g | **Carbs:** {recipe['carbs_g']} g | **Protein:** {recipe['protein_g']} g")
        st.image(recipe["img_src"], width=300)
        st.write(f"**Ingredients:** {recipe['ingredients']}")
        st.write(f"**Recipe Instructions:** {recipe.get('directions', 'Instructions not available')}")
