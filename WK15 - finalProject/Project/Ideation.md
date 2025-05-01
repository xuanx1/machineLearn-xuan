# Milestone 1 - Ideation
encode -> normalize -> train -> evaluate

## Concept 1 
### Wildfire Risk Prediction in the US

### Data Sources:

https://firms.modaps.eosdis.nasa.gov/

https://www.ncdc.noaa.gov/

https://www.usgs.gov/core-science-systems/national-land-imaging-program/land-cover


### Data Processing:

-Merge datasets on location and date (fire reports + weather)

-Encode vegetation types and soil moisture levels numerically

-Normalize temperature, humidity, and wind speed


### 80/20 Train/Test Split and Train


### Evaluation:

-Calculate accuracy of risk classification using Precision-Recall scores

-Feature weightage (e.g., temperature vs. wind speed)

-Geo-Heatmaps to visualise high-risk areas


## Concept 2
### Fragrance Creation and Ingredient / Perfume Blending

### Data Sources:

https://www.kaggle.com/datasets/kanchana1990/perfume-e-commerce-dataset-2024

https://www.kaggle.com/datasets/nandini1999/perfume-recommendation-dataset?resource=download

https://www.kaggle.com/code/foolwuilin/scent-notes-for-different-sexual-categories/input

https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset


### Data Processing:
-Merge datasets on perfume name and ingredients and notes

-Explore collinearity between variables

-Encode perfume names, ingredients/notes, or gender

-Normalise price, rating, and year

-Create a "blend level" based on ingredient similarity + rating difference


### 80/20 Train/Test Split and Train


### Evaluation:

-Calculate accuracy using RMSE

-Visualize as a pair recommendation - choosing 1 perfume will suggest few other perfumes that goes well with it

-Predict user preference based on notes and fragrances



