<p align="center">
  <img src="images\Hub_title.png" alt="[YOUR_IMAGE_ALT]">
</p>

![django](https://img.shields.io/badge/django-209117?style=for-the-badge&logo=django&logoColor=white) ![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![scikitlearn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)  ![numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)  ![pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) ![nlp](https://img.shields.io/badge/nlp-209117?style=for-the-badge&logo=nlp&logoColor=white)


SafariHub’s recommender system is designed to provide personalized recommendations for tour destinations, hotels, and tour operators in Kenya. By leveraging user preferences and historical interactions, the system aims to assist travelers in making informed choices. This can greatly enhance the overall travel experience for users.



## Table Of Contents


  - [Table Of Contents](#table-of-contents)
    - [Business Understanding](#business-understanding)
        - [Overview](#overview)
        - [Problem Statement](#problem-statement)
        - [Objectives](#objectives)
    - [Data Understanding](#data-understanding)
    - [Modeling](#modeling)
        - [Evaluation](#evaluation)
    - [Deployment](#deployment)
        - [Example Usage](#example-usage)
        - [Installation](#installation)
    - [Contributors](#contributors)
    - [Repository Structure](#repository-structure)



## Business Understanding
### Overview

Tourism is a thriving industry in Kenya, and travelers often face the challenge of choosing the right destinations for their trips. With an overwhelming number of options available, personalized recommendations are crucial. SafariHub’s recommender system aims to address this problem by suggesting relevant tour destinations, hotels, and tour operators in Kenya based on user preferences and historical interactions.

---

### Problem Statement

Travelers struggle to select the most suitable tourist destinations for their trips. By leveraging user data and interactions, our recommendation system will assist users in making informed choices, enhancing their travel experiences. The beneficiaries include travelers, tourism agencies, and local business owners.

---

### Objectives

- Build a collaborative filtering model to recommend destinations
- Reduce cold-start problem by incorporating content-based features
- Model Recall score ≥ 80%
- Model Accuracy ≥ 80%



## Data Understanding

- The dataset was scraped using the open source [APIFY Tripadvisor Scraper](https://apify.com/maxcopell/tripadvisor).
- It contains information about tourist destinations in Kenya, including their names, categories, ratings, review counts, images, and other relevant features.
- The data's relevance lies in its ability to help us recommend destinations to travelers based on their preferences and historical interactions.



## Modeling

### Best Model: Tuned KNN Hotel Recommender

A hybrid recommendation system that combines NLP and content-based approaches to generate similar items.


    Distance Metrics:
    
    Average Mean Squared Error (MSE): 0.3309639568619343
    
    Average Root Mean Squared Error (RMSE): 0.5752946695928395
    
    Average Mean Absolute Error (MAE): 0.41681208879029213
    
    *************************************************************************************
    
    Classification Report:
    
                  precision    recall  f1-score   support
    
     Not Similar       0.93      0.94      0.93       182
         Similar       0.62      0.58      0.60        31
    
        accuracy                           0.89       213
       macro avg       0.78      0.76      0.77       213
    weighted avg       0.88      0.89      0.89       213
    
    *************************************************************************************
    
    Recommended hotels for Diani Sea Lodge:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>rating</th>
      <th>priceRange</th>
      <th>priceLevel</th>
      <th>distances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>Jacaranda Indian Ocean Beach Resort</td>
      <td>4.0</td>
      <td>KES 13,080 - KES 23,207</td>
      <td>Standard</td>
      <td>0.800187</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Medina Palms</td>
      <td>4.5</td>
      <td>KES 30,802 - KES 42,897</td>
      <td>Premium</td>
      <td>0.868753</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Sandies Tropical Village</td>
      <td>4.0</td>
      <td>KES 16,596 - KES 25,598</td>
      <td>Standard</td>
      <td>0.886675</td>
    </tr>
    <tr>
      <th>227</th>
      <td>Silver Palm Spa and Resort</td>
      <td>4.5</td>
      <td>KES 25,176 - KES 30,942</td>
      <td>Standard</td>
      <td>0.891293</td>
    </tr>
    <tr>
      <th>248</th>
      <td>The Charming Lonno Lodge</td>
      <td>5.0</td>
      <td>KES 42,335 - KES 47,257</td>
      <td>Luxury</td>
      <td>0.897824</td>
    </tr>
  </tbody>
</table>
</div>

### Evaluation

- The KNN model exhibits relatively low error metrics (MSE, RMSE, MAE). This suggests that the KNN model is better at minimizing errors when predicting similarity between hotels.

- The KNN model offers recommendations that are more consistent in terms of proximity to the target hotel, which is an essential factor in making relevant recommendations.

>The KNN model therefore is the most suitable option for recommending hotels.



## Deployment
![django](https://img.shields.io/badge/django-209117?style=for-the-badge&logo=django&logoColor=white)

The model was deployed using Django framework.

### Example Usage

The user clicks on an image to generate recommendations.

<p align="center">
    <img src="images\LakeNaivasharecommendation.png" alt="LakeNaivasharecommendation" width="1000" height="550" />
</p>


### Installation

Steps:
```python
git clone https://github.com/Misfit911/SafariHub
```
```python
pip install -r requirements.txt
```
```python
cd SafariHub_app
```
```python
# Windows
python manage.py runserver

# Mac
python3 manage.py runserver
```



## Contributors


- [Bradley Ouko](https://github.com/Misfit911)
- [Lynn Komen](https://github.com/LynnKomen1)
- [Mourine Kitili](https://github.com/kitili)
- [Phinidy George](https://github.com/BigTime5)
- [Nashon Okumu](https://github.com/NashonOkumu)



## Repository Structure


`├── SafariHub_app`           Deployment

`├──  data`                   Data files

`├── images`                 Images for Django web app
  
`├── README.md`                README.md 
 
`├── SAFARIHUB.docx`           Report document
 
`├── SafariHub.pdf`            Report in PDF format
 
`└── safarihub.ipynb`          Jupyter notebook file