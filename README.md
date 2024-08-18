<p align="center">
  <img src="images\Hub_title.png" alt="[YOUR_IMAGE_ALT]">
</p>

<p align="center">
  <img src="images\Screenshot 2024-08-08 115134.png" alt="">
</p>

### Overview
Tourism is a thriving industry in Kenya, and travelers often face the challenge of choosing the right destinations for their trips. Our project aims to address this problem by creating a recommendation system that assists users in discovering personalized tourist destinations in the country.

### Problem Statement

Travelers often struggle to choose the most suitable tourist destinations for their trips. With an overwhelming number of options available, personalized recommendations are crucial. Our project aims to address this challenge by creating a recommendation system that suggests relevant destinations in Kenya based on user preferences and historical interactions.

Beneficiaries include Travelers, Tourism Agencies and Local Business Owners.

### Objectives:

- "Build a collaborative filtering model to recommend destinations."
- "Reduce cold-start problem by incorporating content-based features."
- "Model Recall score ≥ 80%"
- "Model Accuracy ≥ 80%"

<p align="center">
  <img src="images\Screenshot 2024-08-08 115159.png" alt="">
</p>

### Data Sources and Relevance
- The dataset was scraped using the open source **APIFY Tripadvisor Scraper**.
- It contains information about tourist destinations, including their names, categories, ratings, review counts, images, and other relevant features.
- The data's relevance lies in its ability to help us recommend destinations to travelers based on their preferences and historical interactions.

### Data Limitations
- **Missing Values**: Some entries lack ratings, images, or price information.
- **Limited Price Data**: Only 1487 entries have price-related details.
- **Data Quality**: Ensure data quality and handle missing values appropriately.

## Model
We propose building a hybrid recommendation system that combines collaborative filtering and content-based approaches. Success metrics include accuracy, recall and precision scores.


## Conclusion
Our project has significant implications for travelers, tourism agencies, and local businesses. By solving this problem, we contribute to enhancing travel experiences and promoting local economies.

## Repository Structure



├── SafariHub_app/            # Django web app deployement
│   
├── data/                     # Additional data files
│  
├── images/                   # Images for Django web app
│   
├── README.md                 # README.md 
│  
├── SAFARIHUB.docx            # SAFARIHUB report document
│   
├── SafariHub.pdf             # SAFARIHUB report in PDF format
│   
└── safarihub.ipynb           # Jupyter notebook file
  
└── .ppt file                 # Presentation
   

