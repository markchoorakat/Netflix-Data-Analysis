# Netflix-Data-Analysis
This project involves loading, cleaning, analyzing, and visualizing data from a Netflix dataset. We'll use Python libraries like Pandas, Matplotlib, and Seaborn to work through the project. The goal is to explore the dataset, derive insights, and prepare for title recommendation system.

This report details the process and findings of an in-depth analysis of the Netflix content dataset, spanning titles added up to 2021. The primary objectives of this project were to clean and preprocess the raw data, perform exploratory data analysis (EDA) to uncover trends and insights, and develop a functional content-based recommendation engine.

The project successfully utilized a suite of Python libraries, including Pandas for data manipulation, Matplotlib and Seaborn for data visualization, and Scikit-learn for building the machine learning-based recommendation model. Key insights were derived regarding Netflix's content strategy, geographical focus, and popular genres. The project culminates in an interactive recommendation system that suggests similar titles to a user based on shared directors and genres.
2. Data Cleaning and Preprocessing

A clean and well-structured dataset is crucial for accurate analysis. The initial dataset (netflix1.csv) consisted of 8,790 records. The following data cleaning and preprocessing steps were performed:

   Handling Missing Values: A significant number of records had missing values in the director (2,588) and country (287) columns. To retain these records for analysis, missing values were filled with the placeholder "Unknown". A small number of rows with missing date_added and rating were dropped to ensure data integrity.

   Duplicate Removal: The dataset was checked for duplicate entries, and any identical rows were removed to prevent skewed analysis.

   Data Type Conversion: The date_added column was converted from a string format to a proper datetime object. This enabled time-series analysis, from which new features like year_added and month_added were extracted.

After cleaning, the resulting dataset was robust and ready for exploratory analysis.
3. Exploratory Data Analysis (EDA) and Key Insights

Data visualization was employed to explore the dataset and identify significant patterns.

  Content Mix (Movies vs. TV Shows): The analysis revealed that Netflix's library is predominantly composed of movies, which outnumber TV shows significantly. This suggests a strategy focused on single-view content over serialized productions.

  Content Growth Over Time: A line chart of content added per year showed a dramatic surge in library growth, particularly between 2016 and 2019. This period marks Netflix's aggressive global expansion and investment in original content.

  Geographic Distribution: The United States is the largest single producer of content available on the platform, followed by India and the United Kingdom. This highlights the primary markets Netflix caters to.

  Popular Genres and Directors: The most common genres are "International Movies," "Dramas," and "Comedies," indicating a focus on universally appealing content. A word cloud visualization provided an intuitive look at genre popularity. The analysis also identified the most prolific directors on the platform.

  Audience Ratings: The most prevalent content rating is TV-MA (Mature Audiences), followed by TV-14. This suggests that a large portion of the Netflix catalog is targeted towards adults and older teenagers.

4. Content-Based Recommendation Engine

A key outcome of this project was the development of an interactive recommendation system to enhance user experience.

   Methodology: A content-based filtering approach was used. This method recommends items based on their intrinsic properties. For this project, the features used to determine similarity were the director and the genres (listed_in) of each title.

   Implementation:

   Feature Combination: The director and genre information for each title were combined into a single text string.

   TF-IDF Vectorization: The Scikit-learn TfidfVectorizer was used to convert this collection of text strings into a numerical matrix. This technique assigns higher weights to terms that are rare and therefore more descriptive (e.g., a specific director's name) and lower weights to common terms (e.g., the genre "Dramas").

   Cosine Similarity: The cosine similarity metric was then computed across the TF-IDF matrix. This produces a score between 0 and 1, indicating how similar any two titles are based on their shared features.

   Interactive System: The final script implements a user-friendly command-line interface that prompts a user to enter a movie or TV show title. It then returns a list of the top 10 most similar titles from the dataset.

5. Conclusion and Future Work

This project successfully demonstrates a complete data analysis workflow, from cleaning raw data to deploying a functional machine learning model. The analysis provided a clear overview of the Netflix content library, revealing strategic insights into its growth and composition.
