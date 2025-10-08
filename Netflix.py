import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import collections


print("--- Step 1: Libraries Imported Successfully ---")

# Step 2: Load the Dataset
try:
    df = pd.read_csv('netflix1.csv')
    print("\n--- Step 2: Dataset Loaded Successfully ---")
    print("Original Dataset Shape:", df.shape)
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'netflix1.csv' not found. Please ensure the dataset is in the same directory as the script.")
    exit()

# Step 3: Data Cleaning

print("\n--- Step 3: Data Cleaning Initiated ---")

# Replace 'Not Given' with NaN to standardize missing values
df.replace('Not Given', pd.NA, inplace=True)

# Handle Missing Values
# Check for missing values in each column
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing 'director' and 'country' with 'Unknown'
df['director'] = df['director'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# For 'date_added' and 'rating', the missing rows are few. We will drop them for this analysis.
df.dropna(subset=['date_added', 'rating'], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Handle Duplicates
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"\nRemoved {initial_rows - df.shape[0]} duplicate rows.")


# Convert Data Types
# Convert 'date_added' to datetime objects
df['date_added'] = pd.to_datetime(df['date_added'].str.strip())

# Create new columns for 'year_added' and 'month_added'
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

print("\nConverted 'date_added' to datetime and extracted year and month.")
print("Cleaned Dataset Shape:", df.shape)
print("\n Data Cleaning Complete")


# Step 4: Exploratory Data Analysis (EDA)

print("\n--- Step 4: Exploratory Data Analysis (EDA) ---")

# Set global style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.autolayout'] = True


# Content Type Distribution (Movie vs. TV Show)
plt.figure()
type_counts = df['type'].value_counts()
sns.barplot(x=type_counts.index, y=type_counts.values, palette="viridis", hue=type_counts.index, dodge=False, legend=False)
plt.title('Distribution of Content Types on Netflix')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.savefig('content_type_distribution.png')
print("\nGenerated plot: content_type_distribution.png")


#  Content Added Over Time
plt.figure()
content_by_year = df['year_added'].value_counts().sort_index()
sns.lineplot(x=content_by_year.index, y=content_by_year.values, marker='o', color='royalblue')
plt.title('Content Added to Netflix Over the Years')
plt.xlabel('Year Added')
plt.ylabel('Number of Titles Added')
plt.grid(True)
plt.savefig('content_added_over_time.png')
print("Generated plot: content_added_over_time.png")


#  Top 10 Countries with Most Content
plt.figure()
# excluded 'Unknown' from this analysis
top_countries = df[df['country'] != 'Unknown']['country'].value_counts().nlargest(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette='plasma', hue=top_countries.index, dodge=False, legend=False)
plt.title('Top 10 Countries Producing Content on Netflix')
plt.xlabel('Number of Titles')
plt.ylabel('Country')
plt.savefig('top_10_countries.png')
print("Generated plot: top_10_countries.png")


#  Top 15 Directors
plt.figure()
#  excluded 'Unknown' from this analysis
top_directors = df[df['director'] != 'Unknown']['director'].value_counts().nlargest(15)
sns.barplot(x=top_directors.values, y=top_directors.index, palette='magma', hue=top_directors.index, dodge=False, legend=False)
plt.title('Top 15 Directors on Netflix by Number of Titles')
plt.xlabel('Number of Titles')
plt.ylabel('Director')
plt.savefig('top_15_directors.png')
print("Generated plot: top_15_directors.png")


#  Popular Genres (Word Cloud and Bar Plot)
# Process genres
genres_list = df['listed_in'].str.split(', ').explode()
genre_counts = collections.Counter(genres_list)

# Word Cloud
wordcloud = WordCloud(
    background_color='white',
    width=1000,
    height=600,
    colormap='viridis'
).generate_from_frequencies(genre_counts)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Netflix Genres')
plt.savefig('genres_wordcloud.png')
print("Generated plot: genres_wordcloud.png")

# Bar Plot for Top 15 Genres
plt.figure(figsize=(12, 8))
top_genres = pd.Series(genre_counts).nlargest(15)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='cubehelix', hue=top_genres.index, dodge=False, legend=False)
plt.title('Top 15 Most Popular Genres on Netflix')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.savefig('top_15_genres.png')
print("Generated plot: top_15_genres.png")


#  Distribution of Ratings
plt.figure(figsize=(14, 8))
sns.countplot(y=df['rating'], order=df['rating'].value_counts().index, palette='crest', hue=df['rating'], dodge=False, legend=False)
plt.title('Distribution of Content Ratings on Netflix')
plt.xlabel('Count')
plt.ylabel('Rating')
plt.savefig('ratings_distribution.png')
print("Generated plot: ratings_distribution.png")

print("\n--- EDA Complete ---")
print("All visualization plots have been saved to the current directory.")


# Step 5: Conclusion and Insights

print("\n--- Step 5: Conclusion and Insights ---")
print("""
In this project, I have successfully performed a comprehensive analysis of the Netflix dataset.

1.  Data Cleaning
2.  Exploratory Data Analysis
    
These insights provide a clear picture of Netflix's content strategy and library composition over the years.
""")


# Step 6: Next Steps

print("\n--- Step 6: Next Steps ---")
print("""
This analysis serves as a great starting point. Future work could expand on this in several ways:

1.  **Feature Engineering**: Create new features to deepen the analysis. For example:
    -   Extract the primary country from the 'country' column for multi-country productions.
    -   Separate 'duration' into numerical columns for minutes (for movies) and seasons (for TV shows).
    -   Count the number of cast members for each title.

2.  **Machine Learning**: Use the cleaned data to build predictive models.
    -   **Recommendation System**: Build a content-based recommendation engine that suggests similar movies/shows
      based on genre, director, and cast.
    -   **Trend Prediction**: Forecast future content trends based on historical data.

3.  **Advanced Visualization**: Create interactive dashboards using tools like Plotly, Dash, or Tableau to allow
    for more dynamic and detailed exploration of the data.
""")

# Display all the generated plots at the end
plt.show()

#Step 7: Content-Based Recommendation Engine

print("\n--- Step 7: Building a Content-Based Recommendation Engine ---")

# I built a recommendation engine based on 'director' and 'listed_in' (genres).

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a new dataframe for the recommendation engine to keep the original clean.
df_rec = df.copy()

# For the recommendation engine, we'll combine key features into a single string.
df_rec['features'] = df_rec['director'].fillna('') + ' ' + df_rec['listed_in'].fillna('')

# Text Vectorization using TF-IDF
# TF-IDF (Term Frequency-Inverse Document Frequency) will convert our text features into a matrix of numerical values.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_rec['features'])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

#  Compute Cosine Similarity
# We will compute the cosine similarity between all items based on the TF-IDF matrix.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Cosine Similarity Matrix Shape:", cosine_sim.shape)

# Create a mapping from title to DataFrame index
indices = pd.Series(df_rec.index, index=df_rec['title']).drop_duplicates()

# Create the Recommendation Function
def get_recommendations(title, cosine_sim=cosine_sim, data=df_rec, indices=indices):
    """
    This function takes a title and returns the top 10 most similar movies/shows.
    """
    try:
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies (excluding the movie itself, which is at index 0)
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return data['title'].iloc[movie_indices]
    except KeyError:
        return f"Title '{title}' not found in the dataset."

#  Get Recommendations Interactively from the User
print("\n--- Interactive Recommendation Engine ---")
print("Enter a movie or TV show title to get recommendations (or type 'exit' to quit).")

while True:
    user_title = input("\nEnter a title: ")

    if user_title.lower() == 'exit':
        break

    recommendations = get_recommendations(user_title)

    # Check if the result is a string (error message) or a pandas Series (recommendations)
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print(f"\n--- Top 10 Recommendations for '{user_title}' ---")
        # Use to_string() to avoid truncation and display cleanly
        print(recommendations.to_string())


print("\n--- Project Complete ---")

