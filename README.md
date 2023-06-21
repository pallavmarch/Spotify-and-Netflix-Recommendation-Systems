# Spotify Recommendation System
Creating a Spotify Recommendation System using the following steps:

1. Imported the required CSV file.
2. Verified the columns present in the dataset, including the identification of any blanks/NA values and the count of unique values in the 'Genre' and other columns.
3. Filtered the dataset to retain only the relevant columns necessary for further analysis.
4. Converted the values in the 'Loudness.dB' column to positive values and updated the column accordingly.

Popularity Graphs:
5. Plotted a bar graph using the dataset to represent the popularity of each genre among the songs present. The analysis revealed that 'Pop-Film' had the highest popularity, followed by the 'K-Pop' genre.
6. Created a new dataset to measure the popularity of artists and generated a table summarizing the results.
7. Produced a second graph to assess popularity, specifically focusing on artists whose mean popularity exceeded 87.

Heatmap:
8. Created a new dataset named 'da1' containing the numerical columns from the original dataset.
9. Plotted a heatmap, which indicated a significant correlation between Loudness and Acousticness.

KMeans:
10. Compiled a list of data types, including 'int64' and 'float64', to be considered for clustering. Selected desired columns and stored them in the 'xy' DataFrame.
11. Initialized an instance of the KMeans algorithm to cluster the data into ten distinct groups, assigning a cluster label to each row.

Recommendation System:
12. Renamed the 'Track.Name' column to 'Trackname'.
13. Developed a class named 'SpotifyRec'.
14. Defined the constructor, which takes the 'rec_data' parameter.
15. Implemented the 'change_data' method to allow updating of the recommendation data after the object has been created.
16. Created the 'get_recomm' method to retrieve song recommendations based on a given song name. It accepts two parameters: 'song_name' and 'amount' (defaulting to 1), indicating the number of recommendations to return.
17. Initialized an empty list called 'distances' to store the calculated distances between the given song and other songs.
18. Selected the data for the given song by filtering the 'rec_data_' DataFrame based on the matching song name, selecting the first result using 'head(1)'. Created a new DataFrame called 'res_data' by excluding the data for the given song from the 'rec_data_' DataFrame.
19. Iterated over the rows of 'res_data' and calculated the distance between the numerical features of the given song and each row in 'res_data', excluding certain non-numerical columns [0, 1, 2, 3, 4, 19]. Appended the calculated distances to the 'distances' list.
20. Added a new column called 'distance' to the 'res_data' DataFrame, containing the calculated distances. Sorted the 'res_data' DataFrame in ascending order based on the 'distance' column.
21. Selected the specified number of recommendations from 'res_data' and returned a DataFrame containing the columns 'Artist.Name', 'Trackname', 'album_name', and 'Genre'.
