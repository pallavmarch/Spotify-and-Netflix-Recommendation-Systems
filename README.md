# Netflix_recommendation

1.	The data was loaded
2.	The number of rows, columns and types  of values were checked for the columns
3.	The unnecessary columns were removed from the dataset. The 'title','director','cast','country','rating','listed_in','type' columns were kept
4.	The contents from the 'cast' columns were splitted such that each actor's name were a singular row value of the list. The list os values were placed in actors_list 
5.	The duplicate values were removed in the actors_list by using a set and then the values were sorted.
6.	A Binary Matrix was created for the 'cast' column, by checking if each value of the actors_list is present in the 'cast' column of the data or not
7.	The binary list created with respect to the unique actor's name from the actor_list was stored in the 'binary_actors_df' dataframe
8.	The above 4 steps, starting from the contents being splitted to the binary list being stored in a new dataframe- The steps were repeated for the 'director', 'cast', 'country', 'rating' and 'listed_in' columns and the new dataframes created were termed as 'binary_directors_df', 'binary_country_df', 'binary_genre_df' and 'binary_rating_df'.
9.	All the 5 binary list datasets were concatenated together
10.	The Recommendation system would be created now
11.	A function called 'recommender' was created which passes the argument 'search'
12.	Two empty lists were created called cs_list and binary_list, which'll store cosine similarity values and binary values
13.	If the search matches with any value from the 'title' column, the respective index nuber is fetched of the title
14.	Two loops were generated to convert the binary list to 1D Numpy array, termed as Point 1 and Point 2
15.	The dot product was claculated for the Point 1 and Point 2 vectors
16.	The Euclidean norm of point1 and point2 were calculated
17.	Finally the cosine similarity was calculated and was appeneded to the  empty list of cs_list
18.	The data was copied to a new dataframe called 'movies_copy'
19.	A new column was added to the 'movies_copy' called cos_sim which stores the values of the cs_list
20.	A result variable was produced which copied the 'movies_copy' dataframe and sorted it descending  according the cs_list
21.	The top 5 rows are selected with the highest cosine similarity values
22.	If the search value doesn't match with any of the titles present, it's mentioned that the title is not present in the dataset
