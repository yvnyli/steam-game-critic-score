# MLZoomcamp_Project2
Second project for ML Zoomcamp 2025

## Dataset:

### **Steam Games Dataset** ([link](https://huggingface.co/datasets/FronkonGames/steam-games-dataset))

This dataset contains information about games published on the largest PC game platform, Steam. 

There are **83560 unique games** (rows) with their basic information such as name, publisher, language, and platform; marketing information such as description, genres, tags, header image; as well as rating and reviews.

An interesting thing is that only **a very small portion (5%) of the games received a Metacritic score**. Some of this might be due to a mismatch in data scraping. But given that there are only ~6K game ratings on Metacritic, vs. ~83K games in the dataset, it is true that most games have not gained attention of game critics. 

However, not being rated does not necessarily mean that a game is low quality or not worth playing. It might be due to low budget in marketing or the publisher being a smaller profile one. In other words, there could be **hidden gems** in the vast majority of games that flew under the radar.

## Modeling Problem:

Therefore, the goal of the project is to build a model that can **predict Metacritic score**, in the hopes to **find underrated great games**.

## Modeling Strategy:

The 3910 games with ground-truth Metacritic scores are used for training (tuning), validation and testing. I will try different classes of models and tune their hyperparameter. The winning model based on validation accuracy will be used to predict Metacritic scores on games that don't have it. The accuracy of the predictions should be similar to the accuracy on the test set.

Optinal: train a classifier for probability of having Metacritic score. The idea is, predicted score might be reliable for games similar to the training games, but unreliable for games that are more different. The probability measures this similarity.
