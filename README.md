# Steam Game Critic Score
Give ~80K games a score similar to Metacritic. See what the model says about your favorite games. Check out a game's score before you buy.

ML Zoomcamp 2025: If you are here for peer review, please check out [Guide_for_evaluators.md](https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/Guide_for_evaluators.md) for a map of where things are. Thank you.

## Dataset:

### **Steam Games Dataset** ([link](https://huggingface.co/datasets/FronkonGames/steam-games-dataset))

This dataset contains information about games published on the largest PC game platform, Steam. 

There are **83560 unique games** (rows) with their basic information such as name, publisher, language, and platform; marketing information such as description, genres, tags, header image; as well as rating and reviews.

An interesting thing is that only **a very small portion (5%) of the games received a Metacritic score**. Some of this might be due to a mismatch in data scraping. But given that there are only ~6K game ratings on Metacritic, vs. ~83K games in the dataset, it is true that most games have not gained attention of game critics. 

However, not being rated does not necessarily mean that a game is low quality or not worth playing. It might be due to low budget in marketing or the publisher being a smaller profile one. In other words, there could be **hidden gems** in the vast majority of games that flew under the radar.

## Modeling Problem:

Therefore, the goal of the project is to build a model that can **predict Metacritic score**, in the hopes to **find underrated great games**.

## Modeling Strategy:

The **3910 games with ground-truth Metacritic scores** are used for training (validation and tuning) and testing. I will try different classes of models and tune their hyperparameter. The winning model based on test accuracy will be used to predict Metacritic scores on games that don't have it. The accuracy of the predictions should be similar to the accuracy on the test set.

Bonus: I also train a classifier for probability of having Metacritic score. The idea is, predicted score might be reliable for games similar to the training games, but unreliable for games that are more different. The probability measures this similarity. I then labeled games into 4 **confidence tiers**: very low, low, medium, and high, which puts a grain of salt on the interpretation of the scores.


## Data processing: 
- Numerics: There are 15 numerical columns. In all of them, the dominant mode is 0, making the rest of the distribution hard to see/take into effect. So I made two features out of each column. One is a binary indicating whether the value is zero (zero actually means missing data in many of them). The other is the value, with log transformation applied for performance. The transformed values look like these:

<div align="center">
  <img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/log1ps.png" alt="Histograms of numerics" width="800">
</div>
  
- Release date: The only datetime column, which I turned into a numeric feature by subtracting the Epoch (days since 1/1/1970).
- Multi-hot: There are 5 columns containing lists of categorical labels, which can be represented by multi-hot vectors. These columns are categories, genres, tags, languages, and audio languages. Word clouds below. The problem is that there are 732 multi-hot features out of the 5 columns, which is too many for our training data size (732:3910 is roughly 1:5). So I used SVD to reduce dimensionality down to 70 features. This also eliminated colinearity.

<table align="center">
  <tr>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/wordcloud_categories.png" alt="Word cloud of categories" width="400"></td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/wordcloud_genres.png" alt="Word cloud of genres" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/wordcloud_tags.png" alt="Word cloud of tags" width="400"></td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/wordcloud_lang.png" alt="Word cloud of languages" width="400"></td>
  </tr>
</table>

- Other indicators: Other columns such as "support email", "about the game", "notes" and "reviews" are not usable for the model immediately, and they have big portions of missing values. So I simply created indicator (binary) features that =1 if the field is not NA and =0 otherwise.

In total, the training data matrix is 3910 samples by 137 features.
  
## Results: 
For the bonus mission which is to make an imputation confidence measure, I trained a simple (i.e. linear) and stable (i.e. l1 regularized) logistic regression on whether or not a game has a Metacritic score (=1 if has score, =0 otherwise). I made sure the model is well-calibrated, meaning that in all the games that receive a ~0.8 probability by the model, 80% of them have a score, and so on. Because so few games in the dataset have a score (4.7%), this confidence model is very cautious, giving only 4.3% of games p>0.4. This table is how I decided to group games into confidence tiers.

| Predicted p | Interpretation | Percent of games with score in this tier| Number (%) of games without score in this tier|
| :------------:| :-----------------------------------:|:--------------------------:|:--------------:|
| (p > 0.7)    | Very high confidence  | 33.1% | 296 (0.4%) |
| 0.4–0.7      | High confidence       | 27.3% | 897 (1.1%) |
| 0.1–0.4      | Medium confidence     | 29.3% | 3921 (4.9%) |
| <= 0.1        | Low confidence (Extrapolation)  | 10.3% | 74536 (93.6%) |

#### Training:
I trained linear (Elastic net), tree based (XGBoost), and neural network (multi-layer perceptron) regressors. Tuned hyperparameter for each model class on 5-fold CV of 85% data. 
<table align="center">
  <tr>
    <td><h5>Model Class</h5>
    <td><h5>Tuning</h5></td>
    <td><h5>Performance</h5></td>
  </tr>
  <tr>
    <td>Elastic Net</td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/tuning_EN.png" alt="Tuning elastic nets" height="200"></td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/errors_EN.png" alt="Error distribution of the best elastic net model" height="200"></td>
  </tr>
  <tr>
    <td>XGBRegressor</td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/tuning_XGBR.png" alt="Tuning XGBRegressor" height="200"></td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/errors_XGBR.png" alt="Error distribution of the best XGBRegressor model" height="200"></td>
  </tr>
  <tr>
    <td>Multi-layer Perceptron</td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/tuning_MLP.png" alt="Tuning multi-layer perceptron" height="200"></td>
    <td><img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/errors_MLP.png" alt="Error distribution of the best multi-layer perceptron model" height="200"></td>
  </tr>
</table>
As you can see in the error distributions, the models had very similar performances. This suggests the training data has limited predicting power, and is linear enough for the basic model to learn. More complex models cannot reduce the noise in the dataset, i.e. the difference between sample that are not captured by the features.

In the end, the best XGBoost model won over the other two with a very slight advantage on the 15% testing data, so it was used to make ~80K predictions.

You must be wondering, does the model performance really follow the confidence measure? Here is the result. Higher confidence tiers indeed have smaller errors than lower confidence tiers, but only by a little. The difference is visible in the plot but not statistically significant. MLP performance follows confidence the most, even though the confidence model is closer to elastic net. Is there something to it?
<div align="center">
  <img src="https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/images/AE_conf_tier.png" alt="Absolute error by confidence tier" width="800">
</div>

## Try it out on the cloud:

Cloud deployment: 





## Next ideas:
There are some columns containing a lot of information that could not be used directly or by multi-hot encoding. These include description and reviews which are long form text, and header image, screenshots, and movies which are media. In the next step to improve my model, I would use a pretrained neural network to turn the text and media into embeddings, which are vectors (multiple numerical columns) that our models can use. I might use [CLIP](https://github.com/openai/CLIP), which can embed image and text into similar representation.
