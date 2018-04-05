# Utah Vehicle Accidents Prediction
> Can we predict car accident risk on individual road segments in Utah? Let's do our best using 
> the power of ArcGIS and Python!

For questions about these notebooks, or to obtain the data needed to run them, please contact
[Daniel Wilson](mailto:dwilson@esri.com)

## Approach
We are approaching this from a **supervised machine learning** perspective. Using a training set of
hours where accidents occurred, augmented by a sample of times/roads where they didn't, we are 
training a [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) model with a 
cross-entropy loss function (logistic regression) to estimate the accident risk. Note that I say 
risk - we are not estimating the probability directly, but the risk is *proportional* to the 
probability. This issue stems from class imbalance: There are about 400k accidents, but when 
looking at all the segments and times where they didn't occur, we're now talking about 200M+ 
training examples. That's what we call a severe class imbalance issue. We keep the training set
slightly imbalanced, and use an informative sampling approach to get the model to learn the fine
grained differences between what causes a car accident or not.

## The Data
We have 7 years from car crash records for the state of Utah, but that's only the start of our
training set for this supervised machine learning problem. We use many different features to try
to differentiate between car accidents occuring or not. These include:

* Road Features  
  * Curvature (Sinuosity)
  * Orientation (East/West vs North/South)
  * Speed Limit
  * Surface Type
  * Surface Width
  * Annual Average Daily Traffic (AADT)
  * Proximity to nearest intersection
  * Proximity to nearest signal
  * Proximity to nearest major road
  * Proximity to nearest billboard
  * Population density
  * Historical accident counts
* Weather Features
  * Temperature
  * Visibility
  * Precipitation/Snow Depth
  * Icy
  * Raining
  * Snowing
  * Hailing
  * Thunderstorming
* Temporal Features
  * Solar Azimuth/Elevation
  * Time of day
  * Day of the week
  * Month

There are more that we could add, and likely will as we refine this process. 

We use [ArcGIS Pro](http://pro.arcgis.com/en/pro-app/) and 
[Arcpy](http://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-arcpy-.htm) to perform some 
geoprocessing, followed by the [ArcGIS API for Python](https://developers.arcgis.com/python/),
[Pandas](https://pandas.pydata.org/), and [scikit-learn](http://scikit-learn.org/) to prep the data.
We also use some miscellaneous libraries along the way, such as 
[Astral](https://astral.readthedocs.io/en/stable/index.html) for solar geometry computation.

The weather data is pulled from the [NOAA's](https://www.ncdc.noaa.gov/data-access/quick-links) 
[API](https://www.ncdc.noaa.gov/access-data-service/api/v1/data) and aggregated into hourly bins per
weather station. Currently, we chose to snap each road to the nearest weather station. We realize
this is a poor approach, and will implement interpolation/regression to accurately asses weather
conditions for each road. We would also like to bring in better weather sources, as well as real time
weather forecasting as we build this into a more realtime application for situational awareness.

Any categorical variables are one-hot encoded using panda's `pd.get_dummies` and all continuous 
variables are converted to a z-score using `scikit-learn`'s `StandardScaler`. 

## The Model
We use [XGBoost](http://xgboost.readthedocs.io/en/latest/) library for the supervised machine
learning model. We also experimented with Deep Neural Networks, but found gradient boosting to work
better. The model is trained using a `max_depth` of 6, `min_child_weight` of 5.0, and a L2 penalty term 
(`reg_lambda`) of 1.0. We train until completion of best accuracy on a validation set with 25 rounds of 
early stopping. 

## The Notebooks

* prep_static_features.ipynb - ArcPy preparation of the road features
* get_weather.ipynb - pulls/transforms the NOAA weather data
* prepare_training_data.ipynb - builds a training set CSV file
* train_model.ipynb - trains the model and plots results.



