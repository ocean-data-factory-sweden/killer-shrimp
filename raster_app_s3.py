import json, os
import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import rasterstats
import matplotlib.pyplot as plt
import pydeck as pdk
import rasterio
from rasterio.features import shapes
from rasterio.plot import show, show_hist, plotting_extent
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyimpute import load_training_vector, load_targets, impute, evaluate_clf
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from yarl import URL
import boto3
import urllib.request

placeholder = st.empty()
placeholder_1 = st.empty()
placeholder.title('ODF Suitability Modelling - D. Villosus')
client = boto3.client("s3")

data_path = URL('https://odf-open-data.s3.eu-north-1.amazonaws.com/data')
models = {"RandomForest": RandomForestClassifier(1, n_jobs=8),
          "BalancedForest": BalancedRandomForestClassifier(10)}

current_rasters = [str(URL(data_path/'current_rasters' / 'Absence_Salinity_today_gps.tif')),
          str(URL(data_path/'current_rasters'/ 'Absence_Temperature_today_gps.tif')),
          str(URL(data_path/'current_rasters'/ 'Absence_Substrate_gps.tif')),
          str(URL(data_path/'current_rasters'/ 'Absence_Depth_gps.tif')),
         str(URL(data_path/'current_rasters'/'Absence_Exposure_gps.tif'))]
future_rasters = [str(URL(data_path/'future_rasters'/ 'Absence_Salinity_ClimateChange_gps.tif')),
          str(URL(data_path/'future_rasters'/ 'Absence_Temperature_ClimateChange_gps.tif')),
          str(URL(data_path/'future_rasters'/ 'Absence_Substrate_gps.tif')),
          str(URL(data_path/'future_rasters'/ 'Absence_Depth_gps.tif')),
         str(URL(data_path/'future_rasters'/ 'Absence_Exposure_gps.tif'))]

def main():
    scenario = st.sidebar.selectbox("Choose scenario", ["Current climate", "Future climate"])
    model = st.sidebar.selectbox("Choose model", "DummyClassifier",
    "DecisionTree", "RandomForest"])
    st.sidebar.markdown('Click to run model')
    submit = st.sidebar.button('Train and Plot')
    if submit and scenario == "Current climate":
        return run_model(model, name='current', rasters=current_rasters)
    elif submit and scenario == "Future climate":
        return run_model(model, name='future', rasters=future_rasters)

def gen_splits():
    # data
    absence_data = pd.read_csv(str(URL(data_path/ "point_sampler"/ "absence"/ "Absence_filter_Nodata_mod100.csv"),
                               sep=';', index_col=0))
    presence_data = pd.read_csv(str(URL(data_path/ "point_sampler"/ "presence"/
                                    "Presence_baltic_manual_fill_missing_data.csv"), sep=';', index_col=0))
    absence_data['obs'] = 0; presence_data['obs'] = 1
    data = absence_data.append(presence_data).reset_index()
    data = data[~data.eq(-9999.0).any(1)]
    X = data[[i for i in data.columns if i != 'obs' and i != 'pointid']]; y = data['obs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rus = SMOTENC(categorical_features=[2], random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test

def fit_eval(model, X_train, y_train, X_test, y_test):
    m = model
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    #fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)
    #cm = confusion_matrix(y_test, pred)
    #plot_confusion_matrix(cm, title=f'Model: {type(m).__name__}', target_names=['Absent','Present'], normalize=False)
    return m

def run_model(model, name, rasters=current_rasters):
    placeholder.title("You chose {:s} climate rasters and {:s}".format(name, model))

    #st.image(np.where(masking, -0.1, res.read(1, masked=True)))
    plotit(model, name, 'something')
    #plotit(np.where(masking, -0.1, res.read(1, masked=True)), 'D. Villosus Suitability', cmap='GnBu')


def plotit(model, name, title, cmap='Blues'):

    placeholder.header("Loading trained model...")

    file_path = str(URL(data_path / "models" / f'data_{model}_{name}.json'))

    file = urllib.request.urlopen(file_path)
    response = file.read().decode('utf-8')
    data = json.loads(response)

    geojson = pdk.Layer(
        'GeoJsonLayer',
        data,
        opacity=0.5,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation='properties.raster_val * 20000',
        get_fill_color='[255 * properties.raster_val, 0, 0]',
        #get_line_color=[255, 255, 255],
        pickable=True
        )
    INITIAL_VIEW_STATE = pdk.ViewState(
      latitude=58.49,
      longitude=19.86,
      zoom=5,
      max_zoom=16,
      min_zoom=5,
      pitch=45,
      bearing=0
    )


    placeholder.title("Suitability Map")
    r = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9",layers=[geojson],
                initial_view_state=INITIAL_VIEW_STATE)
    placeholder.empty()
    placeholder_1.empty()
    st.write(r) # == st.deck_json_chart



if __name__ == "__main__":
    main()
