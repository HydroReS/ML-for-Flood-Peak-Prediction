############################################################################################################################################
# CONUS # PURPOSE: Re-train Models - 6/23/2023 - Kept Good-performing Regions (10 total across CONUS)
# LOCAL - New LightGBM algorithm -Models trained on 10 hydroclimatic, unique regions in CONUS LOCAL Dataset -
# newly prepped Q-R matching data that uses max precip to select event as trigger event; model includes aridity,
# pet_mean and forest_frac as the only static predictors
# metric_to_optimize = 'mse'; 5000 trials (so larger HP space)
# master_df = master_df[['CatchmentID', 'huc_02', 'P_Trig_max', 'P_Trig_mean', 'API', 'frac_forest',
#                 'P_Trig_Temp_Max_av', 'pet_mean','aridity', 'label', 'Normalized_Peak']]

# CONUS
# IMG_CONUS_01 = {'max_leaf_nodes': 57, 'max_iter': 40, 'l2_regularization': 38.02877518906046, 'learning_rate': 0.40856976142428353, 'validation_fraction': 0.1}#Real original
IMG_CONUS_01 = {'max_leaf_nodes': 58, 'max_iter': 210, 'l2_regularization': 171.09115851447717, 'learning_rate': 0.01106908858532809, 'validation_fraction': 0.1}#Arid-Steppe
IMG_CONUS_02 = {'max_leaf_nodes': 53, 'max_iter': 290, 'l2_regularization': 93.45450573826565, 'learning_rate': 0.49936855484506526, 'validation_fraction': 0.15}
IMG_CONUS_03 = {'max_leaf_nodes': 51, 'max_iter': 20, 'l2_regularization': 73.9668120868638, 'learning_rate': 0.4991047433361446, 'validation_fraction': 0.1}
IMG_CONUS_04 = {'max_leaf_nodes': 12, 'max_iter': 100, 'l2_regularization': 122.54553112815724, 'learning_rate': 3.551274152176884e-05, 'validation_fraction': 0.2}
IMG_CONUS_05 = {'max_leaf_nodes': 3, 'max_iter': 100, 'l2_regularization': 858.0104498430641, 'learning_rate': 0.406928152241928, 'validation_fraction': 0.1}
IMG_CONUS_06 = {'max_leaf_nodes': 54, 'max_iter': 150, 'l2_regularization': 113.31313012579433, 'learning_rate': 0.01955180460103594, 'validation_fraction': 0.2}
IMG_CONUS_07 = {'max_leaf_nodes': 20, 'max_iter': 290, 'l2_regularization': 79.36316473052155, 'learning_rate': 0.4996018533852052, 'validation_fraction': 0.2}
IMG_CONUS_08 = {'max_leaf_nodes': 52, 'max_iter': 280, 'l2_regularization': 168.16769098246593, 'learning_rate': 0.48547947345183945, 'validation_fraction': 0.15}
IMG_CONUS_09 = {'max_leaf_nodes': 27, 'max_iter': 180, 'l2_regularization': 33.74766083878153, 'learning_rate': 0.4056862332323357, 'validation_fraction': 0.15}
IMG_CONUS_10 = {'max_leaf_nodes': 22, 'max_iter': 220, 'l2_regularization': 32.63866711376537, 'learning_rate': 0.025868100959434454, 'validation_fraction': 0.15}

# CH-Models trained on 4 hydroclimatic, unique regions in Switzerland (CH) remotely-sensed datasets; Local Dataset
IMG_CH_01 = {'max_leaf_nodes': 23, 'max_iter': 160, 'l2_regularization': 0.20160054770907987, 'learning_rate': 0.4147069420001154, 'validation_fraction': 0.1}
IMG_CH_02 = {'max_leaf_nodes': 18, 'max_iter': 40, 'l2_regularization': 0.24757609141413184, 'learning_rate': 0.4138981609414564, 'validation_fraction': 0.1}
IMG_CH_03 = {'max_leaf_nodes': 53, 'max_iter': 90, 'l2_regularization': 80.25400096415672, 'learning_rate': 0.4973048336817242, 'validation_fraction': 0.15}
IMG_CH_04 = {'max_leaf_nodes': 35, 'max_iter': 200, 'l2_regularization': 34.92074029282366, 'learning_rate': 0.34552905370825604, 'validation_fraction': 0.2}

best_param_dict = {}
best_param_dict["IMG_CONUS"] = {1:IMG_CONUS_01,2:IMG_CONUS_02,3:IMG_CONUS_03,4:IMG_CONUS_04,5:IMG_CONUS_05,
                                   6:IMG_CONUS_06,7:IMG_CONUS_07,8:IMG_CONUS_08,9:IMG_CONUS_09,10:IMG_CONUS_10}
best_param_dict["IMG_CH_01"] = {1:IMG_CH_01,2:IMG_CH_02,3:IMG_CH_03,4:IMG_CH_04}