import optuna
import optuna.visualization as ov
import pickle
import plotly.io as pio

# Load the study object for features-EH
with open('hyper/optuna_study_feature.pkl', 'rb') as f:
    study = pickle.load(f)

# 1. Contour plot for ResNet models
fig1 = ov.plot_contour(study, params=['learning_rate', 'weight_decay'])
pio.write_image(fig1, 'hyper/plot/contour_plot_ResNet_EH.pdf')

fig2 = ov.plot_contour(study, params=['learning_rate', 'weight_decay'])
pio.write_image(fig2, 'hyper/plot/contour_plot_ResNet_AU.pdf')

# 2. Hyperparameter importance plot for EfficientNet models
# fig3 = ov.plot_param_importances(study)
# pio.write_image(fig3, 'hyper/plot/param_importance_plot_EfficientNet_EH.pdf')

# fig4 = ov.plot_param_importances(study)
# pio.write_image(fig4, 'hyper/plot/param_importance_plot_EfficientNet_AU.pdf')

# 3. Parallel coordinate plot for all models
fig5 = ov.plot_parallel_coordinate(study, params=['batch_size', 'learning_rate', 'dropout_rate', 'weight_decay'])
pio.write_image(fig5, 'hyper/plot/parallel_coordinate_plot_feature.pdf')

# Load the study object for IF_IFOF
with open('hyper/optuna_study.pkl', 'rb') as f:
    study = pickle.load(f)

# 1. Contour plot for ResNet models
fig6 = ov.plot_contour(study, params=['learning_rate', 'weight_decay'])
pio.write_image(fig6, 'hyper/plot/contour_plot_ResNet_IF.pdf')

fig7 = ov.plot_contour(study, params=['learning_rate', 'weight_decay'])
pio.write_image(fig7, 'hyper/plot/contour_plot_ResNet_IFOF.pdf')

# 2. Hyperparameter importance plot for EfficientNet models
# fig8 = ov.plot_param_importances(study)
# pio.write_image(fig8, 'hyper/plot/param_importance_plot_EfficientNet_IF.pdf')

# fig9 = ov.plot_param_importances(study)
# pio.write_image(fig9, 'hyper/plot/param_importance_plot_EfficientNet_IFOF.pdf')

# 3. Parallel coordinate plot for all models
fig10 = ov.plot_parallel_coordinate(study, params=['batch_size', 'learning_rate', 'dropout_rate', 'weight_decay'])
pio.write_image(fig10, '~/Student_Engagement/Results/DL/parallel_coordinate_plot_feature_IF_IFOF.pdf')
