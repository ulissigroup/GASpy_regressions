from gaspy_regress import create_volcano_plots
# All volcano plots will be in plotly "2D_plots" folder
# All volcano plots will be dE values by default
# volcano plots: dE(adsorbate2) vs dE(adsorbate1)

# OH vs. CO
adsorbate1 = 'CO'
adsorbate2 = 'OH'
# if you want to change dE's to dG's, you can also change the correction value here:
adsorbate1_correction = 0
adsorbate2_correction = 0
create_volcano_plots.plot_2D_plot(adsorbate1, adsorbate2, adsorbate1_correction, adsorbate2_correction)

# C vs. O
adsorbate1 = 'C'
adsorbate2 = 'O'
# if you want to change dE's to dG's, you can also change the correction value here:
adsorbate1_correction = 0
adsorbate2_correction = 0
create_volcano_plots.plot_2D_plot(adsorbate1, adsorbate2, adsorbate1_correction, adsorbate2_correction)

# O vs. N
adsorbate1 = 'O'
adsorbate2 = 'N'
adsorbate1_correction = 0
adsorbate2_correction = 0
create_volcano_plots.plot_2D_plot(adsorbate1, adsorbate2, adsorbate1_correction, adsorbate2_correction)

# CO vs. H
adsorbate1 = 'CO'
adsorbate2 = 'H'
adsorbate1_correction = 0
adsorbate2_correction = 0
create_volcano_plots.plot_2D_plot(adsorbate1, adsorbate2, adsorbate1_correction, adsorbate2_correction)
