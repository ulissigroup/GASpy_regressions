FROM ulissigroup/gaspy:latest

# Reset the default user to `root` so that we can make changes
USER root

# Install new conda packages
RUN conda config --append channels lmmentel
RUN conda config --append channels pytorch
RUN conda config --append channels gpytorch
RUN conda config --append channels plotly
RUN conda install \
    seaborn>=0.9.0 \
    mendeleev>=0.4.3 \
    tensorflow>=1.14.0  tensorflow-gpu>=1.14.0 \
    pytorch>=1.1.0  torchvision>=0.2.1 \
    gpytorch>=0.3.4 \
    skorch>=0.3.0 \
    tpot>=0.9.5 xgboost>=0.80 \
    plotly>=4.1.1 \
    chart-studio>=1.0.0
    
RUN conda clean -lity

# Patch skorch
RUN sed -i '/Dataset\ does\ not\ have consistent\ lengths/s/^/#/g' /miniconda3/lib/python3.6/site-packages/skorch/dataset.py
RUN sed -i '/if\ len(len_set)\ !=\ 1:/s/^/#/g' /miniconda3/lib/python3.6/site-packages/skorch/dataset.py

# Add GASpy_regressions to the python path
ENV PYTHONPATH $GASPY_HOME:$GASPY_HOME/GASpy_regressions

# Set the default user back to `user` so that we can use Shifter
USER user
