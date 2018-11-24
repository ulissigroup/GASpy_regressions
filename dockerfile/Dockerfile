FROM jupyter/tensorflow-notebook
LABEL maintainers.owner="Zack Ulissi <zulissi@andrew.cmu.edu>"
LABEL maintainers.maintainer0="Kevin Tran <ktran@andrew.cmu.edu>"
SHELL ["/bin/bash", "-c"]

# Install GASpy. Note that we install it by assuming that the user will mount
# their working version of GASpy to the container.
ENV GASPY_HOME=$HOME/GASpy
RUN mkdir -p $GASPY_HOME
ENV PYTHONPATH $GASPY_HOME:$GASPY_HOME/GASpy_regressions

# Install conda packages
# deap, update_checker, tqdm, stopit, and xgboost are TPOT dependencies.
# nltk and gensim are MagPie dependencies.
RUN conda config --prepend channels conda-forge
RUN conda config --append channels matsci
RUN conda config --append channels pytorch
RUN conda config --append channels lmmentel
RUN conda install \
    pytest=3.8.0 \
    mongodb=4.0.2   pymongo=3.7.1 \
    multiprocess=0.70.5 \
    ase=3.16.2 \
    pymatgen=2018.9.1 \
    fireworks=1.7.2 \
    mendeleev=0.4.3 \
    luigi=2.7.8 \
    plotly=3.2.0 \
    pytorch=0.4.1  torchvision=0.2.1 \
    deap=1.2.2  update_checker=0.16 tqdm=4.24.0 stopit=1.1.2 xgboost=0.80 skorch

# Pip installations go last, because pip often messes things up.
RUN pip install tpot
RUN pip install git+https://github.com/rossant/ipycache.git

# Manually install Magpie
RUN git clone https://github.com/ulissigroup/magpie_python.git
ENV PYTHONPATH $PYTHONPATH:$HOME/magpie_python

#Patch skorch
RUN sed -i '/Dataset\ does\ not\ have consistent\ lengths/s/^/#/g' /opt/conda/lib/python3.6/site-packages/skorch/dataset.py
RUN sed -i '/if\ len(len_set)\ !=\ 1:/s/^/#/g' /opt/conda/lib/python3.6/site-packages/skorch/dataset.py

# The $GASPY_HOME mount is so that you can use whatever version of GASpy.
# We do this near the end because we can't modify mounted folders after
# declaring them as volumes.
VOLUME $GASPY_HOME
