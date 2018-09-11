FROM ubuntu:18.04
LABEL maintainers.owner="Zack Ulissi <zulissi@andrew.cmu.edu>"
LABEL maintainers.maintainer0="Kevin Tran <ktran@andrew.cmu.edu>"
SHELL ["/bin/bash", "-c"]

# Set up a non-root user, `user`, with a group, `group`.
# Necessary to be able to run on Shifter instead of Docker.
ENV HOME=/home
RUN mkdir -p $HOME
RUN groupadd -r group && \
    useradd -r -g group -d $HOME -s /sbin/nologin -c "Default user" user
RUN cp /root/.bashrc $HOME

# Add some things to the .bashrc file to make life easier.
COPY bashrc_additions.sh .
RUN cat bashrc_additions.sh >> /home/.bashrc
RUN rm bashrc_additions.sh

# Install GASpy. Note that we install it by assuming that the user will mount
# their working version of GASpy to the container.
ENV GASPY_HOME=$HOME/GASpy
RUN mkdir -p $GASPY_HOME
ENV PYTHONPATH $GASPY_HOME/GASpy_regressions:$PYTHONPATH
ENV PYTHONPATH $GASPY_HOME:$PYTHONPATH

# Install packages that we need to install other packages.
# Also install libgl1-mesa-glx for matplotlib.
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y wget git libgl1-mesa-glx

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
RUN echo -e "\nyes\n/miniconda3\nyes\n" | /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh
RUN rm Miniconda3-4.5.4-Linux-x86_64.sh $HOME/.bashrc-miniconda3.bak
ENV PATH /miniconda3/bin:$PATH
RUN conda config --prepend channels conda-forge

# Install conda packages.
# deap, update_checker, tqdm, stopit, and xgboost are TPOT dependencies
# nltk, gensim, and g5py are MagPie dependencies
# networkx, cython, pip, and ncurses are... things we needed to fix our image
RUN conda install -c matsci pymatgen fireworks
RUN conda install -c pytorch pytorch torchvision
RUN conda install -c lmmentel mendeleev
RUN conda install numpy scipy pytest pymongo mongodb multiprocess ase jupyter pandas matplotlib==2.2.2 seaborn plotly scikit-learn keras deap update_checker tqdm stopit xgboost nltk gensim h5py networkx cython pip ncurses

# Pip installations go last, because pip often messes things up.
RUN pip install tpot
RUN pip install --no-dependencies git+https://github.com/inspirehep/magpie.git@v2.0
RUN pip install git+https://github.com/rossant/ipycache.git
# Make sure numpy is latest version; something along the way usually drops it down
RUN conda update numpy

# The $GASPY_HOME mount is so that you can use whatever version of GASpy.
# We do this near the end because we can't modify mounted folders after
# declaring them as volumes.
VOLUME $GASPY_HOME

# Make the default user `user` instead of `root`
RUN chown -R user:group $HOME
USER user
