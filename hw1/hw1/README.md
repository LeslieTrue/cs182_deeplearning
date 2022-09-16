In this assignment you will practice writing backpropagation code and training
Neural Networks. The goals of this assignment
are as follows:

- understand **Neural Networks** and how they are arranged in layered
  architectures
- understand and be able to implement (vectorized) **backpropagation**

## Setup
Make sure your machine is set up with the assignment dependencies.

**[Option 1] Use Anaconda:**
The preferred approach for installing all the assignment dependencies is to use
[Anaconda](https://www.continuum.io/downloads), which is a Python distribution
that includes many of the most popular Python packages for science, math,
engineering and data analysis. Once you install it you can skip all mentions of
requirements and you are ready to go directly to working on the assignment.

**[Option 2] Manual install, virtual environment:**
If you do not want to use Anaconda and want to go with a more manual and risky
installation route you will likely want to create a
[virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)
for the project. If you choose not to use a virtual environment, it is up to you
to make sure that all dependencies for the code are installed globally on your
machine. To set up a virtual environment, run the following:

```bash
cd hw1
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
```

**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `hw1` directory:

```bash
cd deeplearning/datasets
./get_datasets.sh
```

If you are on Mac, this script may not work if you do not have the wget command
installed, but you can use curl instead with the alternative script.
```bash
cd deeplearning/datasets
./get_datasets_curl.sh
```

**Start IPython:**
After you have the CIFAR-10 data, you should start the IPython notebook server
from the `hw1` directory.

**NOTE:** If you are working in a virtual environment on OSX, you may encounter
errors with matplotlib due to the
[issues described here](http://matplotlib.org/faq/virtualenv_faq.html).
You can work around this issue by starting the IPython server using the
`start_ipython_osx.sh` script from the `hw1` directory; the script
assumes that your virtual environment is named `.env`.

### Fully-connected Neural Network
The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

If you use Colab for this notebook, make sure to manually download the completed
notebook and place it in the assignment directory before submitting. Also remember
to download required output file and place it into submission_logs/ directory.
