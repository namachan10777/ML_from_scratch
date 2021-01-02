DATASET = dataset

# winequality
WINE_DATASETS = winequality-red.csv winequality-white.csv
WINE_HOSTINGS = https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality
WINE_DIR=$(DATASET)/wine_quality
WINE_TARGETS = $(addprefix $(WINE_DIR)/,$(WINE_DATASETS))

# mnist
MNIST = \
	train-images-idx3-ubyte \
	train-labels-idx1-ubyte \
	t10k-images-idx3-ubyte \
	t10k-labels-idx1-ubyte
MNIST_DIR = $(DATASET)/mnist
MNIST_TARGETS = $(addprefix $(MNIST_DIR)/,$(MNIST))
MNIST_GZ = $(addsuffix .gz,$(MNIST_TARGETS))
MNIST_HOSTINGS = http://yann.lecun.com/exdb/mnist

IRIS_DIR=$(DATASET)/iris
IRIS_TARGETS=$(IRIS_DIR)/iris.data
IRIS_HOSTING=https://archive.ics.uci.edu/ml/machine-learning-databases/iris

.PHONY: all
all: $(WINE_TARGETS) $(MNIST_TARGETS) $(IRIS_TARGETS)

.PHONY: clean
clean:
	rm -r $(WINE_DIR)

$(WINE_DIR)/%:
	mkdir -p $(DATASET)
	mkdir -p $(WINE_DIR)
	cd $(WINE_DIR) && wget $(patsubst $(WINE_DIR)/%,$(WINE_HOSTINGS)/%,$@)

$(MNIST_DIR)/%.gz:
	mkdir -p $(DATASET)
	mkdir -p $(MNIST_DIR)
	cd $(MNIST_DIR) && wget $(patsubst $(MNIST_DIR)/%,$(MNIST_HOSTINGS)/%,$@)

$(MNIST_DIR)/%: $(MNIST_DIR)/%.gz
	gzip -d $<

$(IRIS_DIR)/%:
	mkdir -p $(IRIS_DIR)
	cd $(IRIS_DIR) && wget $(patsubst $(IRIS_DIR)/%,$(IRIS_HOSTING)/%,$@)
