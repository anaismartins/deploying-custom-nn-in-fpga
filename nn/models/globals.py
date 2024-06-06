DETECTORS = ["H1", "L1", "V1"]

# directories
# DATA_DIR = "./../../../../scratch/melissa.lopez/Projects/EarlyWarning/FreqCL_O3/" # CIT cluster
DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data/FreqCL_O3/"  # nikhef cluster
# DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O4" # gaussian O4
TRAIN_DATA_DIR = (
    "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O3/"
)
VALID_DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O3_validation/"
TEST_DATA_DIR = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data_new/FreqCL_O3_test/"

OUTPUT_DIR = "/data/gravwav/amartins/earlywarningfpgas/output/"
MODEL_PATH = "/home/anaismartins/Desktop/earlywarningfpgas/output/models/50.0_curriculum_ModifiedPaperNet_3epochs_10samples_8e-05lr_50_minibatch_AdamaxGregoptimizer_0.16decayfactor_128patience_0.01early_stopping_delta_128patience.pt"

MODEL = "GregNet"
# MODEL = "GWaveNet2DModified"
FILENAME_ADD = ""

# main training parameters
FOLDER = 4
MINIBATCH_SIZE = 50

if MODEL == "GregNet" or MODEL == "GregNet2D" or MODEL == "GregNet2DModified":
    EPOCHS = 6
    TEST_MINIBATCH_SIZE = 50
    LEARNING_RATE = 8e-5
    OPTIMIZER_NAME = "AdamaxGreg"
    EARLY_STOPPING_PATIENCE = 6

elif MODEL == "GWaveNet" or MODEL == "GWaveNet2D" or MODEL == "GWaveNet2DModified":
    EPOCHS = 20
    TEST_MINIBATCH_SIZE = 200
    LEARNING_RATE = 1e-5
    OPTIMIZER_NAME = "AdamW"
    EARLY_STOPPING_PATIENCE = 2

FILTERS = 64
KERNEL_SIZE_CONVOLUTION = 16
DILATION = 2
STRIDE = 2

EARLY_STOPPING_MIN_DELTA = 0.0001

# colors
UU_YELLOW = "#FFCD00"
UU_BLACK = "#000000"
UU_CREAM = "#FFE6AB"
UU_ORANGE = "#F3965E"
