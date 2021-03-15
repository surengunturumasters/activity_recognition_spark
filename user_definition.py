# feature extraction and preprocess variables and data folders
activity_string = 'walking'
database = 'postgres'
eating_strings = ["eating"]
endpoint = 'msds694.cmxsootjz10m.us-west-2.rds.amazonaws.com'
files = './WISDM/*/*/'
n = 10
properties = {'user': 'students', 'password': 'msdsstudents'}
subject_id = 1600
table = 'activity_code'
url = 'jdbc:postgresql://%s/%s' % (endpoint, database)
window_size = 5

# ML variables and data folders
train_folder = 'input_2/train'
valid_folder = 'input_2/valid'
n_fold = 5
max_iter = [100]
reg_params = [0.001]
num_trees = [10,100]
max_depth = [5,10]
n_digits = 4
seed = 2
max_runtime_secs = 1500


def retrive_file_name(x):
    """Returns subject_id, sensor, device and an arry of readings"""
    file_name = x[0].split("/")[-1].split(".txt")[0]
    file_arg = file_name.split("_")
    return (file_arg[1], file_arg[2], file_arg[3], x[1])


def convert_to_integer(x):
    """Convert a value to integer"""
    try:
        return int(x)
    except ValueError:
        return None


def convert_to_float(x):
    """Convert a value to float"""
    try:
        return float(x)
    except ValueError:
        return None


def conver_to_string(x):
    """Convert a value to string"""
    try:
        return str(x)
    except ValueError:
        return None


def check_same_user(x):
    """
    Return subject_id in the file name
    that is same as subject_id in the content.
    """
    if (x is not None and x[0] == x[3]):
        return (x[0], x[1], x[2], x[4], x[5], x[6], x[7], x[8])


def return_no_none_rows(x):
    """Return True if all the readings are not None"""
    if (x is not None and
            x[0] is not None and x[1] is not None and x[1] is not None and
            x[2] is not None and x[3] is not None and x[4] is not None and
            x[5] is not None and x[6] is not None and x[7] is not None):
        # if(x[5] == 0 or x[6] == 0 or x[7] == 0):
        return True
    else:
        return False


def create_flat_rdd(x):
    """
    Returns subject_id, sensor, device and
    subject_id, activity_code, x, y, z readings
    """
    values = x[3].split(",")
    if len(values) == 6:
        return (convert_to_integer(x[0]),
                x[1],
                x[2],
                convert_to_integer(values[0]),
                conver_to_string(values[1]),
                convert_to_integer(values[2]),
                convert_to_float(values[3]),
                convert_to_float(values[4]),
                convert_to_float(values[5]))


def file_rdd(ss, files):
    """Create a pair RDD using wholeTextFiles"""
    return ss.sparkContext.wholeTextFiles(files)


def create_activity_df(ss, files_rdd, schema):
    """Create dataframe using the schema"""
    activity_data_rdd = files_rdd.mapValues(lambda x: x.split(";\n"))\
        .flatMapValues(lambda x: x)\
        .map(retrive_file_name)\
        .map(create_flat_rdd)\
        .map(check_same_user)\
        .filter(return_no_none_rows)

    return ss.createDataFrame(activity_data_rdd, schema)



